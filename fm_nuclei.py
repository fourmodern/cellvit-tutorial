"""H-optimus-0 하나로 세포핵 검출·분할·임베딩을 하는 파이프라인의 공통 모듈 (CellViT 등 다른 분할 모델에 의존하지 않음).

- NucleiDecoder : 고정된 H-optimus-0의 중간 특징 4개 + RGB → 픽셀 단위 3클래스(배경/핵 내부/경계) + 중심 열지도
- make_targets   : 인스턴스 마스크 → 학습 타깃
- instances_from_maps : 예측 맵 → 인스턴스 라벨 맵 (경계 제거 + watershed)
- detection_f1 / dice : 평가
- RoiRunner      : 큰 영역을 224 타일(절반 겹침)로 훑어 분할 맵 + 토큰 맵을 한 번에 만들기

입력 해상도는 모델 기본값인 0.5 µm/px (224 px = 112 µm). 라이선스: H-optimus-0 Apache 2.0, 이 코드 MIT.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

TOK = 14            # H-optimus-0 패치 토큰 한 변 (px, 0.5 µm/px)
TILE = 224
LAYERS = (9, 19, 29, 39)   # 중간 특징을 꺼낼 블록 (40층 중)


# ─────────────────────────────────────────────────────────────
# 1. 디코더
# ─────────────────────────────────────────────────────────────
class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class NucleiDecoder(nn.Module):
    """UNETR 축약판. 입력: ViT 중간 특징 4개 (B, D, 16, 16) + 원본 RGB (B, 3, 224, 224)
    출력: (B, 4, 224, 224) = [배경, 핵 내부, 경계] 로짓 3채널 + 중심 열지도 1채널
    16×16 → 28 → 56 → 112 → 224 로 올리며 RGB에서 뽑은 얕은 특징을 112·224 해상도에서 skip으로 합친다."""

    def __init__(self, embed_dim=1536, width=192, n_feats=4):
        super().__init__()
        self.reduce = nn.ModuleList([nn.Sequential(nn.Conv2d(embed_dim, width, 1, bias=False), nn.BatchNorm2d(width), nn.GELU())
                                     for _ in range(n_feats)])
        self.fuse = ConvBlock(width * n_feats, width)
        self.up1 = ConvBlock(width, 128)          # 28
        self.up2 = ConvBlock(128, 96)             # 56
        self.img112 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(), ConvBlock(32, 32))   # 224 → 112
        self.up3 = ConvBlock(96 + 32, 64)         # 112
        self.img224 = ConvBlock(3, 16)
        self.up4 = ConvBlock(64 + 16, 48)         # 224
        self.head = nn.Conv2d(48, 4, 1)

    def forward(self, feats, img):
        x = torch.cat([r(f) for r, f in zip(self.reduce, feats)], 1)
        x = self.fuse(x)                                                     # 16
        x = self.up1(F.interpolate(x, size=28, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))   # 56
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)              # 112
        x = self.up3(torch.cat([x, self.img112(img)], 1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)              # 224
        x = self.up4(torch.cat([x, self.img224(img)], 1))
        return self.head(x)


@torch.inference_mode()
def backbone_features(backbone, x):
    """고정 backbone 한 번 통과 → (중간 특징 4개 [B, D, 16, 16], 마지막 층 패치 토큰 [B, 16, 16, D])"""
    final, inter = backbone.forward_intermediates(x, indices=list(LAYERS), return_prefix_tokens=False, norm=True,
                                                  output_fmt="NCHW", intermediates_only=False)
    tokens = final[:, backbone.num_prefix_tokens:].reshape(x.shape[0], 16, 16, -1)
    return [f.float() for f in inter], tokens.float()


# ─────────────────────────────────────────────────────────────
# 1b. 염색 증강
# ─────────────────────────────────────────────────────────────
def stain_jitter(img: np.ndarray, rng: np.random.Generator, p_hed: float = 0.7, p_gamma: float = 0.5):
    """H&E 염색 차이를 흉내내는 증강: HED 색공간에서 헤마톡실린·에오신 농도를 0.6~1.4배로 바꾸고 감마를 흔든다.
    (학습 데이터보다 옅거나 진하게 염색된 외부 슬라이드에 대한 강건성이 크게 좋아진다)"""
    from skimage.color import hed2rgb, rgb2hed
    if rng.random() < p_hed:
        hed = rgb2hed(img)
        hed[..., 0] *= rng.uniform(0.6, 1.4); hed[..., 1] *= rng.uniform(0.6, 1.4)
        hed[..., 0] += rng.uniform(-0.03, 0.03); hed[..., 1] += rng.uniform(-0.03, 0.03)
        img = (np.clip(hed2rgb(hed), 0, 1) * 255).astype(np.uint8)
    if rng.random() < p_gamma:
        img = (255 * (img / 255.0) ** rng.uniform(0.7, 1.4)).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────
# 2. 학습 타깃
# ─────────────────────────────────────────────────────────────
def make_targets(inst: np.ndarray, sigma: float = 2.0, boundary_px: int = 1):
    """인스턴스 라벨 맵 (H, W, int) → (클래스 맵 [0 배경 / 1 내부 / 2 경계], 중심 열지도 [0~1])"""
    h, w = inst.shape
    fg = inst > 0
    # 경계: 자기 인스턴스를 침식했을 때 사라지는 픽셀 (인접한 다른 핵과의 경계가 두껍게 잡힘)
    eroded = np.zeros_like(fg)
    for k in np.unique(inst):
        if k == 0:
            continue
        m = inst == k
        eroded |= ndi.binary_erosion(m, iterations=boundary_px, border_value=0)
    cls = np.zeros((h, w), np.int64)
    cls[fg] = 2
    cls[eroded] = 1
    heat = np.zeros((h, w), np.float32)
    ids = np.unique(inst); ids = ids[ids > 0]
    if len(ids):
        cy, cx = zip(*ndi.center_of_mass(fg, inst, ids))
        yy, xx = np.mgrid[:h, :w]
        for y0, x0 in zip(cy, cx):
            r0, r1 = max(int(y0 - 4 * sigma), 0), min(int(y0 + 4 * sigma) + 1, h)
            c0, c1 = max(int(x0 - 4 * sigma), 0), min(int(x0 + 4 * sigma) + 1, w)
            g = np.exp(-((yy[r0:r1, c0:c1] - y0) ** 2 + (xx[r0:r1, c0:c1] - x0) ** 2) / (2 * sigma ** 2))
            heat[r0:r1, c0:c1] = np.maximum(heat[r0:r1, c0:c1], g)
    return cls, heat


def seg_loss(out, cls, heat):
    """3클래스 CE + 핵 내부 Dice + 열지도 MSE(핵 근처 가중)"""
    logits, hm = out[:, :3], out[:, 3]
    ce = F.cross_entropy(logits, cls)
    p = F.softmax(logits, 1)[:, 1]
    t = (cls == 1).float()
    dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    w = 1 + 4 * heat
    mse = ((hm - heat) ** 2 * w).mean()
    return ce + dice.mean() + 5 * mse


# ─────────────────────────────────────────────────────────────
# 3. 예측 맵 → 인스턴스
# ─────────────────────────────────────────────────────────────
def instances_from_maps(prob: np.ndarray, heat: np.ndarray, min_area: int = 12, heat_thr: float = 0.3):
    """prob: (3, H, W) softmax [배경, 내부, 경계], heat: (H, W).
    핵 전경 = 내부+경계 > 배경. 마커 = '내부' 연결 성분 (경계 채널이 붙은 핵을 떼어 줌) → 전경 안에서 watershed."""
    fg = (prob[1] + prob[2]) > prob[0]
    inside = (prob[1] > prob[0]) & (prob[1] > prob[2])
    markers, _ = ndi.label(inside)
    # 내부가 너무 작아 마커가 안 생긴 핵은 열지도 피크로 보충
    peaks = (heat > heat_thr) & (heat == ndi.maximum_filter(heat, size=5)) & fg & (markers == 0)
    pk, npk = ndi.label(peaks)
    markers[pk > 0] = pk[pk > 0] + markers.max()
    dist = ndi.distance_transform_edt(fg)
    lab = watershed(-dist, markers=markers, mask=fg)
    # 작은 조각 제거
    sizes = np.bincount(lab.ravel())
    small = np.flatnonzero(sizes < min_area); small = small[small > 0]
    if len(small):
        lab[np.isin(lab, small)] = 0
    return lab


def instance_props(lab: np.ndarray):
    """라벨 맵 → [(id, centroid(x, y), area)]"""
    ids = np.unique(lab); ids = ids[ids > 0]
    if not len(ids):
        return []
    cents = ndi.center_of_mass(lab > 0, lab, ids)
    areas = ndi.sum(np.ones_like(lab), lab, ids)
    return [(int(i), (float(c[1]), float(c[0])), float(a)) for i, c, a in zip(ids, cents, areas)]


def contours_from_labels(lab: np.ndarray):
    """라벨 맵 → {id: contour (N, 2) x,y}"""
    import cv2
    out = {}
    for i, _, _ in instance_props(lab):
        m = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = max(cs, key=cv2.contourArea).reshape(-1, 2)
            if len(c) >= 3:
                out[i] = c.astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────
# 4. 평가
# ─────────────────────────────────────────────────────────────
def detection_f1(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float):
    """중심점 매칭 (헝가리안, 거리 ≤ radius) → precision, recall, F1"""
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return 0.0, 0.0, 0.0
    d = np.linalg.norm(pred_xy[:, None] - gt_xy[None], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= radius, d, 1e6))
    tp = int((d[r, c] <= radius).sum())
    prec, rec = tp / len(pred_xy), tp / len(gt_xy)
    return prec, rec, (2 * prec * rec / (prec + rec) if tp else 0.0)


def dice(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


# ─────────────────────────────────────────────────────────────
# 5. 큰 영역 처리: 분할 맵 + 토큰 맵을 한 번의 훑기로
# ─────────────────────────────────────────────────────────────
class RoiRunner:
    """0.5 µm/px 이미지를 224 타일, 112 간격(절반 겹침)으로 훑고 각 타일의 가운데 112×112(분할) / 8×8 토큰(임베딩)만 기록"""

    def __init__(self, backbone, decoder, transform, device="cuda", batch=16):
        self.b, self.d, self.tf, self.dev, self.batch = backbone, decoder, transform, device, batch

    @torch.inference_mode()
    def run(self, img05: np.ndarray):
        from PIL import Image
        h, w = img05.shape[:2]
        gh, gw = int(np.ceil(h / TOK)), int(np.ceil(w / TOK))
        H_, W_ = gh * TOK, gw * TOK
        pad = TILE // 4   # 56
        canvas = np.full((H_ + 2 * pad + TILE, W_ + 2 * pad + TILE, 3), 255, np.uint8)
        canvas[pad:pad + h, pad:pad + w] = img05
        prob = np.zeros((3, H_, W_), np.float32); heat = np.zeros((H_, W_), np.float32)
        fmap = np.zeros((gh, gw, self.b.num_features), np.float16)
        pos = [(y, x) for y in range(0, H_, TILE // 2) for x in range(0, W_, TILE // 2)]
        self.d.eval()
        for i in range(0, len(pos), self.batch):
            bp = pos[i:i + self.batch]
            tiles = [canvas[y:y + TILE, x:x + TILE] for y, x in bp]
            x = torch.stack([self.tf(Image.fromarray(t)) for t in tiles]).to(self.dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                feats, tok = backbone_features(self.b, x)
                out = self.d(feats, x).float()
            sm = F.softmax(out[:, :3], 1).cpu().numpy(); hm = out[:, 3].clamp(0, 1).cpu().numpy()
            tok = tok.cpu().numpy()
            for (y, x0), s_, h_, t_ in zip(bp, sm, hm, tok):
                c0 = TILE // 4; c1 = c0 + TILE // 2                           # 타일 가운데 112×112
                ny, nx = min(TILE // 2, H_ - y), min(TILE // 2, W_ - x0)
                prob[:, y:y + ny, x0:x0 + nx] = s_[:, c0:c0 + ny, c0:c0 + nx]
                heat[y:y + ny, x0:x0 + nx] = h_[c0:c0 + ny, c0:c0 + nx]
                gy, gx = y // TOK, x0 // TOK
                ty, tx = min(8, gh - gy), min(8, gw - gx)
                fmap[gy:gy + ty, gx:gx + tx] = t_[4:4 + ty, 4:4 + tx]
        return prob[:, :h, :w], heat[:h, :w], fmap


def mask_pool(fmap: np.ndarray, lab: np.ndarray, dilate_px: int = 0):
    """토큰 맵 (gh, gw, D)과 인스턴스 라벨 맵 (H, W; 0.5 µm/px) → 인스턴스별 토큰 가중 평균 {id: (D,)}
    각 핵이 덮는 픽셀을 토큰 칸(14×14)별로 세어 가중치로 쓴다."""
    gh, gw, D = fmap.shape
    H, W = lab.shape
    lab_ = lab
    if dilate_px > 0:
        # 라벨별 팽창(겹치면 원래 라벨 우선): 간단히 최대 필터로 빈 곳만 채운다
        grown = ndi.grey_dilation(lab, size=(2 * dilate_px + 1, 2 * dilate_px + 1))
        lab_ = np.where(lab > 0, lab, grown)
    ids = np.unique(lab_); ids = ids[ids > 0]
    ty = (np.arange(H) // TOK).clip(0, gh - 1); tx = (np.arange(W) // TOK).clip(0, gw - 1)
    tok_index = (ty[:, None] * gw + tx[None, :]).ravel()
    flat = lab_.ravel()
    out = {}
    fm = fmap.reshape(-1, D).astype(np.float32)
    # 인스턴스별 (토큰 인덱스 → 픽셀 수) 히스토그램
    order = np.argsort(flat, kind="stable")
    sorted_lab = flat[order]
    starts = np.searchsorted(sorted_lab, ids, "left"); ends = np.searchsorted(sorted_lab, ids, "right")
    for i, s, e in zip(ids, starts, ends):
        ti = tok_index[order[s:e]]
        u, cnt = np.unique(ti, return_counts=True)
        w = cnt / cnt.sum()
        out[int(i)] = (fm[u] * w[:, None]).sum(0)
    return out
