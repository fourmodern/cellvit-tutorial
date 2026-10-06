"""H-optimus-0 단일 backbone 세포 파이프라인 노트북 생성 스크립트 (CellViT 등 외부 분할 모델 없음).

python build_single.py  →  10_single_backbone_pipeline.ipynb
"""
import textwrap

import nbformat as nbf

REPO = "fourmodern/cellvit-tutorial"
NB = "10_single_backbone_pipeline.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(textwrap.dedent(t).strip())


def code(t):
    return nbf.v4.new_code_cell(textwrap.dedent(t).strip())


C = []
C.append(md(f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/{NB}" target="_parent">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'))
C.append(md(r'''
# Backbone 하나로 끝내는 세포 파이프라인: H-optimus-0 검출·분할·임베딩 (CellViT 없음)

09 노트북에서는 세포 **위치**를 CellViT에서 받고 **임베딩**은 H-optimus-0 토큰 맵에서 꺼냈습니다.
이 노트북은 CellViT를 완전히 빼고, **H-optimus-0 하나**로 세포핵 검출·분할·임베딩을 모두 합니다.

```
                         ┌─ 중간 특징 4개 (블록 10·20·30·40, 각 16×16×1536) ─► [작은 디코더 3.6M, 직접 학습] ─► 핵 분할 (배경/내부/경계 + 중심 열지도)
H&E 타일 ─► H-optimus-0 ─┤                                                                                      │
(0.5 µm/px)   (고정)      └─ 마지막 층 토큰 (16×16×1536) ───────────────────────── 핵 마스크로 토큰 가중 평균 ◄──┘ ─► 세포 임베딩
```

**타일 한 번 통과로 분할과 임베딩이 동시에** 나옵니다. 학습하는 것은 디코더(3.6M 파라미터)뿐이고, backbone은 고정합니다.

### 왜 이렇게 만드나

| 요구 | 이 설계 |
|---|---|
| **라이선스**: 상업적 이용 가능 | H-optimus-0 **Apache 2.0** · 학습 데이터 NuInsSeg **CC BY 4.0** · TNBC **CC BY 4.0** · NuCLS **CC0** → 학습한 디코더 가중치도 공유 가능 (출처 표기) |
| **기술적 독립**: CellViT 코드·가중치 미사용 | 디코더·후처리를 직접 구현 (`fm_nuclei.py`, MIT) |
| **모듈 구조**: 부품 교체 가능 | backbone(UNI-2·Virchow2로 교체 가능) / 디코더 / 후처리 / 임베딩 풀링이 분리됨 |

> 피해야 할 것: PanNuke·MoNuSeg·Lizard로 학습한 분할 가중치(CC BY-NC-SA), UNI-2·Virchow2(비상업·파생물 배포 금지).

### 순서

| § | 내용 |
|---|---|
| 1 | 학습 데이터 NuInsSeg (665장, 31개 장기) + 외부 평가용 TNBC |
| 2 | 디코더 학습 (Colab T4 약 15분) → 검증 Dice·검출 F1, 외부 데이터 성능, 예측 예시 |
| 3 | 대장암 슬라이드 1 mm² 영역: 분할 + 토큰 맵 **한 번 훑기** → 세포 윤곽·임베딩 |
| 4 | 세포 임베딩: UMAP, 군집 지도, 세포 갤러리, 유사 세포 검색 |
| 5 | NuCLS 점 라벨로 세포 분류 성능 (09의 CellViT 기반 결과와 비교) |
| 6 | 저장: 디코더 가중치, 토큰 맵, 세포 표 CSV, QuPath GeoJSON |
'''))
C.append(md(r'''
## 0. 준비

- **Colab**: `런타임 → 런타임 유형 변경 → T4 GPU` (재시작 필요 없음)
- H-optimus-0 접근 승인 + Colab 보안 비밀 `HF_TOKEN` (06 노트북 참고)
- 다운로드: H-optimus-0 4.5GB, NuInsSeg 1.6GB, 슬라이드 73MB
'''))
C.append(code(r'''
import sys, os, importlib.util
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
need = [m for m in ["openslide", "umap", "timm", "tifffile", "skimage"] if importlib.util.find_spec(m) is None]
if need or IN_COLAB:
    !pip -q install -U "timm>=1.0.9" openslide-bin openslide-python umap-learn tifffile scikit-image

WORKDIR = Path("/content/single_backbone") if IN_COLAB else Path.cwd()
WORKDIR.mkdir(parents=True, exist_ok=True); os.chdir(WORKDIR)
DATA = WORKDIR / "fm_data"; DATA.mkdir(exist_ok=True)
OUT = WORKDIR / "outputs" / "single_backbone"; OUT.mkdir(parents=True, exist_ok=True)

import urllib.request
for f in ["viz.py", "fm_nuclei.py"]:   # 저장소의 공통 모듈 (로컬에 있으면 그대로)
    if not any((d / f).exists() for d in [WORKDIR, WORKDIR.parent]):
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/fourmodern/cellvit-tutorial/main/{f}", WORKDIR / f)
for d in [WORKDIR, WORKDIR.parent]:
    if (d / "fm_nuclei.py").exists():
        sys.path.insert(0, str(d)); break

from huggingface_hub import login, whoami
try:
    user = whoami()["name"]
except Exception:
    token = None
    if IN_COLAB:
        try:
            from google.colab import userdata; token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    login(token=token) if token else login()
    user = whoami()["name"]
print("WORKDIR:", WORKDIR, "| HuggingFace:", user)
'''))
C.append(code(r'''
import glob, json, time, gc, warnings, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2, tifffile, openslide
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy import ndimage as ndi
import timm
import viz
import fm_nuclei as fn

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0)

import matplotlib.font_manager as fm
_font = WORKDIR / "NanumGothic-Regular.ttf"
if not _font.exists():
    urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf", _font)
fm.fontManager.addfont(str(_font))
plt.rcParams["font.family"] = fm.FontProperties(fname=str(_font)).get_name(); plt.rcParams["axes.unicode_minus"] = False
print("device:", DEVICE, "| torch", torch.__version__, "| timm", timm.__version__)
'''))

# ── 1. data
C.append(md(r'''
## 1. 학습 데이터

**NuInsSeg** (Mahbod et al. 2024, Zenodo 10518968, **CC BY 4.0**): 사람·쥐 31개 장기의 H&E 40x 패치 665장(512×512), 핵 인스턴스 마스크 약 3만 개.
**TNBC** (Naylor et al., Zenodo 3552674, **CC BY 4.0**): 유방암 40x 512×512 이미지 50장 + 이진 핵 마스크 → 학습에 쓰지 않는 **외부 평가용**.

`MPP_IN`에 맞춰 512 px 원본을 `RES` px로 바꾸고(0.5면 256으로 축소, 0.25면 그대로), 학습 때는 224 px을 무작위로 잘라 씁니다. 검증/외부 평가는 가운데 224 px.
'''))
C.append(md(r'''
### 입력 해상도 선택: 0.5 µm/px (기본) vs 0.25 µm/px

| `MPP_IN` | 타일 224 px이 덮는 범위 | 토큰 1칸 | 계산량 | 특징 |
|---|---|---|---|---|
| **0.5** (모델 기본) | 112 µm | 7 µm | 1× | 빠름. 림프구가 빽빽한 곳에서 붙은 핵이 가끔 합쳐짐 |
| **0.25** (40x 원본 그대로) | 56 µm | 3.5 µm | ≈ 3× | 핵 경계가 더 정밀. 문맥이 좁아져 재현율은 떨어질 수 있음 (§2 비교) |

> H-optimus-0는 0.5 µm/px로 학습되었지만, 디코더를 해당 해상도로 다시 학습하면 0.25 µm/px 입력도 동작합니다 (§2에 두 설정의 실제 비교). 
> 아래 값만 바꾸면 나머지는 자동으로 맞춰집니다 (`BASE_MPP` = 40x 원본 0.25 µm/px 기준).
'''))
C.append(code(r'''
MPP_IN = 0.5                      # 0.5 또는 0.25
BASE_MPP = 0.25                   # NuInsSeg·TNBC·슬라이드 모두 40x (0.25 µm/px)
DOWN = BASE_MPP / MPP_IN          # 원본 → 입력 축소 비 (0.5면 1/2, 0.25면 1)
RES = int(round(512 * DOWN))      # 512 px 원본 패치의 입력 크기 (256 또는 512)
CROP0 = (RES - 224) // 2          # 검증 때 가운데 224 자르기 시작점
MATCH_PX = int(round(3 / MPP_IN)) # 검출 매칭 반경 3 µm (px)
MIN_AREA = 12 if MPP_IN == 0.5 else 40
SIGMA = 2.0 if MPP_IN == 0.5 else 3.0
print(f"입력 {MPP_IN} µm/px → 512 원본 패치를 {RES} px로, 토큰 1칸 = {14 * MPP_IN:.1f} µm, 타일 = {224 * MPP_IN:.0f} µm")
'''))
C.append(code(r'''
NUI = DATA / "nuinsseg"
if not NUI.exists():
    z = DATA / "NuInsSeg.zip"; print("NuInsSeg 다운로드 중 (1.6GB) ...")
    urllib.request.urlretrieve("https://zenodo.org/api/records/10518968/files/NuInsSeg.zip/content", z)
    NUI.mkdir(); zipfile.ZipFile(z).extractall(NUI); z.unlink()
TNB = DATA / "tnbc"
if not TNB.exists():
    z = DATA / "tnbc.zip"
    urllib.request.urlretrieve("https://zenodo.org/api/records/3552674/files/TNBC_and_Brain_dataset.zip/content", z)
    TNB.mkdir(); zipfile.ZipFile(z).extractall(TNB); z.unlink()

def load_pair(img_path, lab_path, binary=False):
    img = np.array(Image.open(img_path).convert("RGB"))
    if RES != img.shape[0]:
        img = cv2.resize(img, (RES, RES), interpolation=cv2.INTER_AREA)
    if binary:
        m = (np.array(Image.open(lab_path).convert("L")) > 127).astype(np.uint8)
        lab, _ = ndi.label(cv2.resize(m, (RES, RES), interpolation=cv2.INTER_NEAREST) if RES != m.shape[0] else m)
    else:
        lab = tifffile.imread(lab_path).astype(np.int32)
        if RES != lab.shape[0]:
            lab = cv2.resize(lab, (RES, RES), interpolation=cv2.INTER_NEAREST)
    return img, lab.astype(np.int32)

items = []
for organ in sorted(d.name for d in NUI.iterdir() if d.is_dir() and not d.name.startswith(".")):
    for f in sorted((NUI / organ / "tissue images").glob("*.png")):
        m = NUI / organ / "label masks modify" / (f.stem + ".tif")
        if m.exists():
            items.append((organ, f, m))
ALL = [(o,) + load_pair(f, m) for o, f, m in items]
idx = rng.permutation(len(ALL)); n_val = int(0.15 * len(ALL))
VAL, TRAIN = [ALL[i] for i in idx[:n_val]], [ALL[i] for i in idx[n_val:]]
TNBC = [("TNBC",) + load_pair(f, Path(str(f).replace("Slide_", "GT_")), binary=True)
        for f in sorted(TNB.glob("TNBC_and_Brain_dataset/Slide_*/*.png")) if Path(str(f).replace("Slide_", "GT_")).exists()]
print(f"NuInsSeg {len(ALL)}장 ({len({o for o, _, _ in items})}개 장기) → 학습 {len(TRAIN)} · 검증 {len(VAL)} | TNBC 외부 평가 {len(TNBC)}장")
print(f"핵 수: 학습 {sum(len(np.unique(l)) - 1 for _, _, l in TRAIN):,} · 검증 {sum(len(np.unique(l)) - 1 for _, _, l in VAL):,}")

show = [TRAIN[i] for i in rng.choice(len(TRAIN), 6, replace=False)]
fig, ax = plt.subplots(2, 6, figsize=(19, 6.6))
for j, (o, im, lab) in enumerate(show):
    ax[0, j].imshow(im); ax[0, j].set_title(o, fontsize=10)
    ax[1, j].imshow(viz.overlay(im, [dict(contour=c) for c in fn.contours_from_labels(lab).values()], lambda c: (0, 255, 0), mode="contour", thickness=1))
    ax[1, j].set_title(f"핵 {len(np.unique(lab)) - 1}개", fontsize=10)
[a.axis("off") for a in ax.ravel()]; plt.suptitle(f"NuInsSeg 예시 ({RES}×{RES} px @ {MPP_IN} µm/px) · 아래: 인스턴스 마스크 윤곽", fontsize=13); plt.tight_layout(); plt.show()
'''))
C.append(md(r'''
### 학습 타깃
인스턴스 마스크를 디코더가 배울 세 가지로 바꿉니다: **3클래스 맵**(배경 / 핵 내부 / 핵 경계)과 **중심 열지도**.
경계 클래스는 붙어 있는 핵을 떼어내는 역할을 하고, 열지도는 내부 영역이 작아 마커가 안 생기는 핵을 보충합니다 (`fm_nuclei.make_targets`).
'''))
C.append(code(r'''
o, im, lab = show[0]
cls, heat = fn.make_targets(lab)
fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
ax[0].imshow(im); ax[0].set_title("H&E")
ax[1].imshow(lab % 20, cmap="tab20", interpolation="nearest"); ax[1].set_title("인스턴스 마스크 (정답)")
ax[2].imshow(cls, cmap="viridis", interpolation="nearest"); ax[2].set_title("3클래스: 배경(0) / 내부(1) / 경계(2)")
ax[3].imshow(heat, cmap="magma"); ax[3].set_title("중심 열지도")
[a.axis("off") for a in ax]; plt.tight_layout(); plt.show()
'''))

# ── 2. training
C.append(md(r'''
## 2. 디코더 학습

- backbone(H-optimus-0)은 **고정**, `forward_intermediates`로 블록 10·20·30·40의 특징을 꺼냅니다.
- 디코더(`fn.NucleiDecoder`, 3.6M): 특징 4개를 합쳐 16×16 → 28 → 56 → 112 → 224로 올리고, 112·224 단계에서 RGB 이미지에서 뽑은 얕은 특징을 더합니다(핵 경계는 픽셀 정보가 필요).
- 손실: 3클래스 교차엔트로피 + 내부 Dice + 열지도 MSE.
- 증강: 무작위 자르기·회전·뒤집기 + **염색 증강**(HED 공간에서 헤마톡실린·에오신 농도 0.6~1.4배, 감마). 염색 증강이 없으면 옅게 염색된 외부 슬라이드에서 핵을 많이 놓칩니다 (TNBC Dice 0.55 → 아래 결과와 비교).
- 평가: 핵 픽셀 Dice, 검출 F1(예측 중심과 정답 중심이 3 µm 이내).
'''))
C.append(code(r'''
backbone = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False).eval().to(DEVICE)
for p in backbone.parameters():
    p.requires_grad = False
MEAN, STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)
TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
JIT = transforms.ColorJitter(0.3, 0.3, 0.3, 0.06)
BOUNDARY_PX = 1   # 경계 클래스 두께 (px @ 0.5 µm/px)

def augment(img, lab):
    y0, x0 = rng.integers(0, RES - 224 + 1, 2)
    img, lab = img[y0:y0 + 224, x0:x0 + 224], lab[y0:y0 + 224, x0:x0 + 224]
    k = rng.integers(4); img, lab = np.rot90(img, k), np.rot90(lab, k)
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    img = fn.stain_jitter(np.ascontiguousarray(img), rng)          # 염색 차이 흉내 (HED 농도 + 감마)
    return np.array(JIT(Image.fromarray(img))), np.ascontiguousarray(lab)

@torch.inference_mode()
def predict_center(ds, crop=None):
    crop = CROP0 if crop is None else crop
    # 256 이미지의 가운데 224 → (softmax 3채널, 열지도) 리스트
    decoder.eval(); outs = []
    for b in range(0, len(ds), 16):
        batch = ds[b:b + 16]
        x = torch.stack([TF(Image.fromarray(np.ascontiguousarray(im[crop:crop + 224, crop:crop + 224]))) for _, im, _ in batch]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            feats, _ = fn.backbone_features(backbone, x); out = decoder(feats, x).float()
        outs += list(zip(torch.softmax(out[:, :3], 1).cpu().numpy(), out[:, 3].clamp(0, 1).cpu().numpy()))
    return outs

def evaluate(ds, crop=None):
    crop = CROP0 if crop is None else crop
    d_, f_ = [], []
    for (o, im, lab), (p, h) in zip(ds, predict_center(ds, crop)):
        gt = lab[crop:crop + 224, crop:crop + 224]; pl = fn.instances_from_maps(p, h, min_area=MIN_AREA)
        d_.append(fn.dice(pl > 0, gt > 0))
        g = np.array([c for _, c, _ in fn.instance_props(gt)]); q = np.array([c for _, c, _ in fn.instance_props(pl)])
        f_.append(fn.detection_f1(q, g, MATCH_PX)[2] if len(g) else np.nan)
    return float(np.mean(d_)), float(np.nanmean(f_))

EPOCHS, LR, BS = 12, 1e-3, 16   # 30으로 늘리면 검출 F1이 0.01~0.03 오릅니다 (T4 약 35분)
decoder = fn.NucleiDecoder().to(DEVICE)
opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * int(np.ceil(len(TRAIN) / BS)), pct_start=0.1)
hist, best, t0 = [], -1, time.time()
for ep in range(EPOCHS):
    decoder.train(); perm = rng.permutation(len(TRAIN)); losses = []
    for b in range(0, len(perm), BS):
        xs, cs, hs = [], [], []
        for i in perm[b:b + BS]:
            im, lab = augment(TRAIN[i][1], TRAIN[i][2]); c, h = fn.make_targets(lab, boundary_px=BOUNDARY_PX, sigma=SIGMA)
            xs.append(TF(Image.fromarray(im))); cs.append(torch.from_numpy(c)); hs.append(torch.from_numpy(h))
        x, c, h = torch.stack(xs).to(DEVICE), torch.stack(cs).to(DEVICE), torch.stack(hs).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            feats, _ = fn.backbone_features(backbone, x); out = decoder(feats, x)
        loss = fn.seg_loss(out.float(), c, h)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(loss.item())
    vd, vf = evaluate(VAL)
    hist.append(dict(epoch=ep + 1, loss=np.mean(losses), val_dice=vd, val_f1=vf))
    print(f"epoch {ep + 1:2d}: loss {np.mean(losses):.3f} | 검증 Dice {vd:.3f} · 검출 F1 {vf:.3f} | {time.time() - t0:.0f}s")
    if vf > best:
        best = vf; torch.save(decoder.state_dict(), OUT / f"nuclei_decoder_{MPP_IN}um.pt")
decoder.load_state_dict(torch.load(OUT / f"nuclei_decoder_{MPP_IN}um.pt"))
H = pd.DataFrame(hist)
'''))
C.append(code(r'''
tn_d, tn_f = evaluate(TNBC)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].plot(H.epoch, H.loss, marker="o"); ax[0].set_xlabel("epoch"); ax[0].set_title("학습 손실"); ax[0].grid(alpha=0.3)
ax[1].plot(H.epoch, H.val_dice, marker="o", label="검증 Dice (핵 픽셀)"); ax[1].plot(H.epoch, H.val_f1, marker="s", label="검증 검출 F1 (3 µm)")
ax[1].axhline(tn_d, ls="--", color="C0", alpha=0.6, label=f"TNBC 외부 Dice {tn_d:.3f}"); ax[1].axhline(tn_f, ls="--", color="C1", alpha=0.6, label=f"TNBC 외부 검출 F1 {tn_f:.3f}")
ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3); ax[1].set_title("검증 · 외부 평가")
plt.tight_layout(); plt.show()
print(f"최종 — 검증: Dice {H.val_dice.iloc[-1]:.3f}, 검출 F1 {best:.3f} | TNBC(학습에 미사용, 다른 기관·스캐너): Dice {tn_d:.3f}, 검출 F1 {tn_f:.3f}")
'''))
C.append(md(r'''
> TNBC의 정답은 **이진 마스크**라 붙어 있는 핵이 하나로 합쳐져 있습니다. 검출 F1은 그 영향으로 실제보다 낮게 나옵니다. Dice를 주로 보세요.

**두 해상도를 같은 조건(12 epoch)으로 학습한 결과** (RTX 3090 Ti, 이 노트북을 두 설정으로 실행; 12 epoch 디코더는 실행마다 ±0.03 정도 변동):

| 입력 | 검증 Dice | 검증 검출 F1 | TNBC Dice | TNBC 검출 F1 | ROI 1 mm² 처리 | ROI 핵 수 |
|---|---|---|---|---|---|---|
| 0.5 µm/px | 0.80 | 0.73 | 0.67~0.77 | 0.81 | 47s | 14,285 |
| 0.25 µm/px | 0.77 | **0.77** | 0.71~0.76 | 0.81 | 140s (3×) | 11,683 |

- 패치 벤치마크에서는 0.25가 **검출 F1이 높습니다** (붙은 핵의 경계를 더 잘 나눔). Dice는 0.5가 약간 높습니다 (넓은 문맥).
- 그런데 **슬라이드 ROI에서는 0.25가 핵을 오히려 적게 찾았습니다** — 림프 응집 확대(§3)를 보면 0.25는 윤곽이 더 정확하지만 또렷한 림프구를 꽤 놓칩니다.

**epoch을 30으로 늘리고 `MIN_AREA`를 검증 세트로 고른 뒤 다시 비교해도 결론은 같았습니다**:

| 30 epoch | 검증 검출 F1 (최적 MIN_AREA) | 검증 재현율 | TNBC Dice | 림프 응집 150 µm 구역 핵 수 | ROI 전체 |
|---|---|---|---|---|---|
| 0.5 µm/px (MIN_AREA 8) | 0.74 | 0.68 | 0.74 | **402** | 15,090 |
| 0.25 µm/px (MIN_AREA 16) | **0.77** | 0.73 | 0.72 | 305 | 12,619 |

- `MIN_AREA`는 거의 영향이 없습니다 (검증 F1 변화 < 0.01, 림프 응집 핵 수는 8→40에서 7% 차이). 재현율을 정하는 것은 후처리가 아니라 **디코더 자체**입니다.
- 0.25가 찾은 핵의 88%는 0.5도 찾지만, 0.5가 찾은 핵의 74%만 0.25에 있습니다 → 슬라이드에서 0.25는 "더 정밀하지만 더 엄격한 부분집합"입니다.
- 패치 벤치마크(NuInsSeg)와 실제 슬라이드의 순위가 뒤집히는 이유로는 56 µm 시야에서 빽빽한 림프구의 문맥이 부족해지는 점, 학습 데이터(31개 장기 혼합)에 림프 조직이 적은 점이 의심됩니다. 
  **림프 조직 패치를 학습 데이터에 추가**하는 것이 0.25를 살리는 가장 현실적인 방법입니다.
- **정리**: 기본은 0.5 µm/px. 0.25는 핵 **경계 정밀도**가 중요하고 대상 조직의 라벨 데이터로 재학습할 수 있을 때만 권합니다.

### 예측 예시 (검증 · 외부)
초록 = 정답 윤곽, 빨강 = 예측 윤곽.
'''))
C.append(code(r'''
def show_preds(ds, n=4, crop=None, title=""):
    crop = CROP0 if crop is None else crop
    pick = [ds[i] for i in rng.choice(len(ds), n, replace=False)]
    preds = predict_center(pick, crop)
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 9.4))
    for j, ((o, im, lab), (p, h)) in enumerate(zip(pick, preds)):
        im_, gt = im[crop:crop + 224, crop:crop + 224], lab[crop:crop + 224, crop:crop + 224]
        pl = fn.instances_from_maps(p, h, min_area=MIN_AREA)
        ax[0, j].imshow(viz.overlay(im_, [dict(contour=c) for c in fn.contours_from_labels(gt).values()], lambda c: (0, 230, 0), mode="contour", thickness=1))
        ax[0, j].set_title(f"{o} · 정답 {len(np.unique(gt)) - 1}개", fontsize=10)
        ax[1, j].imshow(viz.overlay(im_, [dict(contour=c) for c in fn.contours_from_labels(pl).values()], lambda c: (255, 40, 40), mode="contour", thickness=1))
        ax[1, j].set_title(f"예측 {len(np.unique(pl)) - 1}개 · Dice {fn.dice(pl > 0, gt > 0):.2f}", fontsize=10)
    [a.axis("off") for a in ax.ravel()]; plt.suptitle(title, fontsize=13); plt.tight_layout(); plt.show()
show_preds(VAL, title="NuInsSeg 검증 (위: 정답, 아래: 예측)")
show_preds(TNBC, title="TNBC 외부 데이터 (위: 정답(이진), 아래: 예측)")
'''))

# ── 3. ROI
C.append(md(r'''
## 3. 슬라이드 영역에 적용: 분할 + 토큰 맵을 한 번에

09와 같은 대장암 슬라이드(TCGA-AD-6890, GDC open access)의 4,096×4,096 px(1 mm²) 영역입니다.
`fn.RoiRunner`가 영역을 `MPP_IN` 해상도로 맞춰 224 타일을 절반씩 겹쳐 훑고, 타일마다 **디코더 출력의 가운데 112×112**와 **토큰의 가운데 8×8**만 기록합니다.
한 번 훑으면 분할 맵과 토큰 맵이 모두 나옵니다.
'''))
C.append(code(r'''
SLIDE = DATA / "TCGA-AD-6890-01Z-00-DX1.svs"
if not SLIDE.exists():
    urllib.request.urlretrieve("https://api.gdc.cancer.gov/data/c7e229f4-7185-4211-979c-6cca1bbe4e0f", SLIDE)
slide = openslide.OpenSlide(str(SLIDE)); MPP = float(slide.properties["openslide.mpp-x"])
ROI_X, ROI_Y, ROI = 17000, 5500, 4096
roi_img = np.array(slide.read_region((ROI_X, ROI_Y), 0, (ROI, ROI)).convert("RGB"))     # 0.25 µm/px
roi05 = cv2.resize(roi_img, (int(ROI * DOWN), int(ROI * DOWN)), interpolation=cv2.INTER_AREA) if DOWN != 1 else roi_img   # 입력 해상도
SCALE = MPP / DOWN    # 입력 1 px = µm
S0 = int(round(1 / DOWN))   # 입력 좌표 → 원본(level-0) 좌표 배율 (0.5면 2, 0.25면 1)

runner = fn.RoiRunner(backbone, decoder, TF, device=DEVICE, batch=16)
t = time.time()
PROB, HEAT, FMAP = runner.run(roi05)
LAB = fn.instances_from_maps(PROB, HEAT, min_area=MIN_AREA)
props = fn.instance_props(LAB); CONT = fn.contours_from_labels(LAB)
print(f"한 번 훑기 {time.time() - t:.0f}s → 분할 맵 {PROB.shape[1:]} · 토큰 맵 {FMAP.shape} | 검출된 핵 {len(props):,}개")
np.save(OUT / "h0_token_map.npy", FMAP)
'''))
C.append(code(r'''
fig, ax = plt.subplots(1, 3, figsize=(22, 7.6))
ax[0].imshow(roi05); ax[0].set_title(f"조직 사진 ({MPP_IN} µm/px)")
ax[1].imshow(PROB[1] + PROB[2], cmap="magma"); ax[1].set_title("핵 확률 (내부 + 경계)")
zs = int(round(150 / SCALE)); zy0, zx0 = int(1300 * DOWN * 2), int(600 * DOWN * 2)   # 림프 응집 (0.25 µm/px 기준 (600, 1300)에서 150 µm)
z = (slice(zy0, zy0 + zs), slice(zx0, zx0 + zs))
cz = [dict(contour=c - (zx0, zy0)) for c in CONT.values() if z[1].start <= c[:, 0].mean() < z[1].stop and z[0].start <= c[:, 1].mean() < z[0].stop]
ax[2].imshow(viz.overlay(np.ascontiguousarray(roi05[z]), cz, lambda c: (255, 40, 40), mode="contour", thickness=1)); ax[2].set_title(f"림프 응집 확대 (150 µm): 핵 {len(cz)}개")
[a.axis("off") for a in ax]; plt.tight_layout(); plt.show()
'''))
C.append(md(r'''
> **보이는 한계**: 0.5 µm/px에서는 림프 응집처럼 핵이 빽빽한 곳에서 붙어 있는 핵 2~3개가 하나로 합쳐지는 경우가 있습니다 (토큰 1칸 = 7 µm). 
> `MPP_IN = 0.25`(3.5 µm 토큰)로 바꾸면 윤곽은 더 정밀해지지만, 30 epoch·MIN_AREA 조정 후에도 림프 응집의 핵을 약 25% 더 놓쳤습니다 (§2 비교 표). 
> 검출 수(약 1.2~1.4만)가 09의 CellViT(약 2.9만)보다 적은 것은 해상도 외에도 CellViT가 작은 조각까지 핵으로 세는 경향, 후처리 기준 차이 때문이므로 숫자 자체보다 확대 그림으로 판단하세요.

### 인터랙티브: 조직 위 핵 윤곽
휠로 확대, 범례 클릭. 윤곽 좌표는 원본 해상도(0.25 µm/px)로 바꿔 그립니다.
'''))
C.append(code(r'''
cells = [dict(id=i, contour=CONT[i] * S0, centroid=(cx * S0, cy * S0), area_um2=a * SCALE ** 2) for i, (cx, cy), a in props if i in CONT]
vx, vy, vs_ = 1024, 1024, 2048
sub = [c for c in cells if vx <= c["centroid"][0] < vx + vs_ and vy <= c["centroid"][1] < vy + vs_]
viz.interactive(roi_img[vy:vy + vs_, vx:vx + vs_], sub, lambda c: (255, 60, 60), lambda c: "핵", hover_of=lambda c: f"핵 #{c['id']}<br>면적 {c['area_um2']:.0f} µm²",
                offset=(vx, vy), title=f"H-optimus-0 디코더로 검출한 핵 {len(sub):,}개 (2048×2048 px 영역)", height=800)
'''))

# ── 4. embeddings
C.append(md(r'''
## 4. 세포 임베딩: 같은 토큰 맵에서 마스크로 꺼내기

각 핵이 덮는 토큰 칸(14×14 px)을 겹친 픽셀 수로 가중 평균합니다 (`fn.mask_pool`).
`DILATE_UM`으로 핵 바깥 세포질 몫을 조금 넓힐 수 있습니다.
'''))
C.append(code(r'''
DILATE_UM = 1.0
DILATE_PX = int(round(DILATE_UM / SCALE))
t = time.time()
EMB = fn.mask_pool(FMAP, LAB, dilate_px=DILATE_PX)
ids = [i for i in EMB if i in CONT]
E = np.stack([EMB[i] for i in ids]); xy05 = np.array([[c[:, 0].mean(), c[:, 1].mean()] for c in (CONT[i] for i in ids)])
print(f"세포 {len(ids):,}개 임베딩 {E.shape} | {time.time() - t:.1f}s (모델 추론 없음)")

import umap
from sklearn.cluster import KMeans
K = 6
En = E / np.linalg.norm(E, axis=1, keepdims=True)
km = KMeans(K, n_init=10, random_state=0).fit(En)
pal = plt.get_cmap("tab10"); KCOL = {f"군집 {c}": tuple(int(255 * v) for v in pal(c)[:3]) for c in range(K)}
clus = np.array([f"군집 {c}" for c in km.labels_])
U = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=0).fit_transform(En)

fig, ax = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1, 1.1]})
for k, col in KCOL.items():
    m = clus == k; ax[0].scatter(U[m, 0], U[m, 1], s=3, color=np.array(col) / 255, label=f"{k} ({m.sum()})")
ax[0].legend(markerscale=5); ax[0].set_title("세포 임베딩 UMAP (k-means 군집)"); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].imshow(roi05)
for k, col in KCOL.items():
    m = clus == k; ax[1].scatter(xy05[m, 0], xy05[m, 1], s=2, color=np.array(col) / 255)
ax[1].axis("off"); ax[1].set_title("군집을 조직 위에")
plt.tight_layout(); plt.show()

read_roi = lambda x, y, s: np.array(slide.read_region((int(x) + ROI_X, int(y) + ROI_Y), 0, (s, s)).convert("RGB"))
gal = [dict(contour=CONT[i] * S0, centroid=tuple(xy05[n] * S0), cluster=clus[n]) for n, i in enumerate(ids)]
viz.cell_gallery(gal, lambda c: c["cluster"], lambda c: KCOL[c["cluster"]], order=list(KCOL), get_crop=read_roi, n=12, size=72,
                 title="군집별 세포 (원본 H&E, 윤곽 = H-optimus-0 디코더 분할)")
'''))
C.append(code(r'''
# 비슷한 세포 찾기
q = int(rng.choice(len(ids)))
sim = En @ En[q]; sim[q] = -1; nn_ = np.argsort(-sim)[:11]
viz.cell_gallery([dict(gal[n], row="질의 + 이웃") for n in [q] + list(nn_)], lambda c: c["row"], lambda c: (255, 60, 60), order=["질의 + 이웃"],
                 get_crop=read_roi, n=12, size=72, seed=0, title=f"세포 #{ids[q]}와 임베딩이 가장 비슷한 세포 11개 (코사인 유사도 {sim[nn_].min():.2f}~{sim[nn_].max():.2f})")
'''))

# ── 5. NuCLS
C.append(md(r'''
## 5. 세포 분류 성능: NuCLS 점 라벨

09 노트북과 같은 평가입니다. 다른 점은 **핵 검출도 이 노트북의 디코더가** 한다는 것입니다.
NuCLS(CC0) 이미지에서 핵을 검출하고, 정답 점(4종: Tumor / nonTIL Stromal / sTIL / Other)과 짝지은(≤ 3 µm) 세포의 임베딩으로 로지스틱 회귀를 학습합니다.

> 09에서 같은 평가(CellViT 검출 + CellViT 임베딩 / H-optimus-0 마스크 임베딩)는 macro-F1 약 0.50이었습니다. 세포 수가 적어 오차가 큽니다.
'''))
C.append(code(r'''
NUCLS_DIR = WORKDIR / "cellvitpp_repo/test_database/training_database/Example-Detection"
if not NUCLS_DIR.exists():
    !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/TIO-IKIM/CellViT-plus-plus.git cellvitpp_repo
    !cd cellvitpp_repo && git sparse-checkout set test_database/training_database/Example-Detection
NUCLS = {0: "Tumor", 1: "nonTIL Stromal", 2: "sTIL", 3: "Other"}
NCOL = {0: (220, 30, 30), 1: (34, 139, 34), 2: (30, 90, 255), 3: (150, 150, 150)}
from scipy.optimize import linear_sum_assignment

def nucls_rows(split):
    rows = []
    for p in sorted((NUCLS_DIR / split / "images").glob("*.png")):
        img = np.array(Image.open(p).convert("RGB"))                         # 256 px @ 0.25 µm/px
        img05 = cv2.resize(img, (int(256 * DOWN), int(256 * DOWN)), interpolation=cv2.INTER_AREA) if DOWN != 1 else img
        lbl = pd.read_csv(NUCLS_DIR / split / "labels" / f"{p.stem}.csv", header=None, names=["x", "y", "label"])
        prob, heat, fmap = runner.run(img05); lab = fn.instances_from_maps(prob, heat, min_area=MIN_AREA)
        pr = fn.instance_props(lab)
        if not pr:
            continue
        emb = fn.mask_pool(fmap, lab, dilate_px=DILATE_PX)
        dc = np.array([c for _, c, _ in pr]); gt = lbl[["x", "y"]].values * DOWN
        d = np.linalg.norm(gt[:, None] - dc[None], axis=-1); gi, di = linear_sum_assignment(d)
        for g, k in zip(gi, di):
            if d[g, k] <= MATCH_PX and pr[k][0] in emb:
                rows.append(dict(label=int(lbl.label[g]), emb=emb[pr[k][0]]))
    return rows
TR_, TE_ = nucls_rows("train"), nucls_rows("test")
print(f"짝지은 세포: train {len(TR_)} · test {len(TE_)}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
Xtr_, ytr_ = np.stack([r["emb"] for r in TR_]), np.array([r["label"] for r in TR_])
Xte_, yte_ = np.stack([r["emb"] for r in TE_]), np.array([r["label"] for r in TE_])
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05, class_weight="balanced")).fit(Xtr_, ytr_)
pred_ = clf.predict(Xte_)
boots = [f1_score(yte_[ii], pred_[ii], average="macro", labels=list(NUCLS)) for ii in (rng.integers(0, len(yte_), len(yte_)) for _ in range(500))]
print(f"macro-F1 {f1_score(yte_, pred_, average='macro', labels=list(NUCLS)):.3f} (bootstrap 95% {np.percentile(boots, 2.5):.3f}–{np.percentile(boots, 97.5):.3f})")
viz.classification_metrics(yte_, pred_, NUCLS, NCOL, title="H-optimus-0 단일 backbone: 검출 + 임베딩 → NuCLS 4종 분류")
'''))

# ── 6. save
C.append(md(r'''
## 6. 저장·내보내기

| 파일 | 내용 |
|---|---|
| `nuclei_decoder_{MPP_IN}um.pt` | 학습한 디코더 가중치 (3.6M, 약 14MB). H-optimus-0 + 이 가중치 + 같은 `MPP_IN`으로 어디서든 재현 |
| `h0_token_map.npy` | 영역 토큰 맵 — 나중에 모델 없이 좌표로 임베딩 꺼내기 |
| `cells.csv` / `cells.npz` | 세포별 슬라이드 좌표(level-0), 면적, 군집 / 임베딩 |
| `cells.geojson` | QuPath `File → Import objects` |

**공유 가능 여부**: H-optimus-0(Apache 2.0)와 CC BY 데이터로 학습한 이 디코더 가중치는 출처를 표기하면 **배포·상업적 이용이 가능**합니다
(H-optimus-0 모델 카드의 의료 규제 관련 면책 조건은 그대로 적용).
'''))
C.append(code(r'''
df = pd.DataFrame({"id": ids, "x_level0": xy05[:, 0] * S0 + ROI_X, "y_level0": xy05[:, 1] * S0 + ROI_Y,
                   "area_um2": [dict((i, a) for i, _, a in props)[i] * SCALE ** 2 for i in ids], "cluster": clus})
df.to_csv(OUT / "cells.csv", index=False)
np.savez_compressed(OUT / "cells.npz", id=np.array(ids), xy_level0=df[["x_level0", "y_level0"]].values, emb=E.astype(np.float16))
feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [(CONT[i] * S0 + (ROI_X, ROI_Y)).tolist() + [(CONT[i][0] * S0 + (ROI_X, ROI_Y)).tolist()]]},
          "properties": {"objectType": "detection", "classification": {"name": c, "color": list(KCOL[c])}}} for i, c in zip(ids, clus)]
(OUT / "cells.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
!ls -lh {OUT}
'''))
C.append(md(r'''
## 정리

| 단계 | 구현 | 외부 의존 |
|---|---|---|
| 특징 추출 | H-optimus-0 고정, 타일당 1회 forward (중간 특징 + 마지막 토큰) | Apache 2.0 |
| 핵 분할 | 직접 학습한 3.6M 디코더 (`fm_nuclei.NucleiDecoder`) + 경계 클래스·열지도 watershed | NuInsSeg CC BY 4.0 |
| 세포 임베딩 | 핵 마스크로 토큰 가중 평균 (`fn.mask_pool`) | — |
| 분류 | 자체 점 라벨로 작은 분류기 (NuCLS CC0 예시) | — |

**다음 단계**
- 입력 해상도: 기본 0.5 µm/px가 빠르고 재현율이 높습니다. `MPP_IN = 0.25`는 경계가 정밀하지만 이 슬라이드에서는 30 epoch에서도 빽빽한 림프구를 더 놓쳤고 계산은 3배입니다. 쓰려면 대상 조직의 라벨로 재학습하고 재현율을 검증하세요.
- backbone 교체: `timm.create_model(...)`과 정규화 값만 바꾸면 UNI-2·Virchow2에서도 같은 디코더 구조를 학습할 수 있습니다 (라이선스 주의).
- 학습 데이터 추가: 자체 조직의 핵 마스크 수십 장만 더해 미세조정하면 도메인 차이를 줄일 수 있습니다.
- 슬라이드 전체: `RoiRunner`를 큰 타일(예: 4096 px) 단위로 돌리고 토큰 맵을 zarr로 저장하세요.
'''))

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, NB)
print("wrote", NB)
