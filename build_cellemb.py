"""Foundation model 잠재공간에서 좌표로 세포 임베딩 꺼내기 노트북 생성 스크립트.

python build_cellemb.py  →  09_cell_embeddings_from_FM.ipynb
"""
import textwrap

import nbformat as nbf

REPO = "fourmodern/cellvit-tutorial"
NB = "09_cell_embeddings_from_FM.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(textwrap.dedent(t).strip())


def code(t):
    return nbf.v4.new_code_cell(textwrap.dedent(t).strip())


C = []
C.append(md(f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/{NB}" target="_parent">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'))
C.append(md(r'''
# Foundation model 잠재공간에서 **좌표로** 세포 임베딩 꺼내기 (H-optimus-0 + CellViT)

CellViT는 패치를 한 번 추론하면 **위치가 정해진 토큰 격자(잠재공간)** 가 나오고, 세포 좌표에 해당하는 토큰만 골라 세포 임베딩을 만듭니다 (01·02 노트북).
H-optimus-0 같은 병리 foundation model도 ViT라서 똑같이 할 수 있습니다.

```
슬라이드 ─► H-optimus-0 (타일을 절반씩 겹쳐 한 번 훑기) ─► 슬라이드 크기의 토큰 맵 (토큰 1칸 = 7 µm, 1,536차원)  ← 한 번만 계산·저장
                                                                    │
CellViT 세포 분할 ─► 세포 중심 좌표 ─────────────── 좌표 → 토큰 위치 ──┴─► 세포 임베딩 (꺼내기만 하면 됨)
```

| | CellViT-SAM-H 잠재공간 | H-optimus-0 잠재공간 |
|---|---|---|
| 토큰 1칸 | 16 px @ 0.25 µm/px = **4 µm** | 14 px @ 0.5 µm/px = **7 µm** |
| 입력 | 1024 px | 224 px 고정 |
| 학습 목적 | 세포 분할·분류 | 조직 패치 표현 (자기지도) |

### 이 노트북에서 하는 것

| Part | 내용 |
|---|---|
| A | 대장암 슬라이드 1 mm² 영역: CellViT 분할 → H-optimus-0 토큰 맵 생성·저장 → 좌표로 세포 임베딩 → 토큰 맵 PCA, 세포 UMAP, 비슷한 세포 찾기, 세포 군집 지도 |
| B | **품질 평가**: NuCLS 세포 라벨로 CellViT 임베딩 vs H-optimus-0 임베딩의 세포 분류 성능 비교 |
'''))
C.append(md(r'''
## 0. 준비

- **Colab**: `런타임 → 런타임 유형 변경 → T4 GPU`. 첫 셀에서 `cellvit` 설치 후 **런타임이 자동 재시작**됩니다 → 처음부터 다시 실행.
- H-optimus-0 접근 승인 + Colab 보안 비밀 `HF_TOKEN` (06 노트북 참고)
- 모델 다운로드: CellViT-SAM-H 2.8GB + H-optimus-0 4.5GB. 두 모델은 GPU에 하나씩 올립니다.
'''))
C.append(code(r'''
import sys, os, importlib.util
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and importlib.util.find_spec("cellvit") is None:
    !apt-get -qq install -y openslide-tools > /dev/null
    !pip -q install cellvit openslide-bin umap-learn "timm>=1.0.9"
    print("설치 완료 → 런타임을 재시작합니다. 재시작 후 처음부터 다시 실행하세요.")
    os.kill(os.getpid(), 9)
'''))
C.append(code(r'''
from pathlib import Path
WORKDIR = Path("/content/cellemb") if IN_COLAB else Path.cwd()
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)
os.environ["CELLVIT_CACHE"] = str(WORKDIR / "models")   # cellvit import 전에
DATA = WORKDIR / "fm_data"; DATA.mkdir(exist_ok=True)
OUT = WORKDIR / "outputs"; OUT.mkdir(exist_ok=True)

from huggingface_hub import login, whoami
try:
    user = whoami()["name"]
except Exception:
    token = None
    if IN_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass
    login(token=token) if token else login()
    user = whoami()["name"]
print("WORKDIR:", WORKDIR, "| HuggingFace:", user)
'''))
C.append(code(r'''
import json, time, gc, warnings, urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import openslide
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib.font_manager as fm
_font = WORKDIR / "NanumGothic-Regular.ttf"
if not _font.exists():
    urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf", _font)
fm.fontManager.addfont(str(_font))
plt.rcParams["font.family"] = fm.FontProperties(fname=str(_font)).get_name()
plt.rcParams["axes.unicode_minus"] = False

for _d in [WORKDIR, WORKDIR.parent]:  # 공통 시각화 도구 viz.py
    if (_d / "viz.py").exists():
        sys.path.insert(0, str(_d)); break
else:
    urllib.request.urlretrieve("https://raw.githubusercontent.com/fourmodern/cellvit-tutorial/main/viz.py", WORKDIR / "viz.py")
    sys.path.insert(0, str(WORKDIR))
import viz
print("device:", DEVICE, "| torch", torch.__version__)
'''))

# ── Part A
C.append(md(r'''
# Part A. 슬라이드에서 세포 임베딩 만들기

## A-1. 슬라이드와 관심 영역

GDC 공개 대장암 진단 슬라이드(TCGA-AD-6890, 40x = 0.25 µm/px, 73MB, 05~07 노트북과 같은 슬라이드)에서
종양 샘 · 기질 · 림프 응집 · 정상 점막이 함께 있는 **4,096×4,096 px (약 1 mm²)** 영역을 씁니다.
'''))
C.append(code(r'''
SLIDE_PATH = DATA / "TCGA-AD-6890-01Z-00-DX1.svs"
if not SLIDE_PATH.exists():
    urllib.request.urlretrieve("https://api.gdc.cancer.gov/data/c7e229f4-7185-4211-979c-6cca1bbe4e0f", SLIDE_PATH)
slide = openslide.OpenSlide(str(SLIDE_PATH))
MPP = float(slide.properties["openslide.mpp-x"])     # ≈ 0.25 µm/px (40x)
W, H = slide.dimensions

ROI_X, ROI_Y, ROI = 17000, 5500, 4096                 # level-0 좌표 (0.25 µm/px)
roi_img = np.array(slide.read_region((ROI_X, ROI_Y), 0, (ROI, ROI)).convert("RGB"))
thumb = np.array(slide.get_thumbnail((1200, 1200)).convert("RGB")); DS = W / thumb.shape[1]

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(thumb); ax[0].add_patch(plt.Rectangle((ROI_X / DS, ROI_Y / DS), ROI / DS, ROI / DS, fill=False, ec="lime", lw=2))
ax[0].set_title("슬라이드 전체 (초록 = 관심 영역)")
ax[1].imshow(roi_img[::4, ::4]); ax[1].set_title(f"관심 영역 {ROI}×{ROI}px = {ROI * MPP / 1000:.2f}×{ROI * MPP / 1000:.2f} mm")
[a.axis("off") for a in ax]; plt.tight_layout(); plt.show()
'''))

C.append(md(r'''
## A-2. CellViT로 세포 분할 (+ CellViT 자체 임베딩)

1024 px 타일을 64 px씩 겹쳐 추론하고, 타일 가장자리 32 px 안쪽에 중심이 있는 세포만 남겨 중복을 없앱니다.
비교를 위해 CellViT 잠재공간에서 꺼낸 세포 임베딩(bbox가 덮는 토큰 평균, 1,280차원)도 함께 저장합니다.
'''))
C.append(code(r'''
from cellvit.utils.cache_models import cache_cellvit_sam_h
from cellvit.utils.tools import unflatten_dict
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM

ck = torch.load(cache_cellvit_sam_h(), map_location="cpu", weights_only=False)
cconf = unflatten_dict(ck["config"], ".")
cellvit = CellViTSAM(model_path=None, num_nuclei_classes=cconf["data"]["num_nuclei_classes"],
                     num_tissue_classes=cconf["data"]["num_tissue_classes"], vit_structure=cconf["model"]["backbone"],
                     regression_loss=cconf["model"].get("regression_loss", False))
cellvit.load_state_dict(ck["model_state_dict"]); cellvit = cellvit.eval().to(DEVICE); del ck
PANNUKE = {v: k for k, v in cconf["dataset_config"]["nuclei_types"].items()}
PAN_COL = {"Neoplastic": (220, 20, 20), "Inflammatory": (30, 90, 255), "Connective": (34, 139, 34), "Dead": (128, 128, 128), "Epithelial": (255, 140, 0)}

@torch.inference_mode()
def cellvit_cells(img, mag=40):
    # 패치(0.25 µm/px) → 세포 리스트 + 세포별 CellViT 임베딩
    x = torch.from_numpy(((img / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)).float()[None].to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        out = cellvit(x, retrieve_tokens=True)
    out = {k: v.float().cpu() for k, v in out.items()}
    pred = {"nuclei_binary_map": F.softmax(out["nuclei_binary_map"], 1), "nuclei_type_map": F.softmax(out["nuclei_type_map"], 1),
            "hv_map": out["hv_map"]}
    _, cd = cellvit.calculate_instance_map(pred, magnification=mag)
    tok, cells = out["tokens"][0], []
    for c in cd[0].values():
        if c["type"] == 0:
            continue
        bb = np.asarray(c["bbox"]) / 16
        r0, c0 = np.floor(bb[0]).astype(int); r1, c1 = np.ceil(bb[1]).astype(int)
        cells.append(dict(centroid=np.asarray(c["centroid"], float), contour=np.asarray(c["contour"]), type=PANNUKE[c["type"]],
                          emb_cellvit=tok[:, r0:r1, c0:c1].flatten(1).mean(1).numpy()))
    return cells

P, OV = 1024, 64
starts = list(range(0, ROI - P + 1, P - OV))
if starts[-1] != ROI - P:
    starts.append(ROI - P)
CELLS, t = [], time.time()
for ty in starts:
    for tx in starts:
        for c in cellvit_cells(roi_img[ty:ty + P, tx:tx + P]):
            cx, cy = c["centroid"]
            lo_x, hi_x = (0 if tx == 0 else OV // 2), (P if tx == starts[-1] else P - OV // 2)
            lo_y, hi_y = (0 if ty == 0 else OV // 2), (P if ty == starts[-1] else P - OV // 2)
            if lo_x <= cx < hi_x and lo_y <= cy < hi_y:          # 겹침 영역 중복 제거
                c["centroid"] = c["centroid"] + (tx, ty); c["contour"] = c["contour"] + (tx, ty)   # ROI 좌표 (0.25 µm/px)
                CELLS.append(c)
print(f"CellViT: 세포 {len(CELLS):,}개 | {time.time() - t:.0f}s |", pd.Series([c["type"] for c in CELLS]).value_counts().to_dict())
cellvit = None; gc.collect(); torch.cuda.empty_cache()
'''))

C.append(md(r'''
## A-3. H-optimus-0 토큰 맵 만들기 (한 번만 계산)

1. 관심 영역을 **0.5 µm/px**로 줄입니다 (4096 → 2048 px). H-optimus-0의 입력 해상도입니다.
2. 가장자리에 흰색 56 px 여백을 붙이고, **224 px 타일을 112 px 간격(절반 겹침)** 으로 훑습니다.
3. 각 타일의 16×16 토큰 중 **가운데 8×8 토큰만** 슬라이드 토큰 맵에 기록합니다.
   타일 가장자리 토큰은 주변 문맥이 잘려 품질이 떨어지는데, 이웃 타일의 가운데 토큰이 그 자리를 채워 **경계 없는 맵**이 됩니다.

```
타일 A ┌──────────────┐               토큰 맵에는 각 타일의 가운데(■)만 기록
       │   ┌──────┐   │               → 112 px(8토큰) 간격으로 빈틈없이 맞물림
       │   │ ■■■■ │   │ 타일 B ┌──────────────┐
       │   │ ■■■■ │   │        │   ┌──────┐   │
       └───┴──────┴───┘        │   │ ■■■■ │   │
         ◄─ 112 px ─►          └───┴──────┴───┘
```
'''))
C.append(code(r'''
import timm
h0 = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False).eval().to(DEVICE)
H0_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517))])
NPRE = h0.num_prefix_tokens          # CLS + register 4개 = 5
TOK, TILE, STRIDE, PAD = 14, 224, 112, 56   # 토큰 14 px, 타일 224 px, 절반 겹침, 여백 (모두 0.5 µm/px 기준)

@torch.inference_mode()
def token_map(img05, batch=32):
    # 0.5 µm/px 이미지 → (H/14, W/14, 1536) 토큰 맵. 토큰 (i, j) = 원본 픽셀 [14i, 14i+14) × [14j, 14j+14)
    h, w = img05.shape[:2]
    gh, gw = int(np.ceil(h / TOK)), int(np.ceil(w / TOK))
    H_, W_ = gh * TOK + 2 * PAD, gw * TOK + 2 * PAD
    canvas = np.full((H_ + TILE, W_ + TILE, 3), 255, np.uint8)
    canvas[PAD:PAD + h, PAD:PAD + w] = img05
    fmap = np.zeros((gh, gw, h0.num_features), np.float16)
    pos = [(y, x) for y in range(0, gh * TOK, STRIDE) for x in range(0, gw * TOK, STRIDE)]   # 원본 기준 타일 가운데 영역 시작점
    for b in range(0, len(pos), batch):
        bp = pos[b:b + batch]
        x = torch.stack([H0_TF(Image.fromarray(canvas[y:y + TILE, x0:x0 + TILE])) for y, x0 in bp]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            tok = h0.forward_features(x)[:, NPRE:].reshape(-1, 16, 16, h0.num_features).float().cpu().numpy()
        for (y, x0), t_ in zip(bp, tok):
            gy, gx = y // TOK, x0 // TOK
            ny, nx = min(8, gh - gy), min(8, gw - gx)
            fmap[gy:gy + ny, gx:gx + nx] = t_[4:4 + ny, 4:4 + nx]   # 가운데 8×8 토큰만
    return fmap

roi05 = cv2.resize(roi_img, (ROI // 2, ROI // 2), interpolation=cv2.INTER_AREA)   # 0.25 → 0.5 µm/px
t = time.time()
FMAP = token_map(roi05)
print(f"토큰 맵 {FMAP.shape} ({FMAP.nbytes / 1e6:.0f} MB, fp16) | {time.time() - t:.0f}s")

# 저장: 토큰 맵 + 좌표 변환에 필요한 정보 → 나중에 모델 없이도 좌표로 바로 꺼낼 수 있음
np.save(OUT / "h0_token_map.npy", FMAP)
json.dump(dict(slide=SLIDE_PATH.name, roi_x=ROI_X, roi_y=ROI_Y, roi_size=ROI, level0_mpp=MPP, map_mpp=MPP * 2, token_px=TOK),
          open(OUT / "h0_token_map.json", "w"), indent=1)
'''))

C.append(md(r'''
### 토큰 맵 들여다보기 (PCA → RGB)

1,536차원 토큰을 PCA 3개 성분으로 줄여 RGB로 칠합니다. **라벨 없이도** 종양 샘, 기질, 림프 응집, 점막이 다른 색으로 나뉘면
토큰 맵이 위치별 조직 정보를 담고 있다는 뜻입니다.
'''))
C.append(code(r'''
from sklearn.decomposition import PCA
flat = FMAP.reshape(-1, FMAP.shape[-1]).astype(np.float32)
sat = cv2.cvtColor(cv2.resize(roi05, (FMAP.shape[1], FMAP.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)[..., 1].ravel()
tissue = sat > 25
pca = PCA(3, random_state=0).fit(flat[tissue])
rgb = pca.transform(flat)
lo, hi = np.percentile(rgb[tissue], 1, 0), np.percentile(rgb[tissue], 99, 0)
rgb = np.clip((rgb - lo) / (hi - lo), 0, 1); rgb[~tissue] = 1
rgb = rgb.reshape(FMAP.shape[0], FMAP.shape[1], 3)

fig, ax = plt.subplots(1, 3, figsize=(21, 7.4))
ax[0].imshow(roi05); ax[0].set_title("조직 사진 (0.5 µm/px)")
ax[1].imshow(rgb, interpolation="nearest"); ax[1].set_title(f"H-optimus-0 토큰 맵 PCA ({FMAP.shape[0]}×{FMAP.shape[1]} 토큰, 1칸 = 7 µm)")
ax[2].imshow(roi05); ax[2].imshow(cv2.resize((rgb * 255).astype(np.uint8), roi05.shape[1::-1], interpolation=cv2.INTER_NEAREST), alpha=0.55)
ax[2].set_title("겹쳐 보기")
[a.axis("off") for a in ax]; plt.tight_layout(); plt.show()
'''))

C.append(md(r'''
## A-4. 좌표로 세포 임베딩 꺼내기

세포 중심 좌표(0.25 µm/px, ROI 기준)를 토큰 맵 좌표로 바꿔 값을 꺼냅니다. 두 가지 방식을 만듭니다.

| 이름 | 방법 | 담는 정보 |
|---|---|---|
| `emb_h0` | 세포 중심 위치의 값을 주변 4개 토큰에서 **bilinear 보간** | 그 세포 자리 (약 7 µm) |
| `emb_h0_mask` | **그 세포의 핵 윤곽(+ 세포질 몫 `DILATE_UM`)** 과 겹친 토큰들을 **겹친 면적 비율로 가중 평균** | **딱 그 세포 크기만큼** |
| `emb_h0_ctx` | 3×3 토큰 평균 맵에서 같은 방식으로 보간 | 세포 + 주변 약 20 µm 문맥 |

**세포 하나는 몇 µm일까?** 세포핵 지름은 림프구 5~7 µm, 섬유아세포 3×10 µm(길쭉함), 종양세포 10~20 µm 정도이고, 
세포질까지 포함하면 림프구 7~10 µm, 종양세포·대식세포 15~30 µm입니다. H&E에서 세포질 경계는 잘 보이지 않아 CellViT는 **핵**을 분할합니다. 
그래서 고정된 크기 대신 **세포마다 분할된 핵 윤곽을 그대로 쓰고**, 세포질 몫으로 `DILATE_UM`(기본 2 µm)만큼 넓혀서 그 영역의 토큰만 섞습니다. 
H-optimus-0 토큰 1칸(7 µm)이 핵 하나 크기라, 보통 핵 하나가 토큰 1~4개에 걸치고 그 비율대로 평균됩니다.

```
세포 중심 (x, y) @ 0.25 µm/px  →  ÷2  →  (x', y') @ 0.5 µm/px  →  토큰 좌표 = (x' / 14 − 0.5, y' / 14 − 0.5)
```
'''))
C.append(code(r'''
def sample_tokens(fmap, xy05, smooth=1):
    # fmap에서 0.5 µm/px 좌표 xy05 (N, 2)의 값을 bilinear 보간으로 꺼낸다. smooth=3이면 3×3 평균 맵에서
    t = torch.from_numpy(fmap.astype(np.float32)).permute(2, 0, 1)[None]          # (1, D, gh, gw)
    if smooth > 1:
        t = F.avg_pool2d(t, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
    gh, gw = fmap.shape[:2]
    u = (xy05[:, 0] / TOK - 0.5) / (gw - 1) * 2 - 1                                 # 토큰 중심 기준 정규화 좌표
    v = (xy05[:, 1] / TOK - 0.5) / (gh - 1) * 2 - 1
    grid = torch.tensor(np.stack([u, v], 1), dtype=torch.float32)[None, :, None]   # (1, N, 1, 2)
    return F.grid_sample(t, grid, mode="bilinear", align_corners=True)[0, :, :, 0].T.numpy()   # (N, D)

DILATE_UM = 2.0   # 핵 윤곽을 바깥으로 넓힐 크기 (세포질 몫). 0이면 핵만

def mask_pool(fmap, contours05, dilate_px):
    # 세포마다: 핵 윤곽(0.5 µm/px 좌표)을 그 주변 토큰 격자 위에 픽셀 단위로 그리고,
    # 토큰 칸(14×14 px)별로 겹친 픽셀 수를 세어 가중치로 → 토큰 가중 평균 = 세포 크기만큼의 임베딩
    gh, gw, D = fmap.shape
    out = np.zeros((len(contours05), D), np.float32)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)) if dilate_px > 0 else None
    for n, cnt in enumerate(contours05):
        x0, y0 = np.floor(cnt.min(0)).astype(int) - dilate_px; x1, y1 = np.ceil(cnt.max(0)).astype(int) + dilate_px
        tx0, ty0 = max(x0 // TOK, 0), max(y0 // TOK, 0)
        tx1, ty1 = min(x1 // TOK + 1, gw), min(y1 // TOK + 1, gh)
        m = np.zeros(((ty1 - ty0) * TOK, (tx1 - tx0) * TOK), np.uint8)
        cv2.fillPoly(m, [np.round(cnt - (tx0 * TOK, ty0 * TOK)).astype(np.int32)], 1)
        if kern is not None:
            m = cv2.dilate(m, kern)
        w = m.reshape(ty1 - ty0, TOK, tx1 - tx0, TOK).sum((1, 3)).astype(np.float32)   # 토큰별 겹친 픽셀 수
        if w.sum() == 0:
            continue
        out[n] = (fmap[ty0:ty1, tx0:tx1].astype(np.float32) * w[..., None]).sum((0, 1)) / w.sum()
    return out

xy = np.array([c["centroid"] for c in CELLS])            # ROI 좌표, 0.25 µm/px
t = time.time()
E_H0 = sample_tokens(FMAP, xy / 2)                       # 0.5 µm/px로
E_H0_CTX = sample_tokens(FMAP, xy / 2, smooth=3)
E_H0_MASK = mask_pool(FMAP, [c["contour"] / 2 for c in CELLS], dilate_px=int(round(DILATE_UM / (MPP * 2))))
E_CV = np.stack([c["emb_cellvit"] for c in CELLS])
types = np.array([c["type"] for c in CELLS])
print(f"세포 {len(xy):,}개 임베딩 꺼내기 {time.time() - t:.2f}s (모델 추론 없음) | H-optimus-0 {E_H0.shape} · CellViT {E_CV.shape}")

cells_df = pd.DataFrame({"x": xy[:, 0] + ROI_X, "y": xy[:, 1] + ROI_Y, "cellvit_type": types})   # 슬라이드 level-0 좌표
np.savez_compressed(OUT / "cell_embeddings.npz", xy_level0=cells_df[["x", "y"]].values, type=types,
                    h0=E_H0.astype(np.float16), h0_mask=E_H0_MASK.astype(np.float16), h0_ctx=E_H0_CTX.astype(np.float16),
                    cellvit=E_CV.astype(np.float16))
cells_df.head()
'''))

C.append(md(r'''
## A-5. 세포 임베딩 보기

### 세포 UMAP: H-optimus-0 vs CellViT
색은 CellViT가 예측한 PanNuke 타입입니다. CellViT 임베딩은 이 타입을 맞히도록 학습된 공간이라 타입별로 잘 모이는 게 당연하고,
H-optimus-0는 세포 타입 라벨 없이 학습했는데도 타입별 구조가 보이는지가 관심사입니다.
'''))
C.append(code(r'''
import umap
fig, ax = plt.subplots(1, 3, figsize=(22, 6.8))
for a, (name, E) in zip(ax, [("H-optimus-0 (세포 크기 마스크)", E_H0_MASK), ("H-optimus-0 (주변 문맥 포함)", E_H0_CTX), ("CellViT", E_CV)]):
    U = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=0).fit_transform(E.astype(np.float32))
    for k, col in PAN_COL.items():
        m = types == k
        if m.any():
            a.scatter(U[m, 0], U[m, 1], s=3, color=np.array(col) / 255, label=f"{k} ({m.sum()})")
    a.set_title(name); a.set_xticks([]); a.set_yticks([])
ax[-1].legend(markerscale=5, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout(); plt.show()
'''))
C.append(md(r'''
### 비슷한 세포 찾기
같은 질의 세포에 대해 두 잠재공간에서 코사인 유사도가 가장 높은 세포들을 모았습니다 (원본 조직 사진에서 잘라 표시).
'''))
C.append(code(r'''
def nearest(E, q, k=11):
    En = E.astype(np.float32); En /= np.linalg.norm(En, axis=1, keepdims=True)
    s = En @ En[q]; s[q] = -1
    return np.argsort(-s)[:k]

read_roi = lambda x, y, s: np.array(slide.read_region((int(x) + ROI_X, int(y) + ROI_Y), 0, (s, s)).convert("RGB"))
rng = np.random.default_rng(1)
queries = {k: rng.choice(np.flatnonzero(types == k)) for k in ["Neoplastic", "Inflammatory", "Connective"] if (types == k).any()}
for k, q in queries.items():
    picked = []
    for name, E in [("H-optimus-0", E_H0_MASK), ("CellViT", E_CV)]:
        for i in np.concatenate([[q], nearest(E, q)]):
            picked.append(dict(CELLS[i], row=f"{name}", tag=i))
    viz.cell_gallery(picked, lambda c: c["row"], lambda c: PAN_COL[c["type"]], order=["H-optimus-0", "CellViT"],
                     get_crop=lambda x, y, s: read_roi(x, y, s), n=12, size=72, seed=0,
                     title=f"질의 세포(각 줄 맨 앞, {k})와 가장 비슷한 세포 11개 — 윤곽 색 = CellViT 타입")
'''))
C.append(md(r'''
> `cell_gallery`는 줄 안에서 무작위로 섞어 보여 줍니다. 각 줄은 질의 세포 1개 + 그 모델이 찾은 이웃 11개입니다.

### 세포 군집을 조직 위에
H-optimus-0 세포 임베딩을 k-means로 묶고, 군집을 조직 사진 위에 칠합니다.
세포 자체 + 주변 문맥이 섞인 임베딩이라 **"어떤 환경에 있는 어떤 세포인가"** 로 나뉘는 경향이 있습니다.
'''))
C.append(code(r'''
from sklearn.cluster import KMeans
K = 6
km = KMeans(K, n_init=10, random_state=0).fit(E_H0_CTX.astype(np.float32) / np.linalg.norm(E_H0_CTX.astype(np.float32), axis=1, keepdims=True))
pal = plt.get_cmap("tab10")
KCOL = {f"군집 {c}": tuple(int(255 * v) for v in pal(c)[:3]) for c in range(K)}
for c_, lab in zip(CELLS, km.labels_):
    c_["cluster"] = f"군집 {lab}"

fig, ax = plt.subplots(1, 2, figsize=(20, 9.6), gridspec_kw={"width_ratios": [1.15, 1]})
view = roi_img[::2, ::2]
ax[0].imshow(view)
for k in KCOL:
    m = np.array([c["cluster"] == k for c in CELLS])
    ax[0].scatter(xy[m, 0] / 2, xy[m, 1] / 2, s=2, color=np.array(KCOL[k]) / 255, label=f"{k} ({m.sum()})")
ax[0].legend(markerscale=6, loc="upper left", bbox_to_anchor=(1.01, 1)); ax[0].axis("off"); ax[0].set_title("H-optimus-0 세포 군집 (주변 문맥 포함 임베딩)")
ct = pd.crosstab(pd.Series([c["cluster"] for c in CELLS], name="군집"), pd.Series(types, name="CellViT 타입"), normalize="index")
im = ax[1].imshow(ct.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax[1].text(j, i, f"{ct.values[i, j]:.0%}", ha="center", va="center", color="w" if ct.values[i, j] > 0.6 else "k")
ax[1].set_xticks(range(ct.shape[1])); ax[1].set_xticklabels(ct.columns, rotation=20); ax[1].set_yticks(range(ct.shape[0])); ax[1].set_yticklabels(ct.index)
ax[1].set_title("군집별 CellViT 타입 구성 (행 = 100%)")
plt.tight_layout(); plt.show()
viz.cell_gallery(CELLS, lambda c: c["cluster"], lambda c: KCOL[c["cluster"]], order=list(KCOL), get_crop=lambda x, y, s: read_roi(x, y, s),
                 n=12, size=72, title="군집별 세포 (원본 조직)")
'''))

C.append(md(r'''
> **이 예제에서 보이는 것**: CellViT는 림프 응집의 세포를 거의 모두 `Inflammatory` 하나로 부르지만, 
> H-optimus-0 세포 군집은 같은 림프구를 **응집 중심부**와 **바깥 테두리**처럼 위치(미세환경)에 따라 다른 군집으로 나눕니다. 
> 세포 자체의 모양뿐 아니라 주변 조직 문맥이 임베딩에 들어 있기 때문입니다. 세포 타입 분류에는 방해가 될 수 있지만, 
> 공간 생물학적 상태(예: 배중심 vs 외투층)를 구분하는 데는 유용할 수 있습니다.
'''))

# ── Part B
C.append(md(r'''
# Part B. 품질 평가: 세포 분류로 비교하기

**어느 잠재공간이 세포 타입 정보를 더 잘 담고 있을까?** 세포 라벨이 있는 데이터에서 같은 세포에 대해 임베딩을 뽑고,
같은 선형 분류기(로지스틱 회귀)로 세포 타입을 맞혀 봅니다.

- 데이터: **NuCLS** (유방암, CC0) — 02 노트북과 같은 CellViT++ 예제. 256×256 px(40x) 이미지 train 10장 / test 5장, 세포 중심 점 라벨 4종
- 같은 세포 비교를 위해 CellViT가 검출한 세포와 정답 점을 짝지은(거리 ≤ 12 px) 세포만 씁니다.
- H-optimus-0 토큰 맵은 이미지를 0.5 µm/px(128 px)로 줄이고 흰 여백을 붙여 224 px 한 장으로 계산합니다.

> 데이터가 작아서(세포 수백 개) 수치의 오차가 큽니다. 어느 정도 경향만 보세요.
'''))
C.append(code(r'''
NUCLS_DIR = WORKDIR / "cellvitpp_repo/test_database/training_database/Example-Detection"
if not NUCLS_DIR.exists():
    !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/TIO-IKIM/CellViT-plus-plus.git cellvitpp_repo
    !cd cellvitpp_repo && git sparse-checkout set test_database/training_database/Example-Detection
NUCLS = {0: "Tumor", 1: "nonTIL Stromal", 2: "sTIL", 3: "Other"}
NCOL = {0: (220, 30, 30), 1: (34, 139, 34), 2: (30, 90, 255), 3: (150, 150, 150)}

# CellViT를 다시 GPU에 (H-optimus-0와 번갈아)
h0 = h0.cpu(); torch.cuda.empty_cache()
ck = torch.load(cache_cellvit_sam_h(), map_location="cpu", weights_only=False)
cellvit = CellViTSAM(model_path=None, num_nuclei_classes=cconf["data"]["num_nuclei_classes"],
                     num_tissue_classes=cconf["data"]["num_tissue_classes"], vit_structure=cconf["model"]["backbone"],
                     regression_loss=cconf["model"].get("regression_loss", False))
cellvit.load_state_dict(ck["model_state_dict"]); cellvit = cellvit.eval().to(DEVICE); del ck

from scipy.optimize import linear_sum_assignment
def nucls_split(split):
    rows = []
    for p in sorted((NUCLS_DIR / split / "images").glob("*.png")):
        img = np.array(Image.open(p).convert("RGB"))
        lbl = pd.read_csv(NUCLS_DIR / split / "labels" / f"{p.stem}.csv", header=None, names=["x", "y", "label"])
        det = cellvit_cells(img)
        if not det:
            continue
        dc = np.array([c["centroid"] for c in det])
        d = np.linalg.norm(lbl[["x", "y"]].values[:, None] - dc[None], axis=-1)
        gi, di = linear_sum_assignment(d)
        for g, k in zip(gi, di):
            if d[g, k] <= 12:
                rows.append(dict(image=p.stem, img=img, x=lbl.x[g], y=lbl.y[g], label=int(lbl.label[g]), emb_cellvit=det[k]["emb_cellvit"],
                                 contour=det[k]["contour"]))
    return rows

TR, TE = nucls_split("train"), nucls_split("test")
cellvit = None; gc.collect(); torch.cuda.empty_cache(); h0 = h0.to(DEVICE)
print(f"짝지은 세포: train {len(TR)} · test {len(TE)}")

def add_h0(rows):
    # 이미지마다 토큰 맵을 한 번 계산하고, 그 이미지의 세포들은 좌표로 꺼내기만
    for name in sorted({r["image"] for r in rows}):
        rs = [r for r in rows if r["image"] == name]
        img05 = cv2.resize(rs[0]["img"], (rs[0]["img"].shape[1] // 2, rs[0]["img"].shape[0] // 2), interpolation=cv2.INTER_AREA)
        fm_ = token_map(img05)
        pts = np.array([[r["x"], r["y"]] for r in rs], float) / 2
        em = mask_pool(fm_, [r["contour"] / 2 for r in rs], dilate_px=int(round(DILATE_UM / 0.5)))
        for r, e, ec, ek in zip(rs, sample_tokens(fm_, pts), sample_tokens(fm_, pts, smooth=3), em):
            r["emb_h0"], r["emb_h0_ctx"], r["emb_h0_mask"] = e, ec, ek
add_h0(TR); add_h0(TE)
'''))
C.append(code(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score

ytr = np.array([r["label"] for r in TR]); yte = np.array([r["label"] for r in TE])
def feats(rows, kind):
    if kind == "CellViT + H-optimus-0":
        return np.hstack([feats(rows, "CellViT"), feats(rows, "H-optimus-0 (세포 크기 마스크)")])
    key = {"CellViT": "emb_cellvit", "H-optimus-0 (세포 중심점)": "emb_h0", "H-optimus-0 (세포 크기 마스크)": "emb_h0_mask",
           "H-optimus-0 (주변 문맥 포함)": "emb_h0_ctx"}[kind]
    return np.stack([r[key] for r in rows]).astype(np.float32)

KINDS = ["CellViT", "H-optimus-0 (세포 중심점)", "H-optimus-0 (세포 크기 마스크)", "H-optimus-0 (주변 문맥 포함)", "CellViT + H-optimus-0"]
res, PRED = {}, {}
for kind in KINDS:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05, class_weight="balanced"))
    clf.fit(feats(TR, kind), ytr)
    p = clf.predict(feats(TE, kind)); PRED[kind] = p
    # 작은 데이터라 bootstrap으로 신뢰구간도
    r_ = np.random.default_rng(0); boots = []
    for _ in range(500):
        ii = r_.integers(0, len(yte), len(yte))
        boots.append(f1_score(yte[ii], p[ii], average="macro", labels=list(NUCLS)))
    res[kind] = dict(macro_f1=f1_score(yte, p, average="macro", labels=list(NUCLS)), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                     bal_acc=balanced_accuracy_score(yte, p), dim=feats(TR, kind).shape[1])
R = pd.DataFrame(res).T
display(R.round(3))

COLS = ["#c2185b", "#f0a868", "#d0602a", "#e8c33d", "#555555"]
fig, ax = plt.subplots(figsize=(12, 4.8))
x = np.arange(len(R))
ax.bar(x, R.macro_f1, yerr=[R.macro_f1 - R.lo, R.hi - R.macro_f1], capsize=5, color=COLS, ec="k")
for i, v in enumerate(R.macro_f1):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
ax.set_xticks(x); ax.set_xticklabels([f"{k}\n({int(d)}차원)" for k, d in zip(R.index, R.dim)], fontsize=9)
ax.set_ylabel("test macro-F1 (오차막대 = bootstrap 95% 구간)"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"NuCLS 세포 4종 분류 (train {len(ytr)} · test {len(yte)}개 세포)")
plt.tight_layout(); plt.show()
best = R.macro_f1.idxmax()
viz.classification_metrics(yte, PRED[best], NUCLS, NCOL, title=f"가장 좋은 임베딩: {best}")
'''))
C.append(md(r'''
> **결과 읽는 법**: 이 예제(테스트 세포 156개)에서는 모든 임베딩의 macro-F1이 0.5 안팎이고 **신뢰구간이 크게 겹쳐** 우열을 가릴 수 없습니다. 
> 즉 조직 패치용으로 학습한 H-optimus-0를 좌표로 꺼내 써도 세포 분류용 CellViT 임베딩과 **비슷한 수준의 세포 정보**를 담고 있다는 정도까지만 말할 수 있습니다. 
> 실제 비교는 수천 개 이상의 라벨 세포로, 이미지(환자) 단위 교차검증을 해야 합니다.
'''))

C.append(md(r'''
## 정리

| 단계 | 핵심 |
|---|---|
| 토큰 맵 | 슬라이드(또는 영역)를 0.5 µm/px로 맞추고, 224 px 타일을 절반 겹쳐 훑어 **가운데 8×8 토큰만** 기록 → 경계 없는 맵, **한 번만 계산·저장** |
| 좌표 변환 | 0.25 µm/px 좌표 ÷ 2 → ÷ 14 − 0.5 = 토큰 좌표, bilinear 보간으로 꺼내기 |
| 세포 임베딩 | 중심점 보간(`emb_h0`), **세포 크기 마스크 가중 평균(`emb_h0_mask`)**, 3×3 평균(`emb_h0_ctx`, 주변 문맥 포함) |
| 저장 | `h0_token_map.npy` + 좌표 정보 json → 나중에 모델 없이 어떤 좌표든 꺼낼 수 있음 |

**주의할 점**
- H-optimus-0 토큰 1칸은 7 µm로 세포핵 하나 크기입니다. 세포가 빽빽한 곳에서는 이웃 세포 정보가 섞입니다.
- H-optimus-0는 세포 단위로 학습한 모델이 아니므로, 세포 타입 구분은 CellViT 같은 세포 모델 임베딩이 더 나을 수 있습니다.
  대신 조직 문맥 정보가 풍부해서 **두 임베딩을 이어 붙이는 것**이 유용할 수 있습니다 (Part B 결과 참고).
- 슬라이드 전체에 적용할 때는 토큰 맵이 커지므로 (6 mm 영역 ≈ 2.6 GB, fp16) 조직이 있는 타일만 저장하거나 zarr 같은 청크 저장 형식을 쓰세요.
- 같은 방식이 UNI-2(토큰 14 px), Virchow2(토큰 14 px)에도 그대로 적용됩니다 — 모델 로딩과 정규화 값, `num_prefix_tokens`만 바꾸면 됩니다.
'''))

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
               "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, NB)
print("wrote", NB)
