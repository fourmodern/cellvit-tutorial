"""CellViT / CellViT++ 튜토리얼 노트북 생성 스크립트.

python build_notebooks.py  →  01_CellViT_tasks.ipynb, 02_CellViT_plusplus.ipynb
"""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


# ─────────────────────────────────────────────────────────────
# 공통 셀
# ─────────────────────────────────────────────────────────────
def setup_cells():
    return [
        md(
            "## 0. 환경 설정\n"
            "\n"
            "**Colab 사용 시**\n"
            "1. 메뉴 `런타임 → 런타임 유형 변경 → T4 GPU` 선택\n"
            "2. 아래 설치 셀을 실행하면 패키지 설치 후 **런타임이 자동으로 재시작**됩니다 "
            "(cellvit이 `numpy<2`를 요구하기 때문).\n"
            "3. 재시작 후 **처음부터 다시 실행**하세요. 설치 셀은 자동으로 건너뜁니다.\n"
            "\n"
            "로컬 Jupyter에서는 `pip install cellvit openslide-bin umap-learn`이 된 커널이면 설치 셀이 아무것도 하지 않습니다."
        ),
        code(
            "# 0-1. 설치 (Colab에서 최초 1회)\n"
            "import sys, os, importlib.util\n"
            "\n"
            'IN_COLAB = "google.colab" in sys.modules\n'
            'if IN_COLAB and importlib.util.find_spec("cellvit") is None:\n'
            "    !apt-get -qq install -y openslide-tools > /dev/null\n"
            "    !pip -q install cellvit openslide-bin umap-learn\n"
            '    print("설치 완료 → 런타임을 재시작합니다. 재시작 후 처음부터 다시 실행하세요.")\n'
            "    os.kill(os.getpid(), 9)  # Colab 런타임 재시작"
        ),
        code(
            "# 0-2. 경로와 옵션\n"
            "from pathlib import Path\n"
            "\n"
            'WORKDIR = Path("/content/cellvit_tutorial") if IN_COLAB else Path.cwd()\n'
            "WORKDIR.mkdir(parents=True, exist_ok=True)\n"
            "os.chdir(WORKDIR)\n"
            "# 로컬 venv 커널에서도 cellvit-* CLI를 찾도록 현재 파이썬의 bin 경로를 PATH 앞에 추가\n"
            'os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]\n'
            "\n"
            "# 모델 가중치(SAM-H 약 2.8GB)를 Google Drive에 캐시하면 다음 세션에서 재다운로드하지 않습니다.\n"
            "USE_DRIVE_CACHE = False\n"
            "if IN_COLAB and USE_DRIVE_CACHE:\n"
            "    from google.colab import drive\n"
            '    drive.mount("/content/drive")\n'
            '    os.environ["CELLVIT_CACHE"] = "/content/drive/MyDrive/cellvit_cache"\n'
            "else:\n"
            '    os.environ["CELLVIT_CACHE"] = str(WORKDIR / "models")  # cellvit import 전에 설정해야 함\n'
            "\n"
            '# "SAM"  : CellViT-SAM-H  (논문 최고 성능, encoder = SAM ViT-H, embedding 1280차원)\n'
            '# "HIPT" : CellViT-256    (encoder = HIPT ViT-S/16, embedding 384차원, 가볍고 빠름)\n'
            'MODEL_NAME = "SAM"\n'
            'print("WORKDIR:", WORKDIR, "| CACHE:", os.environ["CELLVIT_CACHE"], "| MODEL:", MODEL_NAME)'
        ),
    ]


HELPERS = '''# 공통 유틸 함수
import json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import openslide

from cellvit.utils.tools import unflatten_dict
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256

warnings.filterwarnings("ignore")

# 한글 폰트: 시스템에 NanumGothic이 없으면 Google Fonts에서 받아 등록
import matplotlib.font_manager as fm, urllib.request
_font = WORKDIR / "NanumGothic-Regular.ttf"
if not _font.exists():
    urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf", _font)
fm.fontManager.addfont(str(_font))
plt.rcParams["font.family"] = fm.FontProperties(fname=str(_font)).get_name()
plt.rcParams["axes.unicode_minus"] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_cellvit(ckpt_path, device=DEVICE):
    """CellViT 체크포인트 → (model, config). cellvit.inference.CellViTInference._load_model과 동일한 방식."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    conf = unflatten_dict(ckpt["config"], ".")
    common = dict(
        num_nuclei_classes=conf["data"]["num_nuclei_classes"],
        num_tissue_classes=conf["data"]["num_tissue_classes"],
        regression_loss=conf["model"].get("regression_loss", False),
    )
    if ckpt["arch"] == "CellViTSAM":
        model = CellViTSAM(model_path=None, vit_structure=conf["model"]["backbone"], **common)
    else:
        model = CellViT256(model256_path=None, **common)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device), conf


def to_tensor(img, conf):
    """RGB uint8 (H, W, 3) → 정규화된 (1, 3, H, W) 텐서"""
    norm = conf["transformations"].get("normalize", {})
    mean = np.array(norm.get("mean", (0.5, 0.5, 0.5)))
    std = np.array(norm.get("std", (0.5, 0.5, 0.5)))
    x = (img / 255.0 - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)).float()[None]


@torch.inference_mode()
def run_cellvit(model, conf, img):
    """패치 하나에 대해 forward. H, W는 16의 배수여야 함 (40x, 0.25 µm/px 기준)."""
    x = to_tensor(img, conf).to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        out = model(x, retrieve_tokens=True)
    return {k: v.float().cpu() for k, v in out.items()}


def postprocess(model, out, magnification=40):
    """NP/HV/NT 맵 → (instance map, 세포 리스트). 배경(type 0)으로 분류된 인스턴스는 제외."""
    pred = {
        "nuclei_binary_map": F.softmax(out["nuclei_binary_map"], dim=1),
        "nuclei_type_map": F.softmax(out["nuclei_type_map"], dim=1),
        "hv_map": out["hv_map"],
    }
    inst_map, cell_dicts = model.calculate_instance_map(pred, magnification=magnification)
    cells = [dict(c, id=int(i)) for i, c in cell_dicts[0].items() if c["type"] != 0]
    return inst_map[0].numpy().astype(np.int32), cells


def cell_embeddings(tokens, cells, token_size=16):
    """세포 bbox가 덮는 ViT 토큰을 평균 → 세포당 1개 벡터 (CellViT++와 동일한 방식)
    tokens: (D, H/16, W/16), bbox: [[row_min, col_min], [row_max, col_max]]"""
    embs = []
    for c in cells:
        bb = np.asarray(c["bbox"]) / token_size
        r0, c0 = np.floor(bb[0]).astype(int)
        r1, c1 = np.ceil(bb[1]).astype(int)
        embs.append(tokens[:, r0:r1, c0:c1].flatten(1).mean(1))
    return torch.stack(embs) if embs else torch.empty(0, tokens.shape[0])


TARGET_MPP = 0.25  # CellViT 학습 해상도 (40x)


def read_patch_40x(slide, x, y, size=1024):
    """level-0 좌표 (x, y)부터 0.25 µm/px 기준 size×size 패치를 읽음.
    20x(0.5 µm/px) 슬라이드라면 512×512를 읽어 2배 업샘플 → CellViT WSI 파이프라인과 같은 처리."""
    mpp = float(slide.properties.get("openslide.mpp-x", TARGET_MPP))
    scale = mpp / TARGET_MPP
    read = int(round(size / scale))
    im = np.array(slide.read_region((x, y), 0, (read, read)).convert("RGB"))
    if read != size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
    return im, scale


def draw_cells(img, cells, colors, thickness=2):
    """세포 윤곽을 타입별 색으로 그림. colors: {type_id: (R, G, B)}"""
    canvas = img.copy()
    for c in cells:
        cnt = np.asarray(c["contour"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(canvas, [cnt], -1, colors.get(c["type"], (0, 0, 0)), thickness)
    return canvas


def legend(ax, names, colors, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(color=np.array(colors[i]) / 255, label=n) for i, n in names.items() if i in colors]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, **kw)

print("device:", DEVICE, "| torch", torch.__version__)
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")'''

DOWNLOAD = '''# 모델 가중치 + 예제 WSI 다운로드 (Zenodo, 처음 한 번만)
from cellvit.utils.cache_models import cache_cellvit_sam_h, cache_cellvit_256
from cellvit.utils.cache_test_database import cache_test_database

CKPT_PATH = cache_cellvit_sam_h() if MODEL_NAME == "SAM" else cache_cellvit_256()
cache_test_database(run_dir=str(WORKDIR))  # → WORKDIR/test_database (x40_svs, x20_svs, BRACS, ...)

!ls -lh {CKPT_PATH}
!ls test_database/*'''

PANNUKE_COLORS = '''# PanNuke 세포 타입 색상 (CellViT 논문 그림과 동일 계열)
PANNUKE_COLORS = {
    1: (255, 0, 0),     # Neoplastic   - 빨강
    2: (34, 221, 77),   # Inflammatory - 초록
    3: (35, 92, 236),   # Connective   - 파랑
    4: (254, 255, 0),   # Dead         - 노랑
    5: (255, 159, 68),  # Epithelial   - 주황
}'''


# ─────────────────────────────────────────────────────────────
# 01. CellViT
# ─────────────────────────────────────────────────────────────
def notebook_cellvit():
    cells = [
        md(
            "# CellViT 튜토리얼: 하나의 모델로 수행하는 5가지 task\n"
            "\n"
            "**CellViT** (Hörst et al., *Medical Image Analysis* 2024)는 H&E 이미지에서 "
            "세포핵을 **찾고 → 분할하고 → 분류하는** Vision Transformer 기반 모델입니다.\n"
            "\n"
            "```\n"
            "              ┌─ [CLS] token ─────────────► ① 조직 타입 분류 (19종)\n"
            "  H&E 패치 ─► ViT Encoder ─┬─ NP decoder ─► ② 세포핵 / 배경 (binary)\n"
            "  (1024²)     (SAM-H or    ├─ HV decoder ─► ③ 수평·수직 거리맵 → 붙은 핵 분리\n"
            "              HIPT-256)    └─ NT decoder ─► ④ 픽셀별 세포 타입 (PanNuke 5종)\n"
            "                    │\n"
            "                    └─ patch tokens ────────► ⑤ 세포별 임베딩 (후속 분석용)\n"
            "```\n"
            "\n"
            "| # | Task | 출력 | 이 노트북 섹션 |\n"
            "|---|---|---|---|\n"
            "| 1 | 조직 타입 분류 | 패치당 19개 조직 확률 | §3 |\n"
            "| 2 | 세포핵 검출 | 세포 중심 좌표 | §5 |\n"
            "| 3 | 세포핵 인스턴스 분할 | 세포별 윤곽(contour) | §5 |\n"
            "| 4 | 세포핵 분류 | Neoplastic / Inflammatory / Connective / Dead / Epithelial | §6 |\n"
            "| 5 | 세포 임베딩 추출 | 세포당 1280차원(SAM-H) 벡터 | §7 |\n"
            "\n"
            "> WSI 전체 추론, 분류기 교체, 나만의 분류기 학습은 `02_CellViT_plusplus.ipynb`에서 다룹니다.\n"
            "\n"
            "**참고**: 모델은 **40x (≈0.25 µm/px)** H&E로 학습되었습니다. "
            "코드/가중치 라이선스는 Apache 2.0 + **Commons Clause**(상업적 이용 제한)입니다."
        ),
        *setup_cells(),
        code(DOWNLOAD),
        code(HELPERS),
        code(PANNUKE_COLORS),
        # 1. 모델
        md(
            "## 1. 모델 불러오기\n"
            "\n"
            "체크포인트에는 가중치와 함께 학습 설정(`config`)이 들어 있습니다. "
            "여기서 세포 타입·조직 타입 이름과 입력 정규화 값을 꺼내 씁니다."
        ),
        code(
            "model, conf = load_cellvit(CKPT_PATH)\n"
            "\n"
            'NUCLEI_TYPES = {v: k for k, v in conf["dataset_config"]["nuclei_types"].items()}  # {0: "Background", 1: "Neoplastic", ...}\n'
            'TISSUE_TYPES = {v: k for k, v in conf["dataset_config"]["tissue_types"].items()}\n'
            "\n"
            "n_params = sum(p.numel() for p in model.parameters()) / 1e6\n"
            'print(f"아키텍처      : {type(model).__name__}  ({n_params:.0f}M params)")\n'
            'print(f"임베딩 차원   : {model.embed_dim}")\n'
            'print(f"세포 타입     : {NUCLEI_TYPES}")\n'
            'print(f"조직 타입 ({len(TISSUE_TYPES)}) : {list(TISSUE_TYPES.values())}")'
        ),
        # 2. 패치
        md(
            "## 2. 예제 패치 준비\n"
            "\n"
            "예제 WSI는 OpenSlide 테스트 데이터의 **피부 조직** (`CMU-1-Small-Region.svs`, Aperio, **20x = 0.5 µm/px**)입니다. "
            "표피(상피)와 진피(결합조직·염증세포)가 함께 있는 영역을 골랐습니다.\n"
            "\n"
            "CellViT는 **40x (0.25 µm/px)** 로 학습되었으므로, 20x 슬라이드는 512×512를 읽어 **2배 업샘플**한 1024×1024를 넣습니다. "
            "(`cellvit-inference`도 내부에서 똑같이 리샘플링합니다.)\n"
            "\n"
            "> 내 슬라이드로 바꾸려면 `WSI_PATH`와 좌표 `X0, Y0`만 바꾸면 됩니다."
        ),
        code(
            'WSI_PATH = WORKDIR / "test_database/x20_svs/CMU-1-Small-Region.svs"\n'
            "X0, Y0 = 768, 768  # level-0(20x) 좌표: 표피-진피 경계\n"
            "\n"
            "slide = openslide.OpenSlide(str(WSI_PATH))\n"
            'print("슬라이드 크기 (level 0):", slide.dimensions, "| mpp:", slide.properties.get("openslide.mpp-x"),\n'
            '      "| 배율:", slide.properties.get("openslide.objective-power"))\n'
            "\n"
            "img, scale = read_patch_40x(slide, X0, Y0, 1024)\n"
            "src = 1024 / scale  # 원본 슬라이드에서 읽은 크기\n"
            'print(f"원본 {src:.0f}×{src:.0f} px 읽음 → ×{scale:.1f} 업샘플 → 입력 {img.shape}")\n'
            "\n"
            "thumb = np.array(slide.get_thumbnail((800, 800)).convert(\"RGB\"))\n"
            "ds = slide.dimensions[0] / thumb.shape[1]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(14, 7))\n"
            "ax[0].imshow(thumb)\n"
            'ax[0].add_patch(plt.Rectangle((X0 / ds, Y0 / ds), src / ds, src / ds, fill=False, ec="lime", lw=2))\n'
            'ax[0].set_title("WSI 썸네일 (초록 박스 = 선택한 패치)")\n'
            "ax[1].imshow(img)\n"
            'ax[1].set_title("CellViT 입력: 1024×1024 @ 0.25 µm/px")\n'
            '[a.axis("off") for a in ax]\n'
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 추론: forward 한 번으로 모든 출력 얻기\n"
            "\n"
            "`retrieve_tokens=True`를 주면 디코더 출력과 함께 encoder 마지막 층의 patch token도 받습니다."
        ),
        code(
            "t = time.time()\n"
            "out = run_cellvit(model, conf, img)\n"
            'print(f"추론 시간: {time.time() - t:.2f}s\\n")\n'
            "for k, v in out.items():\n"
            '    print(f"{k:20s} {tuple(v.shape)}")'
        ),
        # 3. tissue
        md(
            "## 3. Task ① 조직 타입 분류\n"
            "\n"
            "ViT의 `[CLS]` 토큰 위에 붙은 linear head가 PanNuke의 19개 조직 중 하나를 예측합니다. "
            "학습 시 보조(auxiliary) task로 쓰인 출력이라 정밀한 조직 분류기로 쓰기보다는 **품질 확인용**으로 보는 게 좋습니다.\n"
            "\n"
            "> 이 예제에서는 실제 조직이 **피부(Skin)** 인데 Skin을 상위권에 올리지 못합니다. "
            "1024px 패치 하나만 보는 데다 20x를 업샘플한 입력이라 문맥이 부족하기 때문입니다. "
            "조직 타입이 중요하면 슬라이드 메타데이터나 별도의 조직 분류 모델을 쓰세요."
        ),
        code(
            'prob = F.softmax(out["tissue_types"][0], dim=0).numpy()\n'
            "top = np.argsort(prob)[::-1][:5]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 3))\n"
            'ax.barh([TISSUE_TYPES[i] for i in top][::-1], prob[top][::-1], color="steelblue")\n'
            'ax.set_xlabel("확률"); ax.set_title("조직 타입 Top-5 (실제: Skin)")\n'
            "plt.tight_layout(); plt.show()"
        ),
        # 4. raw maps
        md(
            "## 4. 세 디코더의 원시 출력 살펴보기\n"
            "\n"
            "CellViT는 HoVer-Net과 같은 3갈래 디코더 구조를 씁니다.\n"
            "\n"
            "- **NP (Nuclei Pixel)**: 픽셀이 세포핵일 확률\n"
            "- **HV (Horizontal-Vertical)**: 각 픽셀에서 자기 세포핵 중심까지의 수평/수직 거리(−1~1). "
            "핵 경계에서 값이 급변하므로 **붙어 있는 핵을 떼어내는** 단서가 됩니다.\n"
            "- **NT (Nuclei Type)**: 픽셀별 세포 타입 확률"
        ),
        code(
            'np_prob = F.softmax(out["nuclei_binary_map"], 1)[0, 1].numpy()\n'
            'hv = out["hv_map"][0].numpy()\n'
            'nt = out["nuclei_type_map"][0].argmax(0).numpy()\n'
            "\n"
            "nt_rgb = np.zeros((*nt.shape, 3), np.uint8)\n"
            "for t, col in PANNUKE_COLORS.items():\n"
            "    nt_rgb[nt == t] = col\n"
            "\n"
            "crop = (slice(320, 704), slice(400, 784))  # 세포가 많은 영역 확대\n"
            "fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))\n"
            'panels = [(img[crop], "입력", None), (np_prob[crop], "NP: 세포핵 확률", "magma"),\n'
            '          (hv[0][crop], "HV: 수평 거리", "coolwarm"), (hv[1][crop], "HV: 수직 거리", "coolwarm"),\n'
            '          (nt_rgb[crop], "NT: 픽셀별 타입", None)]\n'
            "for a, (im, title, cmap) in zip(ax, panels):\n"
            "    h = a.imshow(im, cmap=cmap)\n"
            '    a.set_title(title); a.axis("off")\n'
            "    if cmap: plt.colorbar(h, ax=a, fraction=0.046)\n"
            "legend(ax[-1], NUCLEI_TYPES, PANNUKE_COLORS)\n"
            "plt.tight_layout(); plt.show()"
        ),
        # 5. instance
        md(
            "## 5. Task ②③ 세포핵 검출 + 인스턴스 분할\n"
            "\n"
            "NP와 HV 맵을 watershed 기반 후처리로 합쳐 **세포핵마다 고유 ID**를 붙입니다. "
            "결과는 세포별로 `centroid`(중심), `contour`(윤곽), `bbox`, `type`, `type_prob`을 가진 딕셔너리입니다."
        ),
        code(
            "inst_map, cells = postprocess(model, out, magnification=40)\n"
            'print(f"검출된 세포핵: {len(cells)}개")\n'
            'print("예시:", {k: (np.round(v, 1).tolist() if k != "contour" else f"{len(v)} points") for k, v in cells[0].items()})\n'
            "\n"
            "rng = np.random.default_rng(0)\n"
            "lut = rng.integers(50, 255, (inst_map.max() + 1, 3)).astype(np.uint8); lut[0] = 0\n"
            'centroids = np.array([c["centroid"] for c in cells])\n'
            "areas = np.array([cv2.contourArea(np.asarray(c['contour'], np.int32)) for c in cells]) * 0.25**2  # µm²\n"
            "\n"
            "fig, ax = plt.subplots(1, 3, figsize=(20, 6.5))\n"
            "ax[0].imshow(img); ax[0].scatter(centroids[:, 0], centroids[:, 1], s=4, c=\"yellow\")\n"
            'ax[0].set_title(f"검출: 세포핵 중심 {len(cells)}개")\n'
            'ax[1].imshow(lut[inst_map]); ax[1].set_title("인스턴스 분할: 세포마다 다른 색")\n'
            '[a.axis("off") for a in ax[:2]]\n'
            'ax[2].hist(areas, bins=50, color="gray"); ax[2].set_xlabel("핵 면적 (µm²)"); ax[2].set_title("핵 크기 분포")\n'
            "plt.tight_layout(); plt.show()"
        ),
        # 6. classification
        md(
            "## 6. Task ④ 세포핵 분류 (PanNuke 5종)\n"
            "\n"
            "각 세포의 타입은 그 세포 영역 안의 NT 픽셀 예측을 **다수결**로 정합니다. "
            "`type_prob`은 다수결 타입이 차지한 픽셀 비율입니다.\n"
            "\n"
            "> **관찰 포인트**: 정상 피부인데도 표피 기저층 세포 일부가 `Neoplastic`(빨강)으로 나옵니다. "
            "PanNuke는 종양 조직 위주로 구성돼 있어 크고 진한 상피 핵을 종양으로 보는 경향이 있습니다. "
            "이런 도메인 차이 때문에 CellViT++에서 분류기를 데이터에 맞게 바꾸는 기능이 중요해집니다."
        ),
        code(
            "overlay = draw_cells(img, cells, PANNUKE_COLORS)\n"
            'types = np.array([c["type"] for c in cells])\n'
            'type_prob = np.array([c["type_prob"] for c in cells])\n'
            "\n"
            "fig = plt.figure(figsize=(20, 8))\n"
            "ax0 = fig.add_subplot(1, 2, 1); ax0.imshow(overlay); ax0.axis(\"off\")\n"
            'ax0.set_title("세포핵 분류 결과"); legend(ax0, NUCLEI_TYPES, PANNUKE_COLORS)\n'
            "\n"
            "ax1 = fig.add_subplot(2, 4, 3)\n"
            "ids = sorted(PANNUKE_COLORS)\n"
            "counts = [(types == i).sum() for i in ids]\n"
            "ax1.bar([NUCLEI_TYPES[i] for i in ids], counts, color=[np.array(PANNUKE_COLORS[i]) / 255 for i in ids], ec=\"k\")\n"
            'ax1.set_title("타입별 세포 수"); ax1.tick_params(axis="x", rotation=45)\n'
            "\n"
            "ax2 = fig.add_subplot(2, 4, 7)\n"
            'ax2.hist(type_prob, bins=20, color="gray"); ax2.set_title("type_prob (다수결 비율)")\n'
            "\n"
            "zoom = (slice(300, 700), slice(300, 700))\n"
            'ax3 = fig.add_subplot(1, 4, 4); ax3.imshow(overlay[zoom]); ax3.axis("off"); ax3.set_title("확대")\n'
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "for i, n in zip(ids, counts):\n"
            '    print(f"{NUCLEI_TYPES[i]:14s} {n:5d}  ({n / len(cells):.1%})")'
        ),
        # 7. embeddings
        md(
            "## 7. Task ⑤ 세포 임베딩 추출\n"
            "\n"
            "encoder 마지막 층의 patch token(16×16 px당 1개) 중 **세포 bbox가 덮는 토큰을 평균**하면 세포당 하나의 벡터가 됩니다. "
            "이 임베딩이 CellViT++에서 분류기를 교체할 때의 입력이고, 세포 그래프(GNN)나 공간 분석의 feature로도 쓸 수 있습니다.\n"
            "\n"
            "아래 UMAP에서 PanNuke 타입별로 군집이 나뉘면, 임베딩이 세포 형태 정보를 담고 있다는 뜻입니다."
        ),
        code(
            'tokens = out["tokens"][0]  # (D, 64, 64)\n'
            "emb = cell_embeddings(tokens, cells).numpy()\n"
            'print("토큰 그리드:", tuple(tokens.shape), "→ 세포 임베딩:", emb.shape)\n'
            "\n"
            "import umap\n"
            "xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 6))\n"
            "for t in ids:\n"
            "    m = types == t\n"
            "    if m.any():\n"
            "        ax.scatter(xy[m, 0], xy[m, 1], s=8, color=np.array(PANNUKE_COLORS[t]) / 255, label=f\"{NUCLEI_TYPES[t]} ({m.sum()})\")\n"
            'ax.legend(markerscale=3); ax.set_title("세포 임베딩 UMAP (색 = CellViT 예측 타입)"); ax.set_xticks([]); ax.set_yticks([])\n'
            "plt.tight_layout(); plt.show()"
        ),
        # 8. export
        md(
            "## 8. 결과 내보내기: QuPath용 GeoJSON\n"
            "\n"
            "세포 윤곽을 **슬라이드 좌표**(패치 오프셋 더함)로 바꿔 GeoJSON으로 저장하면 QuPath에서 "
            "`File → Import objects`로 원본 WSI 위에 겹쳐 볼 수 있습니다. "
            "(WSI 전체에 대해서는 `cellvit-inference --geojson`이 같은 일을 해 줍니다 → 02 노트북)"
        ),
        code(
            "features = []\n"
            "for c in cells:\n"
            '    poly = (np.asarray(c["contour"]) / scale + [X0, Y0]).tolist()  # 40x 패치 좌표 → 슬라이드 level-0 좌표\n'
            "    poly.append(poly[0])  # 폴리곤 닫기\n"
            '    name = NUCLEI_TYPES[c["type"]]\n'
            "    features.append({\n"
            '        "type": "Feature",\n'
            '        "geometry": {"type": "Polygon", "coordinates": [poly]},\n'
            '        "properties": {"objectType": "detection",\n'
            '                       "classification": {"name": name, "color": list(PANNUKE_COLORS[c["type"]])}},\n'
            "    })\n"
            "\n"
            'out_path = WORKDIR / "outputs" / "patch_cells.geojson"\n'
            "out_path.parent.mkdir(exist_ok=True)\n"
            'out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))\n'
            'print(f"{len(features)}개 세포 → {out_path}")'
        ),
        md(
            "## 정리\n"
            "\n"
            "| Task | 사용한 출력 | 핵심 코드 |\n"
            "|---|---|---|\n"
            '| 조직 분류 | `out["tissue_types"]` | softmax |\n'
            '| 검출 + 분할 | `nuclei_binary_map` + `hv_map` | `model.calculate_instance_map()` |\n'
            '| 세포 분류 | `nuclei_type_map` | 세포 영역 내 다수결 |\n'
            '| 세포 임베딩 | `out["tokens"]` | bbox 내 토큰 평균 |\n'
            "\n"
            "**한계**\n"
            "- 분류 체계가 PanNuke 5종으로 **고정**됨 (림프구/형질세포/대식세포 등 구분 불가)\n"
            "- 새 체계로 바꾸려면 원래는 픽셀 단위 마스크 라벨로 전체 재학습이 필요\n"
            "\n"
            "→ 이 한계를 푸는 것이 **CellViT++** 입니다. `02_CellViT_plusplus.ipynb`로 이어집니다."
        ),
    ]
    return cells


# ─────────────────────────────────────────────────────────────
# 02. CellViT++
# ─────────────────────────────────────────────────────────────
def notebook_cellvitpp():
    cells = [
        md(
            "# CellViT++ 튜토리얼: 분류기 교체와 나만의 세포 분류기 학습\n"
            "\n"
            "**CellViT++** (Hörst et al., arXiv:2501.05269, 2025)의 핵심 아이디어는 간단합니다.\n"
            "\n"
            "```\n"
            "  H&E ─► [CellViT 분할 모델 · 고정] ─► 세포 검출/분할 ───────────────┐\n"
            "                 │                                                  ▼\n"
            "                 └─► 세포별 토큰 임베딩 ─► [가벼운 MLP 분류기] ─► 세포 타입\n"
            "                                            ▲ 이 부분만 교체·학습\n"
            "```\n"
            "\n"
            "- 무거운 분할 모델은 **그대로 두고**, 세포 임베딩 위의 작은 분류기(약 13만 파라미터)만 바꿔 끼웁니다.\n"
            "- 분류기 학습에는 **세포 중심점(point) + 클래스** 라벨만 있으면 됩니다. 윤곽 마스크가 필요 없습니다.\n"
            "- 학습은 GPU에서 수십 초~수 분이면 끝납니다.\n"
            "\n"
            "| Part | 내용 |\n"
            "|---|---|\n"
            "| 1 | 사전학습된 분류기 7종 살펴보기 (Lizard, NuCLS, Ocelot, MIDOG, ...) |\n"
            "| 2 | 같은 세포에 분류기를 바꿔 끼워 보기 (Python) |\n"
            "| 3 | WSI 전체 추론 (`cellvit-inference` CLI) → 세포 JSON/GeoJSON/임베딩 그래프 |\n"
            "| 4 | **점 라벨로 나만의 분류기 학습** (NuCLS 예제) → WSI에 적용 |\n"
            "\n"
            "> CellViT 자체의 task(검출·분할·분류·임베딩)는 `01_CellViT_tasks.ipynb`를 먼저 보세요."
        ),
        *setup_cells(),
        code(
            DOWNLOAD
            + "\n\n"
            "# 사전학습 분류기 7종 (Zenodo classifier.zip)\n"
            "from cellvit.utils.cache_models import cache_classifier\n"
            "CLASSIFIER_DIR = cache_classifier() / (\"sam-h\" if MODEL_NAME == \"SAM\" else \"vit256\")\n"
            "\n"
            "# Part 4용 NuCLS 점 라벨 예제 (CellViT++ GitHub repo에서 해당 폴더만 받기)\n"
            'NUCLS_DIR = WORKDIR / "cellvitpp_repo/test_database/training_database/Example-Detection"\n'
            "if not NUCLS_DIR.exists():\n"
            "    !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/TIO-IKIM/CellViT-plus-plus.git cellvitpp_repo\n"
            "    !cd cellvitpp_repo && git sparse-checkout set test_database/training_database/Example-Detection\n"
            "!ls {CLASSIFIER_DIR} {NUCLS_DIR}"
        ),
        code(HELPERS),
        code(
            PANNUKE_COLORS
            + "\n\n"
            "def palette(n):\n"
            '    """분류기마다 클래스 수가 달라서 tab10 기반 색을 자동 생성"""\n'
            '    cmap = plt.get_cmap("tab10")\n'
            "    return {i: tuple(int(255 * v) for v in cmap(i % 10)[:3]) for i in range(n)}"
        ),
        # Part 1
        md(
            "## Part 1. 사전학습된 분류기 살펴보기\n"
            "\n"
            "각 분류기는 `LinearClassifier` (임베딩 → hidden → 클래스 수)이며, "
            "체크포인트 안에 클래스 이름(`label_map`)이 들어 있습니다."
        ),
        code(
            "from cellvit.models.classifier.linear_classifier import LinearClassifier\n"
            "\n"
            "def load_classifier(path):\n"
            '    """CellViT++ 분류기 체크포인트 → (model, {id: 클래스명})"""\n'
            "    ckpt = torch.load(path, map_location=\"cpu\", weights_only=False)\n"
            '    rc = unflatten_dict(ckpt["config"], ".")\n'
            "    clf = LinearClassifier(\n"
            '        embed_dim=ckpt["model_state_dict"]["fc1.weight"].shape[1],\n'
            '        hidden_dim=rc["model"].get("hidden_dim", 100),\n'
            '        num_classes=rc["data"]["num_classes"],\n'
            "    )\n"
            '    clf.load_state_dict(ckpt["model_state_dict"])\n'
            '    return clf.eval(), {int(k): v for k, v in rc["data"]["label_map"].items()}\n'
            "\n"
            "CLASSIFIERS = {}\n"
            'for p in sorted(CLASSIFIER_DIR.glob("*.pth")):\n'
            "    clf, labels = load_classifier(p)\n"
            "    CLASSIFIERS[p.stem] = (clf, labels)\n"
            "    n_par = sum(x.numel() for x in clf.parameters())\n"
            '    print(f"{p.stem:12s} | {len(labels):2d} classes | {n_par/1e3:6.1f}K params | {list(labels.values())}")'
        ),
        # Part 2
        md(
            "## Part 2. 같은 세포, 다른 분류 체계\n"
            "\n"
            "CellViT로 패치를 **한 번만** 추론해서 세포와 임베딩을 얻은 뒤, 분류기만 바꿔 가며 적용합니다. "
            "분할 결과(윤곽)는 그대로이고 색(클래스)만 바뀌는 것을 볼 수 있습니다."
        ),
        code(
            "model, conf = load_cellvit(CKPT_PATH)\n"
            'PANNUKE_TYPES = {v: k for k, v in conf["dataset_config"]["nuclei_types"].items()}\n'
            "\n"
            "# 01 노트북과 같은 패치: 피부 슬라이드(20x)의 표피-진피 경계 → 40x로 업샘플\n"
            'WSI_PATH = WORKDIR / "test_database/x20_svs/CMU-1-Small-Region.svs"\n'
            "slide = openslide.OpenSlide(str(WSI_PATH))\n"
            "img, scale = read_patch_40x(slide, 768, 768, 1024)\n"
            "thumb = np.array(slide.get_thumbnail((800, 800)).convert(\"RGB\"))\n"
            "ds = slide.dimensions[0] / thumb.shape[1]\n"
            "\n"
            "out = run_cellvit(model, conf, img)\n"
            "inst_map, cells = postprocess(model, out)\n"
            'emb = cell_embeddings(out["tokens"][0], cells)\n'
            'print(f"세포 {len(cells)}개, 임베딩 {tuple(emb.shape)}")'
        ),
        code(
            "@torch.no_grad()\n"
            "def classify(clf, emb):\n"
            "    p = F.softmax(clf(emb), dim=1)\n"
            "    return p.argmax(1).numpy(), p.max(1).values.numpy()\n"
            "\n"
            'SHOW = ["lizard", "nucls_main", "ocelot"]\n'
            "zoom = (slice(256, 768), slice(256, 768))\n"
            "\n"
            "fig, ax = plt.subplots(1, len(SHOW) + 1, figsize=(6 * (len(SHOW) + 1), 6.5))\n"
            "ax[0].imshow(draw_cells(img, cells, PANNUKE_COLORS)[zoom])\n"
            'ax[0].set_title("PanNuke (CellViT 기본 NT 디코더)")\n'
            "legend(ax[0], {k: v for k, v in PANNUKE_TYPES.items() if k}, PANNUKE_COLORS)\n"
            "for a, name in zip(ax[1:], SHOW):\n"
            "    clf, labels = CLASSIFIERS[name]\n"
            "    pred, _ = classify(clf, emb)\n"
            '    relabeled = [dict(c, type=int(t)) for c, t in zip(cells, pred)]\n'
            "    cols = palette(len(labels))\n"
            "    a.imshow(draw_cells(img, relabeled, cols)[zoom])\n"
            '    a.set_title(f"{name} 분류기")\n'
            "    legend(a, labels, cols)\n"
            '[a.axis("off") for a in ax]\n'
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "Lizard는 림프구·형질세포·호중구·호산구까지 나누고, Ocelot은 종양/비종양 2분류만 합니다. "
            "**같은 임베딩**에서 목적에 맞는 체계를 고르면 되고, 원하는 체계가 없으면 Part 4처럼 직접 학습합니다."
        ),
        # Part 3
        md(
            "## Part 3. WSI 전체 추론 (CLI)\n"
            "\n"
            "`cellvit-inference`는 슬라이드를 1024px 패치로 나눠 추론하고, 패치 경계에서 중복된 세포를 정리해 "
            "슬라이드 단위 결과를 저장합니다.\n"
            "\n"
            "| 옵션 | 의미 |\n"
            "|---|---|\n"
            "| `--model SAM \\| HIPT` | 분할 모델 |\n"
            "| `--nuclei_taxonomy` | `pannuke`(기본), `binary`, `lizard`, `consep`, `midog`, `nucls_main`, `nucls_super`, `ocelot`, `panoptils` |\n"
            "| `--geojson` | QuPath용 GeoJSON 추가 저장 |\n"
            "| `--graph` | 세포 임베딩 + 좌표를 `cells.pt`로 저장 (Part 4에서 사용) |\n"
            "| `--cpu_count / --ray_worker / --ray_remote_cpus` | Colab 무료(2코어)에서는 작게 설정 |\n"
            "| `process_wsi --wsi_mpp --wsi_magnification` | 메타데이터 없는 TIFF는 해상도를 직접 지정 |\n"
            "\n"
            "예제는 01 노트북의 **피부 슬라이드(20x)** 전체입니다. mpp 메타데이터가 있어서 파이프라인이 자동으로 "
            "0.25 µm/px로 리샘플링하므로 `--wsi_mpp`를 줄 필요가 없습니다. Colab T4에서 수 분 걸립니다."
        ),
        code(
            "# CLI는 별도 프로세스에서 모델을 다시 올리므로, 노트북이 잡고 있는 GPU 메모리를 먼저 비웁니다.\n"
            "# (안 비우면 Colab T4에서 CUDA out of memory)\n"
            "import gc\n"
            "model = None; gc.collect(); torch.cuda.empty_cache()\n"
            "\n"
            'OUTDIR = WORKDIR / "outputs/wsi_lizard"\n'
            "OUTDIR.parent.mkdir(exist_ok=True)\n"
            "CPU = 2 if IN_COLAB else min(8, os.cpu_count())\n"
            "\n"
            "!cellvit-inference \\\n"
            "    --model {MODEL_NAME} \\\n"
            "    --nuclei_taxonomy lizard \\\n"
            "    --outdir {OUTDIR} \\\n"
            "    --geojson --graph \\\n"
            "    --cpu_count {CPU} --ray_worker 1 --ray_remote_cpus 1 \\\n"
            "    process_wsi \\\n"
            "    --wsi_path {WSI_PATH} > {OUTDIR}.log 2>&1\n"
            "\n"
            "!tail -n 4 {OUTDIR}.log   # 전체 로그: outputs/wsi_lizard.log\n"
            "!ls -lh {OUTDIR}/*"
        ),
        md(
            "### 출력 파일 구조\n"
            "\n"
            "| 파일 | 내용 |\n"
            "|---|---|\n"
            "| `cells.json` | 세포별 `contour`, `bbox`, `centroid`, `type`, `type_prob` + `type_map` |\n"
            "| `cell_detection.json` | 중심 좌표와 타입만 (가벼움) |\n"
            "| `cells.geojson` / `cell_detection.geojson` | QuPath에서 `File → Import objects` |\n"
            "| `cells.pt` | `x`: 세포 임베딩 (N×D), `positions`: 좌표, `metadata` |"
        ),
        code(
            'slide_dir = next(p for p in OUTDIR.iterdir() if p.is_dir())\n'
            'cells_json = json.loads((slide_dir / "cells.json").read_text())\n'
            'wsi_cells = cells_json["cells"]\n'
            'LIZARD = {int(k): v for k, v in cells_json["type_map"].items()}\n'
            'graph = torch.load(slide_dir / "cells.pt", weights_only=False)\n'
            "\n"
            'print("type_map:", LIZARD)\n'
            'print(f"세포 수: {len(wsi_cells)} | graph.x: {tuple(graph.x.shape)} | graph.positions: {tuple(graph.positions.shape)}")\n'
            "\n"
            'wsi_types = np.array([c["type"] for c in wsi_cells])\n'
            'wsi_xy = np.array([c["centroid"] for c in wsi_cells])\n'
            "cols = palette(len(LIZARD))\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={\"width_ratios\": [2, 1]})\n"
            "ax[0].imshow(thumb)\n"
            "for t, name in LIZARD.items():\n"
            "    m = wsi_types == t\n"
            "    if m.any():\n"
            "        ax[0].scatter(wsi_xy[m, 0] / ds, wsi_xy[m, 1] / ds, s=3, color=np.array(cols[t]) / 255, label=name)\n"
            'ax[0].legend(markerscale=4, loc="upper left", bbox_to_anchor=(1.01, 1)); ax[0].axis("off")\n'
            'ax[0].set_title("WSI 전체 세포 (Lizard 분류)")\n'
            "names, counts = zip(*[(LIZARD[t], (wsi_types == t).sum()) for t in LIZARD])\n"
            "ax[1].barh(names, counts, color=[np.array(cols[t]) / 255 for t in LIZARD], ec=\"k\")\n"
            'ax[1].set_title("타입별 세포 수"); ax[1].invert_yaxis()\n'
            "plt.tight_layout(); plt.show()"
        ),
        # Part 4
        md(
            "## Part 4. 점 라벨로 나만의 세포 분류기 학습\n"
            "\n"
            "**시나리오**: 병리 전문의가 세포 중심에 점을 찍고 클래스를 달아 준 데이터만 있을 때, "
            "우리 연구에 맞는 분류 체계로 CellViT++를 맞춘다.\n"
            "\n"
            "예제 데이터는 CellViT++ repo에 포함된 **NuCLS** (유방암, CC0) 일부입니다.\n"
            "- train 10장 / test 5장, 256×256 px (40x)\n"
            "- 라벨 CSV: `x, y, class` (점 라벨)\n"
            "- 클래스 4종: Tumor / nonTIL Stromal / sTIL (기질 내 종양침윤림프구) / Other\n"
            "\n"
            "**학습 과정**\n"
            "1. 각 이미지를 CellViT에 넣어 세포 검출 + 임베딩 추출\n"
            "2. 정답 점을 가장 가까운 검출 세포와 1:1 매칭 (헝가리안, 거리 ≤ 12px)\n"
            "3. 매칭된 (임베딩, 클래스) 쌍으로 `LinearClassifier` 학습"
        ),
        code(
            "import pandas as pd\n"
            "from PIL import Image\n"
            "from scipy.optimize import linear_sum_assignment\n"
            "\n"
            'NUCLS_LABELS = {0: "Tumor", 1: "nonTIL Stromal", 2: "sTIL", 3: "Other"}\n'
            "MATCH_RADIUS = 12  # px (40x에서 약 3 µm)\n"
            "\n"
            "def load_split(split):\n"
            "    items = []\n"
            '    for img_path in sorted((NUCLS_DIR / split / "images").glob("*.png")):\n'
            '        lbl = pd.read_csv(NUCLS_DIR / split / "labels" / f"{img_path.stem}.csv", header=None, names=["x", "y", "label"])\n'
            '        items.append((img_path.stem, np.array(Image.open(img_path).convert("RGB")), lbl))\n'
            "    return items\n"
            "\n"
            "def extract(items):\n"
            '    """이미지별 CellViT 추론 → 정답 점과 검출 세포 매칭 → (임베딩, 라벨)"""\n'
            "    X, y, n_gt = [], [], 0\n"
            "    for name, im, lbl in items:\n"
            "        o = run_cellvit(model, conf, im)\n"
            "        _, det = postprocess(model, o)\n"
            "        n_gt += len(lbl)\n"
            "        if not det:\n"
            "            continue\n"
            '        e = cell_embeddings(o["tokens"][0], det)\n'
            '        dc = np.array([c["centroid"] for c in det])              # (x, y)\n'
            '        d = np.linalg.norm(lbl[["x", "y"]].values[:, None] - dc[None], axis=-1)\n'
            "        gi, di = linear_sum_assignment(d)\n"
            "        ok = d[gi, di] <= MATCH_RADIUS\n"
            "        X.append(e[di[ok]]); y.append(lbl[\"label\"].values[gi[ok]])\n"
            "    X, y = torch.cat(X), torch.from_numpy(np.concatenate(y)).long()\n"
            '    print(f"  정답 점 {n_gt}개 중 {len(y)}개 매칭 ({len(y)/n_gt:.0%})")\n'
            "    return X, y\n"
            "\n"
            "model, conf = load_cellvit(CKPT_PATH)  # Part 3에서 비웠던 모델 다시 로드\n"
            'train_items, test_items = load_split("train"), load_split("test")\n'
            'print("train"); X_tr, y_tr = extract(train_items)\n'
            'print("test");  X_te, y_te = extract(test_items)\n'
            'print("train 클래스 분포:", {NUCLS_LABELS[k]: int((y_tr == k).sum()) for k in NUCLS_LABELS})'
        ),
        code(
            "# 학습 데이터 예시: 점 라벨\n"
            "cols4 = palette(4)\n"
            "fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))\n"
            "for a, (name, im, lbl) in zip(ax, train_items[:4]):\n"
            "    a.imshow(im)\n"
            "    for k, n in NUCLS_LABELS.items():\n"
            '        m = lbl["label"] == k\n'
            '        a.scatter(lbl.x[m], lbl.y[m], s=40, color=np.array(cols4[k]) / 255, ec="w", label=n)\n'
            '    a.set_title(name); a.axis("off")\n'
            'ax[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))\n'
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 분류기 학습\n"
            "\n"
            "CellViT++ 공식 설정(`hidden_dim=256`, AdamW, dropout 0.1)과 비슷하게 맞추고, "
            "클래스 불균형은 가중치로 보정합니다. 세포 수가 수백 개라 몇 초면 끝납니다."
        ),
        code(
            "from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score\n"
            "\n"
            "def train_classifier(X, y, n_cls=4, hidden=256, epochs=300, lr=3e-4, seed=0):\n"
            "    torch.manual_seed(seed)\n"
            "    clf = LinearClassifier(embed_dim=X.shape[1], hidden_dim=hidden, num_classes=n_cls, drop_rate=0.1)\n"
            "    w = 1.0 / torch.bincount(y, minlength=n_cls).clamp(min=1).float()\n"
            "    loss_fn = torch.nn.CrossEntropyLoss(weight=w / w.sum() * n_cls)\n"
            "    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)\n"
            "    losses = []\n"
            "    for _ in range(epochs):  # 데이터가 작아서 full-batch\n"
            "        clf.train(); opt.zero_grad()\n"
            "        loss = loss_fn(clf(X), y); loss.backward(); opt.step()\n"
            "        losses.append(loss.item())\n"
            "    return clf.eval(), losses\n"
            "\n"
            "t = time.time()\n"
            "my_clf, losses = train_classifier(X_tr, y_tr)\n"
            'print(f"학습 시간: {time.time() - t:.1f}s")\n'
            "\n"
            "pred, _ = classify(my_clf, X_te)\n"
            "print(classification_report(y_te, pred, labels=list(NUCLS_LABELS), target_names=list(NUCLS_LABELS.values()), zero_division=0))\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
            'ax[0].plot(losses); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].set_title("학습 곡선")\n'
            "ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=list(NUCLS_LABELS)),\n"
            '                       display_labels=list(NUCLS_LABELS.values())).plot(ax=ax[1], cmap="Blues", colorbar=False)\n'
            'ax[1].set_title("Test confusion matrix"); ax[1].tick_params(axis="x", rotation=30)\n'
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 라벨이 얼마나 필요할까?\n"
            "\n"
            "학습 세포 수를 줄여 가며 test macro-F1을 봅니다(각 크기별 5회 반복). "
            "CellViT++ 논문의 주장대로 **적은 수의 점 라벨로도** 성능이 빠르게 올라가는지 확인합니다.\n"
            "\n"
            "> 예제 데이터는 train 약 400개, test 약 190개 세포뿐이라 수치 자체는 변동이 큽니다. 경향만 보세요."
        ),
        code(
            "rng = np.random.default_rng(0)\n"
            "sizes = [25, 50, 100, 200, len(y_tr)]\n"
            "res = []\n"
            "for n in sizes:\n"
            "    for r in range(5):\n"
            "        idx = rng.choice(len(y_tr), n, replace=False)\n"
            "        c, _ = train_classifier(X_tr[idx], y_tr[idx], seed=r)\n"
            "        res.append((n, f1_score(y_te, classify(c, X_te)[0], average=\"macro\")))\n"
            "res = pd.DataFrame(res, columns=[\"n_train\", \"macro_f1\"])\n"
            'summary = res.groupby("n_train").macro_f1.agg(["mean", "std"])\n'
            "print(summary.round(3))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            'ax.errorbar(summary.index, summary["mean"], yerr=summary["std"], marker="o", capsize=4)\n'
            'ax.set_xscale("log"); ax.set_xlabel("학습에 쓴 세포 수 (점 라벨)"); ax.set_ylabel("test macro-F1")\n'
            'ax.set_title("라벨 수에 따른 성능"); ax.grid(alpha=0.3)\n'
            "plt.tight_layout(); plt.show()"
        ),
        md("### test 이미지에서 예측 확인"),
        code(
            "fig, ax = plt.subplots(2, len(test_items), figsize=(4.2 * len(test_items), 8.5))\n"
            "for j, (name, im, lbl) in enumerate(test_items):\n"
            "    o = run_cellvit(model, conf, im)\n"
            "    _, det = postprocess(model, o)\n"
            "    p, _ = classify(my_clf, cell_embeddings(o[\"tokens\"][0], det))\n"
            "    ax[0, j].imshow(im)\n"
            "    for k in NUCLS_LABELS:\n"
            '        m = lbl["label"] == k\n'
            '        ax[0, j].scatter(lbl.x[m], lbl.y[m], s=40, color=np.array(cols4[k]) / 255, ec="w")\n'
            "    ax[1, j].imshow(draw_cells(im, [dict(c, type=int(t)) for c, t in zip(det, p)], cols4, thickness=1))\n"
            '    ax[0, j].set_title(f"{name}: 정답 (점)"); ax[1, j].set_title("예측 (윤곽)")\n'
            '[a.axis("off") for a in ax.ravel()]\n'
            "legend(ax[0, -1], NUCLS_LABELS, cols4)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 학습한 분류기를 저장하고 WSI 전체에 적용\n"
            "\n"
            "CellViT++ 체크포인트와 같은 형식(`model_state_dict` + 평탄화된 `config`)으로 저장합니다. "
            "이 파일은 CellViT++ repo의 `detect_cells.py --classifier_path`에 그대로 넣을 수 있습니다.\n"
            "\n"
            "여기서는 Part 3에서 저장한 **WSI 전체 세포 임베딩(`cells.pt`)** 에 바로 적용합니다. "
            "분할 모델을 다시 돌릴 필요가 없다는 점이 CellViT++ 방식의 장점입니다.\n"
            "\n"
            "> ⚠️ 이 분류기는 **유방암**(NuCLS) 세포로 학습했는데 예제 WSI는 **정상 피부**입니다. "
            "결과는 '분류기를 WSI에 붙이는 방법'의 시연일 뿐이며, 실제로는 학습 데이터와 같은 도메인의 슬라이드에 써야 합니다."
        ),
        code(
            "ckpt = {\n"
            '    "model_state_dict": my_clf.state_dict(),\n'
            '    "config": {"data.num_classes": 4, "data.label_map": NUCLS_LABELS, "model.hidden_dim": 256},\n'
            "}\n"
            'my_path = WORKDIR / "outputs" / "my_nucls_classifier.pth"\n'
            "torch.save(ckpt, my_path)\n"
            "\n"
            "clf_reloaded, labels = load_classifier(my_path)  # Part 1과 같은 로더로 다시 읽힘\n"
            "wsi_pred, wsi_conf = classify(clf_reloaded, graph.x.float())\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={\"width_ratios\": [2, 1]})\n"
            "ax[0].imshow(thumb)\n"
            "pos = graph.positions.numpy()\n"
            "for k, n in labels.items():\n"
            "    m = wsi_pred == k\n"
            "    ax[0].scatter(pos[m, 0] / ds, pos[m, 1] / ds, s=3, color=np.array(cols4[k]) / 255, label=n)\n"
            'ax[0].legend(markerscale=4, loc="upper left", bbox_to_anchor=(1.01, 1)); ax[0].axis("off")\n'
            'ax[0].set_title("WSI 전체 세포 - 직접 학습한 NuCLS 분류기")\n'
            "ax[1].barh(list(labels.values()), [(wsi_pred == k).sum() for k in labels],\n"
            '           color=[np.array(cols4[k]) / 255 for k in labels], ec="k")\n'
            'ax[1].invert_yaxis(); ax[1].set_title("타입별 세포 수")\n'
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "## 정리\n"
            "\n"
            "| | CellViT | CellViT++ |\n"
            "|---|---|---|\n"
            "| 분류 체계 | PanNuke 5종 고정 | 분류기 교체로 자유롭게 |\n"
            "| 새 체계 학습 | 마스크 라벨 + 전체 재학습 | **점 라벨 + MLP만 학습 (수 초~수 분)** |\n"
            "| 재추론 | 필요 | 저장된 임베딩(`cells.pt`)에 분류기만 적용 |\n"
            "\n"
            "**실전에서 더 해 볼 것**\n"
            "- 실제 프로젝트 데이터는 수천 개 이상의 점 라벨과 교차검증으로 학습하세요. "
            "공식 학습 스크립트: CellViT++ repo의 `cellvit/train_cell_classifier_head.py`\n"
            "- 점 라벨링은 CellViT++ repo의 웹 기반 annotation tool이나 QuPath로 할 수 있습니다.\n"
            "- `cells.pt`의 임베딩과 좌표로 세포 그래프(GNN), 이웃 조성 분석, TIL 밀도 같은 공간 지표를 만들 수 있습니다.\n"
            "- mpp 메타데이터가 없는 TIFF는 `process_wsi --wsi_mpp 0.25 --wsi_magnification 40`처럼 해상도를 직접 지정합니다.\n"
            "- 라이선스: Apache 2.0 + Commons Clause → 상업적 이용 시 저자 허가 필요"
        ),
    ]
    return cells


REPO = "fourmodern/cellvit-tutorial"


def colab_badge(path):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/{path}"
    return md(f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = [colab_badge(path)] + cells
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    build(notebook_cellvit(), "01_CellViT_tasks.ipynb")
    build(notebook_cellvitpp(), "02_CellViT_plusplus.ipynb")
