"""HNE2Cell 튜토리얼 노트북 생성 스크립트.

python build_hne2cell.py  →  03_HNE2Cell.ipynb
"""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

REPO = "fourmodern/cellvit-tutorial"
NB_PATH = "03_HNE2Cell.ipynb"


def cells():
    return [
        md(
            f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/{NB_PATH}" target="_parent">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
        ),
        md(
            "# HNE2Cell 튜토리얼: H&E에서 15종 세포를 전사체 수준으로 분류하기\n"
            "\n"
            "**HNE2Cell**은 H&E 전체 슬라이드 이미지(WSI)에서 세포핵을 검출·분할하고 **15종 세포 타입**으로 분류하는 모델입니다. "
            "모델, 추론 코드, 예제 슬라이드가 HuggingFace([`roobee79/HNE2Cell`](https://huggingface.co/roobee79/HNE2Cell))에 공개되어 있습니다. "
            "공개 README의 인용 제목(*Spatial transcriptomics–supervised deep learning enables single-cell mapping of tumor immune "
            "architecture from routine histology*)대로, **공간전사체에서 정한 세포 타입을 정답으로 학습**한 것이 특징입니다.\n"
            "\n"
            "```\n"
            "  공간전사체: 유전자 발현 → 세포 타입 라벨\n"
            "     ▼ (같은 조직의 H&E에 정렬)\n"
            "  H&E 256px 패치 → 224 축소 → 대형 ViT encoder (약 13억 파라미터) ─► CellViT식 디코더\n"
            "                                                            ├─ 핵 분할 (핵 확률 + HV 맵)\n"
            "                                                            └─ 픽셀별 세포 타입 (배경 + 15종)\n"
            "```\n"
            "\n"
            "| | CellViT | CellViT++ | **HNE2Cell** |\n"
            "|---|---|---|---|\n"
            "| 정답 라벨 출처 | 병리의 수작업(PanNuke) | 수작업 점 라벨 | **공간전사체** |\n"
            "| Encoder | SAM-H / HIPT | SAM-H / HIPT (고정) | 대형 병리 ViT (약 1.3B 파라미터, 모델 전체 기준) |\n"
            "| 세포 타입 | 5종 | 데이터셋별 2~7종 | **15종** (CD4/CD8 T, B, Plasma, Macrophage, DC, Fibroblast, Endothelial, Pericyte, ...) |\n"
            "| 입력 | 1024px @ 40x | 1024px @ 40x | 256px @ 40x → 224로 축소 |\n"
            "\n"
            "> 이 노트북은 HuggingFace에 공개된 모델·코드·README만 근거로 작성했습니다. 모델의 학습 데이터와 성능 평가는 공개 README의 인용 정보를 참고하세요.\n"
            "\n"
            "### 이 노트북에서 하는 것\n"
            "\n"
            "| § | 내용 |\n"
            "|---|---|\n"
            "| 1 | 모델·예제 슬라이드(TCGA 폐 편평세포암) 다운로드, 모델 출력 구조 확인 |\n"
            "| 2 | Reinhard 색 정규화 (Colab 메모리에 맞춘 패치 단위 버전) |\n"
            "| 3 | 패치 하나 추론: 원시 출력 → 세포 분할 → 15종 분류 → 5개 계통 그룹 |\n"
            "| 4 | 슬라이드 영역(또는 전체) 추론 + 겹친 패치 중복 제거 |\n"
            "| 5 | 세포 지도, 조성, 공간 분석 (면역세포 침윤, B/T 세포 군집 = TLS 후보) |\n"
            "| 6 | CSV / QuPath GeoJSON 내보내기 |\n"
            "\n"
            "**라이선스**: 코드 MIT, **모델 가중치 CC BY-NC 4.0 (비상업적 연구용)**. "
            "모델·코드: https://huggingface.co/roobee79/HNE2Cell"
        ),
        # ── 0. setup
        md(
            "## 0. 환경 설정\n"
            "\n"
            "**Colab**: `런타임 → 런타임 유형 변경 → T4 GPU`. "
            "Colab 기본 패키지(torch, opencv, scikit-image, numba)를 그대로 쓰므로 **런타임 재시작이 필요 없습니다.**\n"
            "\n"
            "필요 자원: GPU 메모리 약 6GB, 디스크 약 6GB (모델 5.1GB)."
        ),
        code(
            "import sys, os, importlib.util\n"
            "from pathlib import Path\n"
            "\n"
            'IN_COLAB = "google.colab" in sys.modules\n'
            'if importlib.util.find_spec("openslide") is None:\n'
            "    !pip -q install openslide-bin openslide-python\n"
            "\n"
            'WORKDIR = Path("/content/hne2cell") if IN_COLAB else Path.cwd() / "hne2cell_work"\n'
            "WORKDIR.mkdir(parents=True, exist_ok=True)\n"
            "os.chdir(WORKDIR)\n"
            "\n"
            "# 모델(5.1GB)을 Google Drive에 캐시하면 다음 세션에서 재다운로드하지 않습니다.\n"
            "USE_DRIVE_CACHE = False\n"
            "if IN_COLAB and USE_DRIVE_CACHE:\n"
            "    from google.colab import drive\n"
            '    drive.mount("/content/drive")\n'
            '    HF_CACHE = "/content/drive/MyDrive/hne2cell_cache"\n'
            "else:\n"
            '    HF_CACHE = str(WORKDIR.parent / "hne2cell_cache") if not IN_COLAB else str(WORKDIR / "hf_cache")\n'
            'print("WORKDIR:", WORKDIR, "| HF cache:", HF_CACHE)'
        ),
        # ── 1. download
        md(
            "## 1. 모델과 예제 데이터 받기 (HuggingFace `roobee79/HNE2Cell`)\n"
            "\n"
            "| 파일 | 크기 | 용도 |\n"
            "|---|---|---|\n"
            "| `HNE2cell_pub_patch73_jit.pt` | 5.1 GB | TorchScript 모델 (ViT encoder + CellViT식 디코더) |\n"
            "| `TCGA-56-8628-...svs` | 36 MB | 예제 WSI: TCGA 폐 편평세포암(LUSC), 40x |\n"
            "| `standard-ilc.tif` | 355 MB | 색 정규화 기준 이미지 |\n"
            "| `post_processing.py`, `tools.py` | - | 공식 후처리 (HoVer-Net/CellViT식 watershed) |"
        ),
        code(
            "from huggingface_hub import hf_hub_download\n"
            "\n"
            'REPO_ID = "roobee79/HNE2Cell"\n'
            'SLIDE_NAME = "TCGA-56-8628-01Z-00-DX1.AAC57164-E0F9-4DF0-87EA-5C50FB201895.svs"\n'
            "get = lambda f: hf_hub_download(REPO_ID, f, cache_dir=HF_CACHE)\n"
            "\n"
            'for f in ["post_processing.py", "tools.py", "config.json"]:\n'
            "    get(f)\n"
            'CODE_DIR = Path(get("tools.py")).parent\n'
            "sys.path.insert(0, str(CODE_DIR))  # 공식 post_processing 모듈 import용\n"
            "\n"
            "SLIDE_PATH = get(SLIDE_NAME)\n"
            'REF_PATH = get("standard-ilc.tif")\n'
            'MODEL_PATH = get("HNE2cell_pub_patch73_jit.pt")  # 5.1GB, 수 분 소요\n'
            "!ls -lhL {CODE_DIR}"
        ),
        code(
            "import json, time, warnings\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "import cv2\n"
            "import matplotlib.pyplot as plt\n"
            "import openslide, tifffile\n"
            "from skimage import color\n"
            "from post_processing import DetectionCellPostProcessor  # HuggingFace 공식 코드\n"
            "\n"
            'warnings.filterwarnings("ignore")\n'
            'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
            "\n"
            "# 한글 폰트\n"
            "import matplotlib.font_manager as fm, urllib.request\n"
            '_font = WORKDIR / "NanumGothic-Regular.ttf"\n'
            "if not _font.exists():\n"
            '    urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf", _font)\n'
            "fm.fontManager.addfont(str(_font))\n"
            'plt.rcParams["font.family"] = fm.FontProperties(fname=str(_font)).get_name()\n'
            'plt.rcParams["axes.unicode_minus"] = False\n'
            "\n"
            'CONFIG = json.loads((CODE_DIR / "config.json").read_text())\n'
            'CELL_TYPES = {int(k): v for k, v in CONFIG["id2label"].items()}  # 0=Background, 1~15\n'
            'print("device:", DEVICE, "| torch", torch.__version__)\n'
            "print(CELL_TYPES)"
        ),
        md(
            "### 세포 타입과 계통 그룹\n"
            "\n"
            "모델은 15종을 출력합니다. 공식 README의 색 체계(종양=빨강, 면역=파랑 계열, 기질=초록 계열, 상피=주황, 사멸=회색)에 따라 "
            "5개 계통 그룹으로 묶어 보면 큰 그림을 보기 쉽습니다. 이 노트북에서는 그룹을 `tier1`, 15종을 `cell_type`이라고 부릅니다. "
            "색도 공식 `inference.py`의 값을 그대로 씁니다."
        ),
        code(
            "TIER1 = {\n"
            '    "Malignant": ["Malignant"],\n'
            '    "Immune": ["CD4 T", "CD8 T", "B", "Plasma", "Macrophage", "Myeloid", "DC", "Immune_Other"],\n'
            '    "Stromal": ["Fibroblast", "Endothelial", "Pericyte", "Stromal_Other"],\n'
            '    "Epithelial": ["Epithelial"],\n'
            '    "Dead": ["Dead"],\n'
            "}\n"
            "TIER2_TO_TIER1 = {t2: t1 for t1, t2s in TIER1.items() for t2 in t2s}\n"
            "\n"
            "# 공식 inference.py의 CELL_COLORS (RGB)\n"
            "COLORS = {1: (255, 0, 0), 2: (30, 144, 255), 3: (65, 105, 225), 4: (0, 0, 255), 5: (100, 149, 237),\n"
            "          6: (176, 224, 230), 7: (70, 130, 180), 8: (0, 191, 255), 9: (34, 139, 34), 10: (60, 179, 113),\n"
            "          11: (50, 205, 50), 12: (255, 140, 0), 13: (135, 206, 250), 14: (107, 142, 35), 15: (128, 128, 128)}\n"
            "# Immune_Other(13)는 공식 색이 Macrophage(6)와 같아서 구분되도록 살짝 바꿈\n"
            'TIER1_COLORS = {"Malignant": (220, 20, 20), "Immune": (30, 90, 255), "Stromal": (34, 139, 34),\n'
            '                "Epithelial": (255, 140, 0), "Dead": (128, 128, 128)}\n'
            "\n"
            "def legend(ax, items, **kw):\n"
            "    from matplotlib.patches import Patch\n"
            "    ax.legend(handles=[Patch(color=np.array(c) / 255, label=n) for n, c in items],\n"
            '              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, **kw)\n'
            "\n"
            "for t1, t2s in TIER1.items():\n"
            '    print(f"{t1:11s} ← {t2s}")'
        ),
        md(
            "### 모델 불러오기와 출력 구조\n"
            "\n"
            "TorchScript 파일이라 모델 클래스 정의 없이 `torch.jit.load`로 바로 씁니다. "
            "입력은 256×256 패치를 **224×224로 축소**하고 H&E 데이터로 맞춘 평균/표준편차로 정규화합니다(공식 `inference.py`와 동일)."
        ),
        code(
            "t = time.time()\n"
            "model = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval()\n"
            'print(f"로딩 {time.time() - t:.0f}s")\n'
            "\n"
            "from PIL import Image\n"
            "from torchvision import transforms\n"
            "\n"
            "# 공식 inference.py의 TRANSFORM 그대로 (PIL 기반 resize까지 같아야 결과가 일치)\n"
            "TRANSFORM = transforms.Compose([\n"
            "    transforms.Resize((224, 224)),\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize(mean=[0.707223, 0.578729, 0.703617], std=[0.211883, 0.230117, 0.177517]),\n"
            "])\n"
            "\n"
            "@torch.inference_mode()\n"
            "def run_model(patches):\n"
            '    """patches: (B, 256, 256, 3) uint8 RGB → 출력 dict (CPU, float32)"""\n'
            "    x = torch.stack([TRANSFORM(Image.fromarray(p)) for p in patches]).to(DEVICE)\n"
            '    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):\n'
            "        out = model(x)\n"
            "    return {k: v.float().cpu() for k, v in out.items()}\n"
            "\n"
            "dummy = run_model(np.full((1, 256, 256, 3), 255, np.uint8))\n"
            "for k, v in dummy.items():\n"
            '    print(f"{k:20s} {tuple(v.shape)}")'
        ),
        # ── 2. normalization
        md(
            "## 2. Reinhard 색 정규화\n"
            "\n"
            "HNE2Cell은 학습할 때 모든 슬라이드를 기준 이미지(`standard-ilc.tif`)의 색 분포에 맞췄습니다. "
            "추론할 때도 **같은 정규화를 해야** 성능이 나옵니다.\n"
            "\n"
            "Reinhard 방법은 LAB 색공간의 각 채널을 `(x − μ_slide) / σ_slide × σ_ref + μ_ref`로 옮깁니다.\n"
            "\n"
            "> 공식 `normalize.py`는 슬라이드 전체를 원본 해상도로 메모리에 올려 변환합니다 "
            "(이 예제만 해도 15GB 이상 → Colab 무료 RAM 12.7GB 초과). "
            "여기서는 **통계(μ, σ)만 타일 단위로 한 번 구하고 패치마다 변환**하는 방식으로 같은 계산을 메모리 2GB 안에서 합니다.\n"
            "\n"
            "> ✅ **검증**: 이 노트북의 정규화·패치 격자·전처리로 §4 ROI를 추론한 결과를 공식 파이프라인"
            "(`normalize.py` → `patchify.py` → `inference.py`, RAM 35GB 사용)과 세포 단위로 비교했을 때 "
            "**세포 위치 99.5%, 세포 타입 99.7%가 일치**했습니다."
        ),
        code(
            "def lab_stats(rgb_iter):\n"
            '    """RGB 블록들에서 LAB 채널별 평균/표준편차 (0이 아닌 값만, 공식 코드와 동일 기준)"""\n'
            "    s, s2, n = np.zeros(3), np.zeros(3), np.zeros(3)\n"
            "    for rgb in rgb_iter:\n"
            "        lab = color.rgb2lab(rgb)\n"
            "        for c in range(3):\n"
            "            v = lab[..., c][lab[..., c] != 0]\n"
            "            s[c] += v.sum(); s2[c] += (v ** 2).sum(); n[c] += v.size\n"
            "    mean = s / n\n"
            "    return mean, np.sqrt(s2 / n - mean ** 2)\n"
            "\n"
            "def tissue_blocks(img, block=32, sat_thr=0.1):\n"
            '    """공식 코드처럼 채도가 낮은 블록(배경)을 0(검정)으로 지운 이미지"""\n'
            "    hsv_s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1] / 255.0\n"
            "    out = img.copy()\n"
            "    for y in range(0, img.shape[0], block):\n"
            "        for x in range(0, img.shape[1], block):\n"
            "            if hsv_s[y:y + block, x:x + block].mean() < sat_thr:\n"
            "                out[y:y + block, x:x + block] = 0\n"
            "    return out\n"
            "\n"
            "# 기준 이미지 통계 (1024행씩 나눠 계산 → 메모리 절약)\n"
            "ref = tifffile.imread(REF_PATH)\n"
            "REF_MEAN, REF_STD = lab_stats(ref[i:i + 1024] for i in range(0, ref.shape[0], 1024))\n"
            "del ref\n"
            "\n"
            "# 슬라이드 통계: 원본 해상도를 2048px 타일로 나눠 읽으며 조직 블록(128px)만 집계 → 공식과 같은 값, 메모리는 수백 MB\n"
            "slide = openslide.OpenSlide(SLIDE_PATH)\n"
            'MPP = float(slide.properties["openslide.mpp-x"])\n'
            "W, H = slide.dimensions\n"
            "Wc, Hc = W // 128 * 128, H // 128 * 128  # 공식 코드처럼 128 격자에 맞춤\n"
            "def slide_tiles(t=2048):\n"
            "    for y in range(0, Hc, t):\n"
            "        for x in range(0, Wc, t):\n"
            '            im = tissue_blocks(np.array(slide.read_region((x, y), 0, (min(t, Wc - x), min(t, Hc - y))).convert("RGB")), block=128)\n'
            "            if im.any():\n"
            "                yield im\n"
            "t = time.time()\n"
            "SRC_MEAN, SRC_STD = lab_stats(slide_tiles())\n"
            'print(f"슬라이드 통계 계산 {time.time() - t:.0f}s")\n'
            "\n"
            "lvl = 1  # 조직 판별·미리보기용 저해상도 레벨\n"
            "low = np.array(slide.read_region((0, 0), lvl, slide.level_dimensions[lvl]).convert(\"RGB\"))\n"
            "\n"
            'print("슬라이드:", slide.dimensions, f"| {MPP} µm/px | 배율", slide.properties.get("openslide.objective-power"))\n'
            'print("기준 LAB  mean", REF_MEAN.round(2), "std", REF_STD.round(2))\n'
            'print("슬라이드 LAB mean", SRC_MEAN.round(2), "std", SRC_STD.round(2))\n'
            "\n"
            "def reinhard(rgb, block=128):\n"
            '    """uint8 RGB → 기준 색 분포로 정규화한 uint8 RGB.\n'
            "    공식 코드처럼 채도가 낮은 block×block 영역(배경)은 정규화하지 않고 흰색으로 둔다.\"\"\"\n"
            "    lab = color.rgb2lab(rgb)\n"
            "    lab = (lab - SRC_MEAN) * (REF_STD / SRC_STD) + REF_MEAN\n"
            "    out = (np.clip(color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)\n"
            "    bg = tissue_blocks(rgb, block=block).max(-1) == 0\n"
            "    out[bg] = 255\n"
            "    return out"
        ),
        code(
            "thumb = np.array(slide.get_thumbnail((1600, 1600)).convert(\"RGB\"))\n"
            "DS = slide.dimensions[0] / thumb.shape[1]  # 썸네일 1px = level-0 DS px\n"
            "\n"
            "fig, ax = plt.subplots(2, 1, figsize=(16, 9))\n"
            'ax[0].imshow(thumb); ax[0].set_title("원본")\n'
            'ax[1].imshow(reinhard(thumb, block=max(1, int(128 / DS)))); ax[1].set_title("Reinhard 정규화 후 (배경은 흰색)")\n'
            '[a.axis("off") for a in ax]\n'
            "plt.tight_layout(); plt.show()"
        ),
        # ── 3. single patch
        md(
            "## 3. 패치 하나로 보는 HNE2Cell의 출력\n"
            "\n"
            "40x 슬라이드에서 256×256 패치를 읽고 → 정규화 → 모델 → 공식 후처리 순서로 진행합니다."
        ),
        code(
            "def read_rgb(x, y, w, h):\n"
            '    return np.array(slide.read_region((x, y), 0, (w, h)).convert("RGB"))\n'
            "\n"
            "def postprocess(out, i=0, magnification=40):\n"
            '    """모델 출력 → 세포 리스트 (공식 inference.py의 process_batch와 같은 순서)"""\n'
            '    ct, nb, hv, tt = (out[k][i] for k in ["cell_type_map", "nuclei_binary_map", "hv_map", "tissue_type_map"])\n'
            "    pred_map = np.concatenate([\n"
            "        tt.argmax(0)[..., None].numpy(), ct.argmax(0)[..., None].numpy(),\n"
            "        nb.argmax(0)[..., None].numpy(), hv.permute(1, 2, 0).numpy()], axis=-1)\n"
            "    pp = DetectionCellPostProcessor(nr_types=ct.shape[0], magnification=magnification, gt=False)\n"
            "    _, cells = pp.post_process_cell_segmentation(pred_map)\n"
            '    return [dict(centroid=np.asarray(c["centroid"], float), contour=np.asarray(c["contour"]), type=c["type"],\n'
            '                 type_prob=c["type_prob"]) for c in cells.values() if c["type"] > 0]\n'
            "\n"
            "def draw(img, cells, color_fn, thickness=1, offset=(0, 0)):\n"
            "    canvas = img.copy()\n"
            "    for c in cells:\n"
            '        cnt = (c["contour"] + np.asarray(offset)).astype(np.int32).reshape(-1, 1, 2)\n'
            "        cv2.drawContours(canvas, [cnt], -1, color_fn(c), thickness)\n"
            "    return canvas\n"
            "\n"
            "t2_color = lambda c: COLORS[c[\"type\"]]\n"
            "t1_color = lambda c: TIER1_COLORS[TIER2_TO_TIER1[CELL_TYPES[c[\"type\"]]]]"
        ),
        md(
            "### 3-1. 원시 출력\n"
            "\n"
            "종양과 면역세포가 섞인 영역의 256×256 패치 하나를 봅니다. 모델 입력은 224×224이지만 "
            "디코더가 **256×256으로 다시 올려서** 출력하므로, 결과 좌표는 원래 패치 좌표와 같습니다.\n"
            "\n"
            "- `nuclei_binary_map`, `hv_map`: CellViT와 같은 핵 분할 출력 (핵 확률 + 수평/수직 거리)\n"
            "- `cell_type_map`: 픽셀별 16클래스(배경 + 15종) 확률 → 핵 안에서 다수결로 세포 타입 결정\n"
            "- `tissue_type_map`(6채널, 픽셀별), `organ_types`(13개, 패치별): 학습 시 보조 출력으로 보이며, "
            "**공개된 config에 클래스 이름이 없고 공식 후처리에서도 쓰지 않습니다.** 이 튜토리얼에서도 해석하지 않습니다."
        ),
        code(
            "# §4에서 쓸 관심 영역(ROI): 종양과 림프 소포(B/T 세포 집합)가 함께 있는 6,144×6,144 px ≈ 1.5×1.5 mm (level-0 좌표)\n"
            "ROI_X, ROI_Y, ROI_SIZE = 10240, 6656, 6144\n"
            "PATCH_XY = (ROI_X + 1600, ROI_Y + 5120)  # 종양-림프구 경계의 한 패치\n"
            "raw = read_rgb(*PATCH_XY, 256, 256)\n"
            "img = reinhard(raw)\n"
            "out = run_model(img[None])\n"
            "cells = postprocess(out)\n"
            "\n"
            'nb_prob = F.softmax(out["nuclei_binary_map"][0], 0)[1].numpy()\n'
            'hv = out["hv_map"][0].numpy()\n'
            'ct = out["cell_type_map"][0].argmax(0).numpy()\n'
            "ct_rgb = np.full((*ct.shape, 3), 255, np.uint8)\n"
            "for k, col in COLORS.items():\n"
            "    ct_rgb[ct == k] = col\n"
            "\n"
            "fig, ax = plt.subplots(1, 6, figsize=(26, 4.6))\n"
            'panels = [(raw, "원본", None), (img, "Reinhard 정규화", None), (nb_prob, "핵 확률", "magma"),\n'
            '          (hv[0], "HV: 수평", "coolwarm"), (ct_rgb, "cell_type_map (argmax)", None),\n'
            '          (draw(img, cells, t2_color, 2), f"세포 {len(cells)}개 (15종)", None)]\n'
            "for a, (im, title, cmap) in zip(ax, panels):\n"
            '    a.imshow(im, cmap=cmap); a.set_title(title); a.axis("off")\n'
            "present = sorted({c['type'] for c in cells})\n"
            "legend(ax[-1], [(CELL_TYPES[k], COLORS[k]) for k in present])\n"
            "plt.tight_layout(); plt.show()"
        ),
        # ── 4. ROI / WSI
        md(
            "## 4. 슬라이드 영역(또는 전체) 추론\n"
            "\n"
            "공식 파이프라인과 같은 방식으로 **256px 패치를 64px씩 겹쳐**(stride 192) 자릅니다. "
            "겹친 영역의 세포가 두 번 세어지지 않도록, 각 패치에서 **가운데 192×192(가장자리 32px 제외)** 에 "
            "중심이 있는 세포만 남깁니다. 이웃 패치의 가운데 영역들이 빈틈없이 맞물리므로 중복도 누락도 없습니다.\n"
            "\n"
            "> 공식 `inference.py`는 중복 제거 없이 패치별 결과를 그대로 CSV에 씁니다. 이 예제 슬라이드 전체에서 공식 CSV는 "
            "**128,409행**이지만, 위 규칙으로 중복을 지우면 **68,125개**로 README의 기대값(약 63,000개)에 가까워집니다. "
            "또 공식 CSV에는 `Background`(type 0)로 판정된 객체가 약 17% 섞여 있는데, 이 노트북은 이를 제외합니다.\n"
            "\n"
            "| 설정 | 영역 | 패치 수 | Colab T4 예상 |\n"
            "|---|---|---|---|\n"
            "| `FULL_SLIDE = False` (기본) | 6,144×6,144 px ROI (약 1.5×1.5 mm) | 약 1,000 | 2~4분 |\n"
            "| `FULL_SLIDE = True` | 조직 전체 | 약 7,000 | 20~30분 |"
        ),
        code(
            "FULL_SLIDE = False\n"

            "BATCH = 16  # T4에서 GPU 메모리 약 6GB\n"
            "P, OV = 256, 64\n"
            "STRIDE, M = P - OV, OV // 2\n"
            "\n"
            "x_range = (0, W) if FULL_SLIDE else (ROI_X, ROI_X + ROI_SIZE)\n"
            "y_range = (0, H) if FULL_SLIDE else (ROI_Y, ROI_Y + ROI_SIZE)\n"
            "# 패치 시작점을 공식 patchify.py의 전역 격자(0, 192, 384, ...)에 맞춤 → 같은 패치가 잘려 결과가 일치\n"
            "snap = lambda v: -(-v // STRIDE) * STRIDE\n"
            "xs = list(range(snap(x_range[0]), x_range[1] - P + 1, STRIDE))\n"
            "ys = list(range(snap(y_range[0]), y_range[1] - P + 1, STRIDE))\n"
            "\n"
            "# 조직 패치만 (level-1 썸네일의 채도로 판단)\n"
            "low_hsv = cv2.cvtColor(low, cv2.COLOR_RGB2HSV)[..., 1] / 255.0\n"
            "d1 = slide.level_downsamples[lvl]\n"
            "def is_tissue(x, y, thr=0.1):\n"
            "    r0, c0 = int(y / d1), int(x / d1)\n"
            "    return low_hsv[r0:r0 + int(P / d1) + 1, c0:c0 + int(P / d1) + 1].mean() >= thr\n"
            "\n"
            "jobs = [(x, y) for y in ys for x in xs if is_tissue(x, y)]\n"
            'print(f"패치 {len(jobs)}개 (전체 격자 {len(xs) * len(ys)}개 중 조직)")'
        ),
        code(
            "from tqdm.auto import tqdm\n"
            "\n"
            "def keep_core(c, x, y):\n"
            '    """겹침 영역 중복 제거: 패치 가운데 영역(격자 가장자리 패치는 바깥쪽까지)에 중심이 있는 세포만"""\n'
            '    cx, cy = c["centroid"]\n'
            "    lo_x = 0 if x == xs[0] else M; hi_x = P if x == xs[-1] else P - M\n"
            "    lo_y = 0 if y == ys[0] else M; hi_y = P if y == ys[-1] else P - M\n"
            "    return lo_x <= cx < hi_x and lo_y <= cy < hi_y\n"
            "\n"
            "records, contours = [], []\n"
            "t0 = time.time()\n"
            "for b in tqdm(range(0, len(jobs), BATCH), desc=\"HNE2Cell\"):\n"
            "    batch = jobs[b:b + BATCH]\n"
            "    imgs = np.stack([reinhard(read_rgb(x, y, P, P)) for x, y in batch])\n"
            "    out = run_model(imgs)\n"
            "    for i, (x, y) in enumerate(batch):\n"
            "        for c in postprocess(out, i):\n"
            "            if not keep_core(c, x, y):\n"
            "                continue\n"
            '            name = CELL_TYPES[c["type"]]\n'
            '            gx, gy = c["centroid"] + (x, y)\n'
            '            records.append(dict(x=gx, y=gy, type=c["type"], cell_type=name, tier1=TIER2_TO_TIER1[name],\n'
            '                                type_prob=c["type_prob"], area_um2=cv2.contourArea(c["contour"].astype(np.float32)) * MPP ** 2))\n'
            '            contours.append(c["contour"] + (x, y))\n'
            "\n"
            "cells_df = pd.DataFrame(records)\n"
            "dt = time.time() - t0\n"
            'print(f"{len(cells_df):,}개 세포 | {dt:.0f}s ({len(jobs) / dt:.1f} 패치/s)")\n'
            "cells_df.head()"
        ),
        # ── 5. analysis
        md(
            "## 5. 단일세포 지도와 공간 분석\n"
            "\n"
            "### 5-1. 세포 지도"
        ),
        code(
            "x0, x1 = x_range; y0, y1 = (y_range[0], min(y_range[1], H))\n"

            "if FULL_SLIDE:\n"
            "    view, vs, ox, oy = thumb, DS, 0, 0\n"
            "else:\n"
            "    view = reinhard(read_rgb(x0, y0, x1 - x0, y1 - y0)[::4, ::4], block=32)\n"
            "    vs, ox, oy = 4, x0, y0\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(20, 9.5))\n"
            "ax[0].imshow(view); ax[0].set_title(\"H&E (정규화)\")\n"
            "ax[1].imshow(view, alpha=0.35)\n"
            "for t1, col in TIER1_COLORS.items():\n"
            "    d = cells_df[cells_df.tier1 == t1]\n"
            "    ax[1].scatter((d.x - ox) / vs, (d.y - oy) / vs, s=1.5, color=np.array(col) / 255, label=f\"{t1} ({len(d):,})\")\n"
            'ax[1].legend(markerscale=6, loc="upper left", bbox_to_anchor=(1.01, 1)); ax[1].set_title("계통 그룹별 세포 지도")\n'
            '[a.axis("off") for a in ax]\n'
            "plt.tight_layout(); plt.show()"
        ),
        code(
            "# 확대: 같은 영역을 계통 그룹 / 15종으로\n"
            "zx, zy, zs = ROI_X + 1280, ROI_Y + 5120, 768  # 종양 · 림프 소포 · 기질이 만나는 곳\n"
            "zimg = reinhard(read_rgb(zx, zy, zs, zs))\n"
            "zc = [dict(contour=cnt - (zx, zy), type=t) for cnt, t, (cx, cy) in zip(contours, cells_df.type, cells_df[[\"x\", \"y\"]].values)\n"
            "      if zx <= cx < zx + zs and zy <= cy < zy + zs]\n"
            "fig, ax = plt.subplots(1, 3, figsize=(24, 8))\n"
            'ax[0].imshow(zimg); ax[0].set_title("H&E")\n'
            'ax[1].imshow(draw(zimg, zc, t1_color, 2)); ax[1].set_title("계통 그룹 (5개)")\n'
            'ax[2].imshow(draw(zimg, zc, t2_color, 2)); ax[2].set_title("세포 타입 (15종)")\n'
            "legend(ax[1], list(TIER1_COLORS.items()))\n"
            "legend(ax[2], [(CELL_TYPES[k], COLORS[k]) for k in sorted({c['type'] for c in zc})])\n"
            '[a.axis("off") for a in ax]\n'
            "plt.tight_layout(); plt.show()"
        ),
        md("### 5-2. 세포 조성"),
        code(
            "order = [CELL_TYPES[k] for k in range(1, 16)]\n"
            "cnt = cells_df.cell_type.value_counts().reindex(order, fill_value=0)\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={\"width_ratios\": [3, 1]})\n"
            "ax[0].bar(order, cnt.values, color=[np.array(COLORS[k]) / 255 for k in range(1, 16)], ec=\"k\")\n"
            'ax[0].set_yscale("log"); ax[0].set_ylabel("세포 수 (log)"); ax[0].set_title("세포 타입 (15종)")\n'
            'ax[0].tick_params(axis="x", rotation=45)\n'
            "t1 = cells_df.tier1.value_counts(normalize=True).reindex(TIER1_COLORS)\n"
            "ax[1].pie(t1.values, labels=[f\"{k}\\n{v:.0%}\" for k, v in t1.items()], colors=[np.array(c) / 255 for c in TIER1_COLORS.values()])\n"
            'ax[1].set_title("계통 그룹 비율")\n'
            "plt.tight_layout(); plt.show()\n"
            "\n"
            'print(cells_df.groupby("cell_type").agg(n=("type", "size"), 핵면적_µm2=("area_um2", "median"), type_prob=("type_prob", "mean")).reindex(order).round(2))'
        ),
        md(
            "### 5-3. 공간 분석 예시\n"
            "\n"
            "HNE2Cell의 가치는 **세포 타입 + 위치**를 슬라이드 전체에서 얻는 데 있습니다. 두 가지 간단한 지표를 계산해 봅니다.\n"
            "\n"
            "1. **종양 침윤**: 각 면역세포 타입 중 종양세포 30 µm 이내에 있는 비율 (종양 안/경계 vs 기질)\n"
            "2. **TLS(3차 림프 구조) 후보**: B세포와 CD4 T세포가 빽빽하게 모인 군집 (DBSCAN)\n"
            "\n"
            "> TLS를 엄밀하게 정의하려면 형태·구성·위치 기준이 더 필요합니다. 여기서는 개념을 보여주는 단순화된 예시입니다."
        ),
        code(
            "from scipy.spatial import cKDTree\n"
            "from sklearn.cluster import DBSCAN\n"
            "\n"
            "um = lambda px: px * MPP\n"
            "xy = cells_df[[\"x\", \"y\"]].values\n"
            "tumor = cells_df.cell_type == \"Malignant\"\n"
            "dist_to_tumor, _ = cKDTree(xy[tumor]).query(xy) if tumor.any() else (np.full(len(xy), np.inf), None)\n"
            "cells_df[\"dist_to_tumor_um\"] = um(dist_to_tumor)\n"
            "\n"
            "immune = TIER1[\"Immune\"]\n"
            "infil = (cells_df[cells_df.cell_type.isin(immune)]\n"
            "         .assign(near=lambda d: d.dist_to_tumor_um <= 30)\n"
            "         .groupby(\"cell_type\").near.agg([\"size\", \"mean\"]).reindex(immune).dropna())\n"
            'infil.columns = ["세포 수", "종양 30µm 이내 비율"]\n'
            "print(infil.round(3))\n"
            "\n"
            "# TLS 후보: B + CD4 T 세포를 25 µm 반경으로 묶고, 50개 이상 & B 비율 ≥ 30%인 군집\n"
            "lym = cells_df[cells_df.cell_type.isin([\"B\", \"CD4 T\"])]\n"
            "lab = DBSCAN(eps=25 / MPP, min_samples=15).fit_predict(lym[[\"x\", \"y\"]].values) if len(lym) else []\n"
            "lym = lym.assign(cluster=lab)\n"
            "tls = (lym[lym.cluster >= 0].groupby(\"cluster\")\n"
            "       .agg(n=(\"x\", \"size\"), b_frac=(\"cell_type\", lambda s: (s == \"B\").mean()), x=(\"x\", \"mean\"), y=(\"y\", \"mean\"))\n"
            "       .query(\"n >= 50 and b_frac >= 0.3\"))\n"
            'print(f"\\nTLS 후보 {len(tls)}개")\n'
            "print(tls.round(2))"
        ),
        code(
            "fig, ax = plt.subplots(1, 2, figsize=(20, 9.5))\n"
            "# 면역세포 밀도 (100 µm 격자)\n"
            "bins = [np.arange(x0, x1 + 1, 100 / MPP), np.arange(y0, y1 + 1, 100 / MPP)]\n"
            "for a, (name, types) in zip(ax, [(\"종양세포\", [\"Malignant\"]), (\"림프구 (CD4/CD8 T, B, Plasma)\", [\"CD4 T\", \"CD8 T\", \"B\", \"Plasma\"])]):\n"
            "    d = cells_df[cells_df.cell_type.isin(types)]\n"
            "    h, _, _ = np.histogram2d(d.x, d.y, bins=bins)\n"
            "    im = a.imshow(h.T, cmap=\"magma\", extent=[0, (x1 - x0) / vs, (y1 - y0) / vs, 0])\n"
            '    a.set_title(f"{name} 밀도 (세포/100µm 격자)"); a.axis("off"); plt.colorbar(im, ax=a, fraction=0.046)\n'
            "for _, r in tls.iterrows():\n"
            '    ax[1].add_patch(plt.Circle(((r.x - ox) / vs, (r.y - oy) / vs), 150 / MPP / vs, fill=False, ec="cyan", lw=2))\n'
            'ax[1].set_title(ax[1].get_title() + "\\n(하늘색 원 = TLS 후보)")\n'
            "plt.tight_layout(); plt.show()"
        ),
        # ── 6. export
        md(
            "## 6. 결과 내보내기\n"
            "\n"
            "- `cells.csv`: 세포별 좌표(level-0 px), 타입, 계통 그룹, 확률, 면적, 종양까지 거리 → pandas / Seurat / squidpy로 후속 분석\n"
            "- `cells.geojson`: QuPath에서 `File → Import objects`로 원본 슬라이드 위에 겹쳐 보기"
        ),
        code(
            'OUT = WORKDIR / "outputs"; OUT.mkdir(exist_ok=True)\n'
            'cells_df.to_csv(OUT / "cells.csv", index=False)\n'
            "\n"
            "features = [{\n"
            '    "type": "Feature",\n'
            '    "geometry": {"type": "Polygon", "coordinates": [cnt.tolist() + [cnt[0].tolist()]]},\n'
            '    "properties": {"objectType": "detection", "classification": {"name": n, "color": list(COLORS[t])}},\n'
            "} for cnt, n, t in zip(contours, cells_df.cell_type, cells_df.type)]\n"
            '(OUT / "cells.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))\n'
            '!ls -lh {OUT}'
        ),
        md(
            "## 정리\n"
            "\n"
            "| 단계 | 핵심 |\n"
            "|---|---|\n"
            "| 입력 | 40x H&E, **Reinhard 정규화 필수**, 256px 패치(64px 겹침) → 224로 축소 |\n"
            "| 모델 | 대형 ViT encoder + CellViT식 디코더 (약 1.3B 파라미터), TorchScript |\n"
            "| 출력 | 핵 분할 + 15종 세포 타입 (5개 계통 그룹으로 묶을 수 있음) |\n"
            "| 후처리 | 공식 `post_processing.py` (watershed), 겹침 영역은 패치 가운데만 남겨 중복 제거 |\n"
            "\n"
            "**주의할 점**\n"
            "- CD4⁺/CD8⁺ T 세포는 H&E에서 형태가 거의 같아 구분이 본질적으로 어렵습니다. 정량 분석에서는 합쳐서 'T 세포'로 보는 것이 안전합니다.\n"
            "- 20x 슬라이드도 동작하지만(`DetectionCellPostProcessor(magnification=20)`) 작은 면역세포 정확도가 떨어집니다.\n"
            "- 모델 가중치는 **CC BY-NC 4.0** (비상업적 연구용). 상업적 이용은 이화여대 산학협력단 문의.\n"
            "\n"
            "**CellViT/CellViT++와의 관계**: HNE2Cell은 CellViT의 디코더 구조와 후처리를 그대로 이어받고, "
            "encoder를 대형 병리 ViT로 바꾼 뒤 **공간전사체 라벨로 학습**해 세포 타입을 15종까지 늘린 모델입니다."
        ),
    ]


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = cells()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, NB_PATH)
    print("wrote", NB_PATH)


if __name__ == "__main__":
    build()
