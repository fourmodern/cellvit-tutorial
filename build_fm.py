"""병리 foundation model (UNI-2, H-optimus-0) 튜토리얼 노트북 생성 스크립트.

python build_fm.py  →  05_UNI2.ipynb, 06_H-optimus-0.ipynb
"""
import textwrap

import nbformat as nbf

REPO = "fourmodern/cellvit-tutorial"


def md(t):
    return nbf.v4.new_markdown_cell(textwrap.dedent(t).strip())


def code(t):
    return nbf.v4.new_code_cell(textwrap.dedent(t).strip())


def sub(t, M):
    """템플릿을 먼저 dedent한 뒤 값을 끼워 넣는다 (값도 dedent) — 마크다운 들여쓰기가 코드 블록이 되지 않도록"""
    t = textwrap.dedent(t)
    for k, v in M.items():
        t = t.replace("{{" + k + "}}", textwrap.dedent(str(v)).strip())
    return t


# ─────────────────────────────────────────────────────────────
# 모델별 정보 (공개 모델 카드 기준)
# ─────────────────────────────────────────────────────────────
UNI2 = dict(
    NB="05_UNI2.ipynb",
    NAME="UNI-2",
    SHORT="uni2",
    HUB="MahmoodLab/UNI2-h",
    CARD="""
| 항목 | 내용 (HuggingFace 모델 카드 기준) |
|---|---|
| 개발 | Mahmood Lab (Harvard / Brigham and Women's Hospital) |
| 구조 | ViT-H/14 (DINOv2 방식 자기지도 학습), register token 8개 |
| 크기 | 약 6.8억 파라미터, 출력 1,536차원 |
| 학습 데이터 | Mass General Brigham의 H&E·IHC 슬라이드 35만 장 이상에서 뽑은 2억 개 이상 타일 |
| 입력 | 224×224 px, ImageNet 평균/표준편차로 정규화 |
| 라이선스 | **CC BY-NC-ND 4.0** — 비상업적 학술 연구용, 모델 재배포 금지 |
| 논문 | Chen et al., *Nature Medicine* 2024 (UNI) |
""",
    ACCESS="""
1. https://huggingface.co/MahmoodLab/UNI2-h 에서 **Request access** 양식 제출
   - 모델 카드에 따르면 HuggingFace 계정의 **주 이메일이 기관 이메일**이어야 승인됩니다 (gmail 등 개인 이메일은 거절).
2. https://huggingface.co/settings/tokens 에서 **Read** 토큰 발급
3. Colab 왼쪽 🔑(보안 비밀)에 이름 `HF_TOKEN`으로 토큰 저장 → 노트북 접근 허용
""",
    LOAD='''
    import timm
    TIMM_KWARGS = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                       mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                       act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)  # 모델 카드 그대로
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **TIMM_KWARGS)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet
    ''',
    COLOR="#7b3fb5",
)

HOPT = dict(
    NB="06_H-optimus-0.ipynb",
    NAME="H-optimus-0",
    SHORT="hoptimus0",
    HUB="bioptimus/H-optimus-0",
    CARD="""
| 항목 | 내용 (HuggingFace 모델 카드 기준) |
|---|---|
| 개발 | Bioptimus |
| 구조 | ViT-g/14 (DINOv2 계열), register token 4개 |
| 크기 | 약 11억 파라미터, 출력 1,536차원 |
| 학습 데이터 | 독자 수집한 H&E 전체 슬라이드 50만 장 이상 |
| 입력 | **224×224 px, 0.5 µm/px**, H&E 데이터에 맞춘 평균/표준편차로 정규화 |
| 라이선스 | **Apache 2.0** (사용 조건 동의 후 다운로드) |
""",
    ACCESS="""
1. https://huggingface.co/bioptimus/H-optimus-0 에서 사용 조건 동의 양식 제출 (자동 승인)
2. https://huggingface.co/settings/tokens 에서 **Read** 토큰 발급
3. Colab 왼쪽 🔑(보안 비밀)에 이름 `HF_TOKEN`으로 토큰 저장 → 노트북 접근 허용
""",
    LOAD='''
    import timm
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5,
                              dynamic_img_size=False)  # 모델 카드 그대로
    MEAN, STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)  # H&E 데이터 기준
    ''',
    COLOR="#d0602a",
)


# ─────────────────────────────────────────────────────────────
# 공통 셀
# ─────────────────────────────────────────────────────────────
def cells(M, compare_uni2=False):
    C = []
    nb = M["NB"]
    C.append(md(f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/{nb}" target="_parent">'
                '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'))
    C.append(md(sub("""
    # {{NAME}} 튜토리얼: 병리 foundation model로 조직 이해하기

    **{{NAME}}** 은 수많은 H&E 이미지로 **자기지도 학습**한 병리 foundation model입니다.
    224×224 px 조직 패치 하나를 넣으면 **1,536차원 벡터(임베딩)** 하나를 돌려줍니다.
    이 벡터에는 조직의 형태 정보가 압축되어 있어서, 그 위에 간단한 분류기만 얹어도 다양한 과제를 풀 수 있습니다.

    {{CARD}}

    ### 앞의 노트북들과 무엇이 다른가

    | | CellViT / HNE2Cell (01~04) | **{{NAME}}** |
    |---|---|---|
    | 단위 | 세포 하나하나 (핵 분할 + 분류) | **패치 전체** (224px ≈ 112 µm) |
    | 출력 | 세포 윤곽 + 세포 타입 | 패치당 1,536차원 벡터 |
    | 쓰는 법 | 그대로 결과 사용 | 벡터 위에 분류기·군집·검색 등을 얹어 사용 |

    ### 이 노트북에서 하는 것

    | § | 내용 |
    |---|---|
    | 1 | HuggingFace 접근 설정, 모델 불러오기 |
    | 2 | 대장 조직 데이터셋(9종 조직) 받기, 임베딩 추출 |
    | 3 | 임베딩 공간 보기: UMAP, 비슷한 패치 검색 |
    | 4 | **선형 분류기(linear probe)** 로 조직 분류 정확도, 적은 라벨 학습 곡선 (ImageNet 모델과 비교) |
    | 5 | 패치 토큰 PCA: 모델이 패치 안에서 무엇을 구분하는지 |
    | 6 | 전체 슬라이드에 적용: 조직 지도, 비지도 군집, 비슷한 영역 찾기 |
    """ + ("| 7 | **UNI-2와 비교** (같은 데이터, 같은 평가) |\n" if compare_uni2 else ""), M)))
    C.append(md(sub("""
    ## 0. 준비: HuggingFace 접근 권한

    {{NAME}}는 **접근 승인이 필요한(gated) 모델**입니다. 처음 한 번만 아래를 해 두세요.

    {{ACCESS}}
    **Colab 런타임**: `런타임 → 런타임 유형 변경 → T4 GPU`. 이 노트북은 런타임 재시작이 필요 없습니다.
    """, M)))
    C.append(code("""
    import sys, os, importlib.util
    from pathlib import Path

    IN_COLAB = "google.colab" in sys.modules
    need = [m for m in ["openslide", "umap", "timm"] if importlib.util.find_spec(m) is None]
    if need or IN_COLAB:
        !pip -q install -U "timm>=1.0.9" openslide-bin openslide-python umap-learn

    WORKDIR = Path("/content/fm_tutorial") if IN_COLAB else Path.cwd()
    WORKDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKDIR)
    DATA = WORKDIR / "fm_data"
    DATA.mkdir(exist_ok=True)
    print("WORKDIR:", WORKDIR)
    """))
    C.append(code("""
    # HuggingFace 로그인: Colab 보안 비밀 HF_TOKEN → 없으면 직접 입력 창
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
    print("HuggingFace 사용자:", user)
    """))
    C.append(code("""
    import json, time, gc, warnings, urllib.request
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
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
    """))

    # 1. model
    C.append(md(sub("""
    ## 1. 모델 불러오기

    모델 카드의 코드 그대로 `timm`으로 만듭니다. 가중치는 처음 한 번 HuggingFace에서 내려받습니다.
    """, M)))
    C.append(code(sub("""
    t = time.time()
    {{LOAD}}
    model = model.eval().to(DEVICE)
    TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    N_PREFIX = model.num_prefix_tokens  # CLS + register 토큰 수 (패치 토큰 앞에 붙음)
    print(f"{{NAME}} 로딩 {time.time() - t:.0f}s | 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | "
          f"출력 {model.num_features}차원 | 앞쪽 특수 토큰 {N_PREFIX}개")

    @torch.inference_mode()
    def embed(images, batch=64, mdl=None, tfm=None, tokens=False):
        \"\"\"PIL/ndarray 이미지 리스트 → (N, D) 임베딩. tokens=True면 (N, 256, D) 패치 토큰\"\"\"
        mdl, tfm = mdl or model, tfm or TRANSFORM
        out = []
        for i in range(0, len(images), batch):
            x = torch.stack([tfm(Image.fromarray(im) if isinstance(im, np.ndarray) else im) for im in images[i:i + batch]]).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                f = mdl.forward_features(x)[:, mdl.num_prefix_tokens:] if tokens else mdl(x)
            out.append(f.float().cpu())
        return torch.cat(out).numpy()
    """, M)))

    # 2. data
    C.append(md("""
    ## 2. 데이터: 대장 조직 9종 (Kather et al., CRC-VAL-HE-7K)

    대장암 수술 조직에서 잘라낸 **224×224 px, 0.5 µm/px** 패치 7,180장, 9종 조직 라벨 (CC BY 4.0, Zenodo, 약 800MB).
    두 foundation model의 권장 입력(224 px, 0.5 µm/px)과 같은 조건입니다.

    `N_PER_CLASS`로 클래스당 사용할 장수를 정합니다 (기본 300장 → 2,700장, Colab T4에서 임베딩 1~2분).
    `None`이면 전체를 씁니다.
    """))
    C.append(code("""
    import zipfile
    CRC = DATA / "CRC-VAL-HE-7K"
    if not CRC.exists():
        z = DATA / "CRC-VAL-HE-7K.zip"
        print("다운로드 중 (약 800MB) ...")
        urllib.request.urlretrieve("https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content", z)
        zipfile.ZipFile(z).extractall(DATA); z.unlink()

    TISSUE = {  # 약어: (한글 이름, 색)
        "ADI": ("지방", (255, 220, 120)), "BACK": ("배경", (200, 200, 200)), "DEB": ("괴사·잔해", (120, 120, 120)),
        "LYM": ("림프구", (30, 90, 255)), "MUC": ("점액", (120, 210, 230)), "MUS": ("평활근", (230, 120, 150)),
        "NORM": ("정상 점막", (60, 180, 75)), "STR": ("종양 기질", (150, 200, 90)), "TUM": ("종양 상피", (220, 30, 30)),
    }
    CLASSES = list(TISSUE)
    NAME = {k: f"{k} ({v[0]})" for k, v in TISSUE.items()}
    COLOR = {k: v[1] for k, v in TISSUE.items()}

    N_PER_CLASS = 300
    rng = np.random.default_rng(0)
    paths, labels = [], []
    for k in CLASSES:
        files = sorted((CRC / k).glob("*.tif"))
        if N_PER_CLASS:
            files = [files[i] for i in rng.choice(len(files), min(N_PER_CLASS, len(files)), replace=False)]
        paths += files; labels += [k] * len(files)
    labels = np.array(labels)
    images = [np.array(Image.open(p).convert("RGB")) for p in paths]
    print(f"{len(images)}장 | 크기 {images[0].shape} |", pd.Series(labels).value_counts().reindex(CLASSES).to_dict())
    """))
    C.append(code("""
    # 클래스별 예시 8장
    ex = [images[i] for k in CLASSES for i in np.flatnonzero(labels == k)[:8]]
    viz.image_grid(ex, ncols=8, size=1.5, row_labels=[NAME[k] for k in CLASSES], suptitle="9종 대장 조직 예시 (224×224 px = 112×112 µm)")
    """))
    C.append(md(sub("""
    ### 임베딩 추출

    패치 한 장 → {{NAME}} → 1,536차원 벡터. 이후 모든 분석은 이 벡터만으로 합니다.
    """, M)))
    C.append(code("""
    t = time.time()
    X = embed(images)
    print(f"임베딩 {X.shape} | {time.time() - t:.0f}s ({len(images) / (time.time() - t):.0f}장/s)")
    """))

    # 3. embedding space
    C.append(md("""
    ## 3. 임베딩 공간 들여다보기

    ### 3-1. UMAP

    1,536차원을 2차원으로 줄여 그립니다. **라벨을 전혀 쓰지 않은 모델**인데도 같은 조직끼리 모인다면,
    임베딩이 조직 형태를 잘 담고 있다는 뜻입니다. 오른쪽 인터랙티브 그림은 점 위에 마우스를 올리면 클래스가 보입니다.
    """))
    C.append(code("""
    import umap
    U = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    for k in CLASSES:
        m = labels == k
        ax.scatter(U[m, 0], U[m, 1], s=6, color=np.array(COLOR[k]) / 255, label=NAME[k])
    # 클래스마다 대표 패치 하나를 UMAP 위에 붙여 보기
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for k in CLASSES:
        i = np.flatnonzero(labels == k)[np.argmin(np.linalg.norm(U[labels == k] - np.median(U[labels == k], 0), axis=1))]
        ab = AnnotationBbox(OffsetImage(images[i], zoom=0.18), U[i], frameon=True, bboxprops=dict(edgecolor=np.array(COLOR[k]) / 255, lw=2))
        ax.add_artist(ab)
    ax.legend(markerscale=3, loc="upper left", bbox_to_anchor=(1.01, 1)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("UMAP of embeddings (색 = 실제 조직, 사진 = 각 군집 중심의 패치)")
    plt.tight_layout(); plt.show()
    """))
    C.append(code("""
    import plotly.graph_objects as go
    fig = go.Figure()
    for k in CLASSES:
        m = labels == k
        r, g, b = COLOR[k]
        fig.add_trace(go.Scattergl(x=U[m, 0], y=U[m, 1], mode="markers", name=NAME[k], marker=dict(size=5, color=f"rgb({r},{g},{b})"),
                                   text=[f"{NAME[k]}<br>{paths[i].name}" for i in np.flatnonzero(m)], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title="UMAP (인터랙티브: 범례 클릭으로 클래스 켜고 끄기)", height=600, plot_bgcolor="white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.show()
    """))
    C.append(md("""
    ### 3-2. 비슷한 패치 찾기 (임베딩 검색)

    각 클래스에서 질의 패치를 하나 고르고, 코사인 유사도가 가장 높은 패치 6장을 찾습니다.
    **테두리 초록 = 같은 클래스, 빨강 = 다른 클래스.** 라벨 없이도 비슷한 조직을 찾아 주는지 확인합니다.
    """))
    C.append(code("""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    grid, borders, titles, rows = [], [], [], []
    for k in CLASSES:
        q = np.flatnonzero(labels == k)[3]
        sim = Xn @ Xn[q]; sim[q] = -1
        top = np.argsort(-sim)[:6]
        grid += [images[q]] + [images[j] for j in top]
        borders += [(0, 0, 0)] + [(40, 170, 60) if labels[j] == k else (220, 40, 40) for j in top]
        titles += ["질의"] + [f"{labels[j]} {sim[j]:.2f}" for j in top]
        rows.append(NAME[k])
    viz.image_grid(grid, titles=titles, ncols=7, size=1.55, border_colors=borders, row_labels=rows,
                   suptitle="질의 패치(왼쪽)와 가장 비슷한 6장 (제목: 클래스, 코사인 유사도)")
    """))

    # 4. linear probe
    C.append(md("""
    ## 4. 선형 분류기(linear probe)로 정확도 재기

    임베딩은 그대로 두고 **로지스틱 회귀**만 학습합니다. foundation model 평가의 표준 방식입니다.
    데이터를 학습 70% / 평가 30%로 나눕니다 (평가 세트는 학습에 전혀 쓰지 않음).
    """))
    C.append(code("""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score

    idx_tr, idx_te = train_test_split(np.arange(len(labels)), test_size=0.3, stratify=labels, random_state=0)
    y = np.array([CLASSES.index(l) for l in labels])

    def probe(Xf, tr, te, C_=0.5):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C_))
        clf.fit(Xf[tr], y[tr])
        return clf, clf.predict(Xf[te]), clf.predict_proba(Xf[te]).max(1)

    t = time.time()
    clf, pred_full, _ = probe(X, idx_tr, idx_te)
    print(f"학습 세트 전체({len(idx_tr)}장)로 학습 → 평가 {len(idx_te)}장 balanced accuracy "
          f"{balanced_accuracy_score(y[idx_te], pred_full):.3f} ({time.time() - t:.1f}s)")

    # 이 데이터셋은 라벨을 다 쓰면 거의 100%라서, 클래스당 K_SHOW장만으로 학습해 어디서 헷갈리는지 봅니다.
    K_SHOW = 4
    r = np.random.default_rng(0)
    tr_k = np.concatenate([r.choice(idx_tr[y[idx_tr] == c], K_SHOW, replace=False) for c in range(len(CLASSES))])
    clf_k, pred, conf = probe(X, tr_k, idx_te)
    viz.classification_metrics(y[idx_te], pred, {i: k for i, k in enumerate(CLASSES)}, {i: COLOR[k] for i, k in enumerate(CLASSES)},
                               title=f"클래스당 {K_SHOW}장(총 {len(tr_k)}장)만으로 학습한 선형 분류기 — 평가 세트")
    viz.confidence_vs_accuracy(conf, pred == y[idx_te], bins=6)
    """))
    C.append(code("""
    # 틀린 예측 모아 보기 (클래스당 K_SHOW장 분류기): 무엇을 무엇으로 헷갈렸나
    wrong = np.flatnonzero(pred != y[idx_te])
    pick = wrong[np.argsort(-conf[wrong])][:24]
    if len(pick) == 0:
        print("틀린 예측이 없습니다.")
    else:
        viz.image_grid([images[idx_te[i]] for i in pick], titles=[f"정답 {CLASSES[y[idx_te[i]]]}\\n예측 {CLASSES[pred[i]]} ({conf[i]:.2f})" for i in pick],
                       ncols=8, size=1.6, border_colors=[(220, 40, 40)] * len(pick),
                       suptitle=f"클래스당 {K_SHOW}장 분류기가 확신하고 틀린 예측 {len(pick)}장 (전체 오답 {len(wrong)}장)")
    """))
    C.append(md(sub("""
    ### 4-2. 라벨이 적을 때: few-shot 학습 곡선

    클래스당 라벨 k장만으로 학습하면 성능이 어떻게 될까요? 비교를 위해 **ImageNet으로만 학습한 일반 모델(ResNet-50)** 의 임베딩도 같은 방식으로 평가합니다.
    두 곡선의 차이가 곧 '병리 데이터로 사전학습한 효과'입니다. (각 k마다 5번 무작위로 뽑아 평균)
    """, M)))
    C.append(code(sub("""
    import timm
    imnet = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0).eval().to(DEVICE)
    imnet.num_prefix_tokens = 0
    cfg = imnet.pretrained_cfg
    imnet_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])
    X_imnet = embed(images, mdl=imnet, tfm=imnet_tf)
    del imnet; gc.collect(); torch.cuda.empty_cache()

    def few_shot(Xf, ks=(1, 2, 4, 8, 16, 32, 64, 128), seeds=5):
        res = []
        for k in ks:
            for s in range(seeds):
                r = np.random.default_rng(s)
                tr = np.concatenate([r.choice(idx_tr[y[idx_tr] == c], min(k, (y[idx_tr] == c).sum()), replace=False) for c in range(len(CLASSES))])
                _, p, _ = probe(Xf, tr, idx_te)
                res.append((k, balanced_accuracy_score(y[idx_te], p)))
        return pd.DataFrame(res, columns=["k", "acc"]).groupby("k").acc.agg(["mean", "std"])

    FS = {"{{NAME}}": few_shot(X), "ImageNet ResNet-50": few_shot(X_imnet)}
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, d), col in zip(FS.items(), ["{{COLOR}}", "#888"]):
        ax.errorbar(d.index, d["mean"], yerr=d["std"], marker="o", capsize=3, label=name, color=col, lw=2)
        ax.text(d.index[-1] * 1.08, d["mean"].iloc[-1], f"{d['mean'].iloc[-1]:.2f}", va="center", color=col)
    ax.set_xscale("log", base=2); ax.set_xlabel("클래스당 학습 라벨 수 (k)"); ax.set_ylabel("balanced accuracy (평가 세트)")
    ax.set_ylim(0, 1.02); ax.axhline(1 / len(CLASSES), color="k", ls=":", lw=1); ax.text(1, 1 / len(CLASSES) + 0.02, "무작위 추측", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(); ax.set_title("적은 라벨로 학습할 때의 성능")
    plt.tight_layout(); plt.show()
    print(pd.concat({n: d["mean"].round(3) for n, d in FS.items()}, axis=1).T)
    """, M)))

    # 5. token PCA
    C.append(md("""
    ## 5. 패치 토큰 PCA: 모델은 패치 안에서 무엇을 구분할까

    ViT는 224 px 패치를 14×14 px 조각 256개(16×16)로 나눠 각각 토큰을 만듭니다.
    여러 패치의 토큰을 모아 PCA로 3개 주성분을 뽑고 **RGB 색**으로 칠하면, 모델이 같은 '종류'로 보는 영역이 같은 색으로 나타납니다.
    (라벨을 전혀 쓰지 않은 결과입니다. 핵 / 기질 / 지방 / 점액 등이 다른 색으로 갈리는지 보세요.)
    """))
    C.append(code("""
    from sklearn.decomposition import PCA
    sel = [np.flatnonzero(labels == k)[1] for k in CLASSES]
    T = embed([images[i] for i in sel], tokens=True)            # (9, 256, D)
    g = int(np.sqrt(T.shape[1]))
    pcs = PCA(3, random_state=0).fit_transform(T.reshape(-1, T.shape[-1]))
    pcs = (pcs - pcs.min(0)) / (pcs.max(0) - pcs.min(0) + 1e-9)
    pcs = pcs.reshape(len(sel), g, g, 3)

    fig, ax = plt.subplots(2, len(sel), figsize=(2.1 * len(sel), 4.6))
    for j, (i, k) in enumerate(zip(sel, CLASSES)):
        ax[0, j].imshow(images[i]); ax[0, j].set_title(NAME[k], fontsize=9)
        ax[1, j].imshow(np.array(Image.fromarray((pcs[j] * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("조직"); ax[1, 0].set_ylabel("토큰 PCA")
    plt.tight_layout(); plt.show()
    """))

    # 6. WSI
    C.append(md("""
    ## 6. 전체 슬라이드에 적용하기

    예제 슬라이드는 TCGA 공개 데이터의 **대장암 진단 슬라이드** (TCGA-AD-6890, GDC open access, 73MB, 40x = 0.25 µm/px)입니다.
    종양, 정상 점막, 기질, 림프 응집이 한 장에 같이 있습니다.
    모델 입력 해상도 0.5 µm/px에 맞추기 위해 **444×444 px을 읽어 224로 줄입니다.**
    조직 타일 약 1,800개를 읽고 임베딩하는 데 Colab T4에서 5~10분 걸립니다.

    1. **조직 지도**: §4의 대장 조직 분류기를 슬라이드 전체에 적용
    2. **비지도 군집**: 라벨 없이 임베딩을 k-means로 묶어 슬라이드를 영역별로 나누기
    3. **비슷한 영역 찾기**: 질의 타일 하나와 닮은 곳을 슬라이드 전체에서 찾기
    """))
    C.append(code("""
    import openslide, cv2
    SLIDE = DATA / "TCGA-AD-6890-01Z-00-DX1.svs"
    if not SLIDE.exists():  # GDC open access (TCGA-COAD 진단 슬라이드, 73MB)
        urllib.request.urlretrieve("https://api.gdc.cancer.gov/data/c7e229f4-7185-4211-979c-6cca1bbe4e0f", SLIDE)
    SLIDE = str(SLIDE)
    slide = openslide.OpenSlide(SLIDE)
    MPP = float(slide.properties["openslide.mpp-x"])
    TILE = int(round(224 * 0.5 / MPP))  # 0.5 µm/px 224px에 해당하는 level-0 크기 (≈448)
    W, H = slide.dimensions

    thumb = np.array(slide.get_thumbnail((2000, 2000)).convert("RGB"))
    DS = W / thumb.shape[1]
    sat = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[..., 1] / 255
    xy = []
    for y0 in range(0, H - TILE + 1, TILE):
        for x0 in range(0, W - TILE + 1, TILE):
            r0, c0, s = int(y0 / DS), int(x0 / DS), max(1, int(TILE / DS))
            if sat[r0:r0 + s, c0:c0 + s].mean() > 0.12:
                xy.append((x0, y0))
    xy = np.array(xy)
    print(f"슬라이드 {W}×{H} @ {MPP} µm/px | 타일 {TILE}px → 224px | 조직 타일 {len(xy)}개")

    t = time.time()
    tiles = [np.array(slide.read_region((int(x), int(y)), 0, (TILE, TILE)).convert("RGB").resize((224, 224), Image.BILINEAR)) for x, y in xy]
    XW = embed(tiles)
    print(f"슬라이드 임베딩 {XW.shape} | {time.time() - t:.0f}s")
    """))
    C.append(md("""
    ### 6-1. 조직 지도

    §4의 9종 조직 분류기(클래스당 300장으로 학습)를 슬라이드의 모든 타일에 적용합니다.
    왼쪽 조직 사진과 비교해 종양 / 정상 점막 / 기질 / 림프 응집이 제자리에 나오는지 보세요.

    > ⚠️ **장기가 다르면 그대로 쓸 수 없습니다.** 이 분류기를 폐 편평세포암 슬라이드에 적용해 보면 종양을 거의 찾지 못합니다
    > (대장 선암과 폐 편평세포암은 종양 형태가 다르기 때문). 분류기는 적용할 조직과 같은 종류의 데이터로 학습해야 합니다.
    """))
    C.append(code("""
    full_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X, y)
    PW = full_clf.predict_proba(XW)
    lab_w = np.array([CLASSES[i] for i in PW.argmax(1)])

    fig, ax = plt.subplots(2, 1, figsize=(16, 15))
    ax[0].imshow(thumb); ax[0].set_title("조직 사진 (H&E)"); ax[0].set_xticks([]); ax[0].set_yticks([])
    viz.tile_map(thumb, DS, xy, TILE, labels=lab_w, color_map=COLOR, ax=ax[1], legend_order=CLASSES, title="타일별 조직 예측")
    plt.tight_layout(); plt.show()
    viz.tile_map_interactive(thumb, DS, xy, TILE, lab_w, COLOR, order=CLASSES,
                             hover=[f"{NAME[l]}<br>확률 {p:.2f}" for l, p in zip(lab_w, PW.max(1))],
                             title="조직 지도 (인터랙티브: 휠로 확대, 범례 클릭)")
    """))
    C.append(code("""
    # 예측된 조직별로 타일 사진 모아 보기 — 예측이 그럴듯한지 눈으로 확인
    show = [k for k in CLASSES if (lab_w == k).sum() >= 3]
    grid, rows = [], []
    for k in show:
        ii = np.flatnonzero(lab_w == k); ii = ii[np.argsort(-PW[ii, CLASSES.index(k)])][:8]
        grid += [tiles[i] for i in ii] + [np.full((224, 224, 3), 255, np.uint8)] * (8 - len(ii))
        rows.append(f"{NAME[k]}\\n({(lab_w == k).sum()}개)")
    viz.image_grid(grid, ncols=8, size=1.5, row_labels=rows, suptitle="슬라이드에서 각 조직으로 예측된 타일 (확률 높은 순)")
    """))
    C.append(md("""
    ### 6-2. 비지도 군집 (라벨 없이)

    임베딩을 k-means로 8개 군집으로 나눕니다. 라벨을 전혀 쓰지 않았는데도 슬라이드의 구조(종양 둥지, 기질, 림프 응집, 괴사 등)가 드러나는지 봅니다.
    """))
    C.append(code("""
    from sklearn.cluster import KMeans
    K = 8
    km = KMeans(K, n_init=10, random_state=0).fit(XW / np.linalg.norm(XW, axis=1, keepdims=True))
    cl = np.array([f"군집 {c}" for c in km.labels_])
    pal = plt.get_cmap("tab10")
    CCOL = {f"군집 {c}": tuple(int(255 * v) for v in pal(c)[:3]) for c in range(K)}
    order = [f"군집 {c}" for c in range(K)]
    viz.tile_map(thumb, DS, xy, TILE, labels=cl, color_map=CCOL, legend_order=order, title="k-means 군집 지도 (라벨 없음)")

    grid, rows = [], []
    for c in range(K):
        ii = np.flatnonzero(km.labels_ == c)
        ii = ii[np.argsort(np.linalg.norm(XW[ii] / np.linalg.norm(XW[ii], axis=1, keepdims=True) - km.cluster_centers_[c], axis=1))][:8]
        grid += [tiles[i] for i in ii] + [np.full((224, 224, 3), 255, np.uint8)] * (8 - len(ii))
        top = pd.Series(lab_w[km.labels_ == c]).value_counts().index[0]
        rows.append(f"군집 {c} ({(km.labels_ == c).sum()}개)\\n주로 {top}")
    viz.image_grid(grid, ncols=8, size=1.4, row_labels=rows, suptitle="군집별 대표 타일 (군집 중심에 가까운 순) · 오른쪽 글씨 = 조직 분류기가 가장 많이 준 라벨")
    """))
    C.append(md("""
    ### 6-3. 비슷한 영역 찾기

    질의 타일 하나(기본: 종양 상피 확률이 가장 높은 타일)를 고르고, 슬라이드의 모든 타일과 **코사인 유사도**를 칠합니다.
    `QUERY`에 다른 타일 번호를 넣거나 `QUERY_XY`에 좌표를 넣어 원하는 조직을 찾아보세요.
    """))
    C.append(code("""
    QUERY_XY = None  # 예: (14000, 6000) → 그 좌표에 가장 가까운 타일
    QUERY = int(np.argmin(np.linalg.norm(xy - np.array(QUERY_XY), axis=1))) if QUERY_XY else int(PW[:, CLASSES.index("TUM")].argmax())
    XWn = XW / np.linalg.norm(XW, axis=1, keepdims=True)
    simw = XWn @ XWn[QUERY]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 2.4]})
    a0.imshow(tiles[QUERY]); a0.set_title(f"질의 타일 #{QUERY}"); a0.set_xticks([]); a0.set_yticks([])
    viz.tile_map(thumb, DS, xy, TILE, values=simw, cmap="inferno", vmin=np.percentile(simw, 5), vmax=1, ax=a1,
                 title="질의와의 코사인 유사도", colorbar_label="유사도")
    a1.add_patch(plt.Rectangle((xy[QUERY, 0] / DS, xy[QUERY, 1] / DS), TILE / DS, TILE / DS, fill=False, ec="cyan", lw=2))
    plt.tight_layout(); plt.show()

    top = np.argsort(-simw)[1:17]
    viz.image_grid([tiles[i] for i in top], titles=[f"{simw[i]:.2f}" for i in top], ncols=8, size=1.5, suptitle="가장 비슷한 타일 16개")
    """))

    if compare_uni2:
        C.append(md("""
        ## 7. UNI-2와 비교

        같은 데이터·같은 평가 방법으로 **UNI-2**도 돌려 봅니다 (UNI-2 접근 권한이 있어야 하며, 없으면 이 섹션은 건너뜁니다).
        두 모델은 GPU에 하나씩 올립니다.
        """))
        C.append(code("""
        import timm
        X_h0 = X
        model = None; gc.collect(); torch.cuda.empty_cache()
        try:
            uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24, num_heads=24,
                                    init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                                    mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True).eval().to(DEVICE)
            uni_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            t = time.time(); X_uni = embed(images, mdl=uni, tfm=uni_tf); t_uni = time.time() - t
            XW_uni = embed(tiles, mdl=uni, tfm=uni_tf)
            del uni; gc.collect(); torch.cuda.empty_cache()
            HAVE_UNI = True
        except Exception as e:
            print("UNI-2를 불러올 수 없어 비교를 건너뜁니다:", type(e).__name__, str(e)[:200])
            HAVE_UNI = False
        """))
        C.append(code("""
        if HAVE_UNI:
            FS["UNI-2"] = few_shot(X_uni)
            fig, ax = plt.subplots(1, 2, figsize=(17, 5.2))
            for (name, d), col in zip(FS.items(), ["#d0602a", "#888", "#7b3fb5"]):
                ax[0].errorbar(d.index, d["mean"], yerr=d["std"], marker="o", capsize=3, label=name, color=col, lw=2)
            ax[0].set_xscale("log", base=2); ax[0].set_xlabel("클래스당 학습 라벨 수 (k)"); ax[0].set_ylabel("balanced accuracy")
            ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3); ax[0].legend(); ax[0].set_title("few-shot 학습 곡선")

            from sklearn.metrics import f1_score
            f1s = {}
            for name, Xf in [("H-optimus-0", X_h0), ("UNI-2", X_uni)]:
                _, p, _ = probe(Xf, idx_tr, idx_te)
                f1s[name] = f1_score(y[idx_te], p, average=None, labels=range(len(CLASSES)))
            xx = np.arange(len(CLASSES))
            ax[1].bar(xx - 0.2, f1s["H-optimus-0"], 0.4, label="H-optimus-0", color="#d0602a")
            ax[1].bar(xx + 0.2, f1s["UNI-2"], 0.4, label="UNI-2", color="#7b3fb5")
            ax[1].set_xticks(xx); ax[1].set_xticklabels(CLASSES); ax[1].set_ylim(0.5, 1.02); ax[1].legend()
            ax[1].set_title("선형 분류기 클래스별 F1 (전체 학습 세트)"); ax[1].grid(axis="y", alpha=0.3)
            plt.tight_layout(); plt.show()
            print(pd.concat({n: d["mean"].round(3) for n, d in FS.items()}, axis=1).T)
        """))
        C.append(code("""
        if HAVE_UNI:
            # 같은 슬라이드에서 두 모델의 조직 지도가 얼마나 같은지
            uni_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X_uni, y)
            lab_uni = np.array([CLASSES[i] for i in uni_clf.predict(XW_uni)])
            agree = (lab_uni == lab_w).mean()
            fig, ax = plt.subplots(1, 2, figsize=(22, 6.5))
            viz.tile_map(thumb, DS, xy, TILE, labels=lab_w, color_map=COLOR, ax=ax[0], legend_order=CLASSES, title="H-optimus-0 조직 지도")
            viz.tile_map(thumb, DS, xy, TILE, labels=lab_uni, color_map=COLOR, ax=ax[1], legend_order=CLASSES, title="UNI-2 조직 지도")
            fig.suptitle(f"두 모델의 타일 예측 일치율 {agree:.1%}", fontsize=14)
            plt.tight_layout(); plt.show()
            ct = pd.crosstab(pd.Series(lab_w, name="H-optimus-0"), pd.Series(lab_uni, name="UNI-2")).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
            display(ct)
        """))
        C.append(md("""
        **해석 시 주의**: 이 데이터셋은 두 모델 모두 거의 다 맞히는 쉬운 과제라 차이가 작게 나옵니다.
        실제 연구에서는 목적에 맞는(더 어려운) 데이터로 비교해야 합니다.
        또 공개 벤치마크 데이터는 사전학습 데이터와 겹쳤을 가능성도 있습니다.
        """))

    C.append(md(sub("""
    ## 정리

    | 단계 | 핵심 |
    |---|---|
    | 입력 | 224×224 px @ 0.5 µm/px (40x 슬라이드는 448 px을 읽어 224로) |
    | 출력 | 패치당 1,536차원 임베딩 (패치 토큰은 16×16개) |
    | 쓰는 법 | 선형 분류기·kNN·군집·유사도 검색 등을 임베딩 위에 얹기 |

    **이 노트북에서 본 것**
    - 라벨 없이 학습한 임베딩만으로 9종 조직이 UMAP에서 잘 나뉘고, 검색으로 비슷한 조직을 찾을 수 있습니다.
    - 선형 분류기만으로 높은 정확도가 나오고, **클래스당 수십 장의 라벨만으로도** ImageNet 모델보다 훨씬 높은 성능을 냅니다.
    - 슬라이드 전체에 적용하면 조직 지도, 비지도 영역 분할, 비슷한 영역 찾기가 가능합니다.

    **다음 단계**: 슬라이드 단위 예측(예후, 유전자 변이 등)은 타일 임베딩들을 모으는 MIL(multiple instance learning) 모델로 확장합니다.

    **라이선스**: {{LICENSE}}
    """, dict(M, LICENSE="CC BY-NC-ND 4.0 — 비상업적 학술 연구용, 모델·파생 데이터의 상업적 이용 및 모델 재배포 금지" if M["SHORT"] == "uni2"
              else "Apache 2.0 — 단, 의료 목적 사용 시 규제 요건과 독립적 검증은 사용자 책임 (모델 카드 조건)"))))
    return C


def build(M, compare_uni2=False):
    nb = nbf.v4.new_notebook()
    nb.cells = cells(M, compare_uni2)
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, M["NB"])
    print("wrote", M["NB"])


if __name__ == "__main__":
    build(UNI2)
    build(HOPT, compare_uni2=True)
