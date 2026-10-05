# 병리 AI 모델 튜토리얼 (Colab용): CellViT · CellViT++ · HNE2Cell · UNI-2 · H-optimus-0 · Virchow2

| 노트북 | 내용 |
|---|---|
| `01_CellViT_tasks.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/01_CellViT_tasks.ipynb) | CellViT의 5가지 task: 조직 분류, 세포핵 검출, 인스턴스 분할, 세포 분류(PanNuke 5종), 세포 임베딩. 마지막에 QuPath용 GeoJSON 내보내기 |
| `02_CellViT_plusplus.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/02_CellViT_plusplus.ipynb) | CellViT++: 사전학습 분류기 7종, 같은 세포에 분류기 바꿔 끼우기, WSI 전체 추론(CLI), **점 라벨로 나만의 분류기 학습** 후 WSI에 적용 |
| `03_HNE2Cell.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/03_HNE2Cell.ipynb) | HNE2Cell: 공간전사체 라벨로 학습한 15종 세포 분류. 색 정규화, 패치/영역 추론, 계통 그룹, 종양 침윤·TLS 후보 공간 분석, CSV/GeoJSON 내보내기 |
| `04_HNE2Cell_vs_CellViT.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/04_HNE2Cell_vs_CellViT.ipynb) | 같은 6개 영역에서 CellViT(PanNuke)·CellViT++(Lizard)·HNE2Cell 비교: 검출 IoU 매칭, 클래스 교차표, 공통 그룹 일치도(정답 없는 일치도 분석) |
| `05_UNI2.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/05_UNI2.ipynb) | UNI-2 (병리 foundation model, 패치 임베딩): 임베딩 UMAP·검색, 선형 분류기 정확도, 적은 라벨 학습 곡선(ImageNet 대비), 패치 토큰 PCA, 슬라이드 조직 지도·비지도 군집·유사 영역 검색 |
| `06_H-optimus-0.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/06_H-optimus-0.ipynb) | H-optimus-0: 05와 같은 구성 + UNI-2와 같은 데이터·같은 평가로 비교 |
| `07_Virchow2.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/07_Virchow2.ipynb) | Virchow2 (CLS + 패치 평균 2,560차원 임베딩): 05와 같은 구성 + **UNI-2 · H-optimus-0 · Virchow2 세 모델 비교** (few-shot 곡선, 클래스별 F1, 처리 속도, 슬라이드 조직 지도 일치율) |

## 시각화

모든 노트북은 공통 시각화 모듈 `viz.py`를 씁니다 (Colab에서는 GitHub에서 자동으로 받습니다).

| 기능 | 내용 |
|---|---|
| 조직 사진 나란히 보기 | 원본 H&E · 윤곽선 · 반투명 채움을 같은 배율로 |
| 인터랙티브 뷰어 (plotly) | 휠로 확대/드래그로 이동, 세포에 마우스를 올리면 타입·확신도·면적, 범례 클릭으로 타입 켜고 끄기, 버튼으로 분류 체계 전환, 여러 패널 연동 확대 |
| 슬라이드 탐색기 (ipywidgets) | 슬라이더로 위치·크기를 골라 원본 해상도 조직 사진 + 세포 윤곽 |
| 세포 갤러리 | 타입별 실제 세포 사진 모음, 두 모델 판정이 엇갈린 세포 모음 |
| 분포 대시보드 | 타입별 세포 수·비율, 확신도·핵 면적·종양까지 거리 분포 |
| 정확도 / 일치도 | 정답이 있을 때 클래스별 precision·recall·F1, 혼동행렬, 확신도-정답률 곡선 (02 Part 4) · 정답이 없을 때 타입별 모델 간 일치율 (04) |

> 인터랙티브 그림과 탐색기는 Colab/Jupyter에서 실행할 때 보입니다 (GitHub 미리보기에서는 보이지 않음).

## Colab에서 실행

1. 위 표의 **Open in Colab** 배지를 누릅니다.
2. `런타임 → 런타임 유형 변경 → T4 GPU`를 선택합니다.
3. 첫 셀을 실행하면 패키지 설치 후 런타임이 **자동으로 재시작**됩니다. 재시작 후 처음부터 다시 실행하세요.
4. 모델(SAM-H 약 2.8GB)과 예제 데이터(약 600MB)는 노트북이 Zenodo에서 자동으로 받습니다.
   Zenodo 속도에 따라 10~30분 걸릴 수 있습니다. `USE_DRIVE_CACHE = True`로 두면 Google Drive에 캐시됩니다.

## UNI-2 / H-optimus-0 노트북 참고

- 두 모델 모두 HuggingFace에서 **접근 승인(gated)** 이 필요합니다. 모델 페이지에서 신청 → Read 토큰 발급 → Colab 보안 비밀 `HF_TOKEN`에 저장.
  - UNI-2 (`MahmoodLab/UNI2-h`): 모델 카드에 따르면 HuggingFace 계정 주 이메일이 **기관 이메일**이어야 승인. CC BY-NC-ND 4.0.
  - H-optimus-0 (`bioptimus/H-optimus-0`): 양식 제출 후 자동 승인. Apache 2.0.
  - Virchow2 (`paige-ai/Virchow2`): 기관 이메일 필요. CC BY-NC-ND 4.0, 학술 연구 전용(임상·RUO·상업적 이용 금지).
- 학습·평가 데이터: Kather et al. CRC-VAL-HE-7K (대장 조직 9종, 224px @ 0.5 µm/px, CC BY 4.0, 800MB, Zenodo).
- 슬라이드 예제: TCGA-AD-6890 대장암 진단 슬라이드 (GDC open access, 73MB).

## HNE2Cell 노트북 참고

- 모델·코드·예제 슬라이드(TCGA-LUSC)는 HuggingFace `roobee79/HNE2Cell`에서 자동으로 받습니다. 모델이 5.1GB라 첫 실행 시 다운로드 시간이 걸립니다.
- 공식 `normalize.py`는 슬라이드 전체를 메모리에 올려 예제 슬라이드에서도 RAM 35GB를 씁니다. 노트북은 같은 Reinhard 정규화를 타일/패치 단위로 계산해 Colab 무료 런타임에서 돌아갑니다.
- 같은 영역에서 공식 파이프라인 결과와 비교해 세포 위치 99.5%, 세포 타입 99.7% 일치를 확인했습니다.
- 모델 가중치 라이선스: CC BY-NC 4.0 (비상업적 연구용).

## 예제 데이터 (노트북이 자동 다운로드)

| 데이터 | 출처 | 용도 |
|---|---|---|
| `CMU-1-Small-Region.svs` (피부, 20x) | OpenSlide 테스트 데이터 (`cellvit-download-examples`) | 01 패치 실습, 02 WSI 추론 |
| NuCLS 점 라벨 (유방암, train 10장 / test 5장, CC0) | CellViT++ GitHub `test_database/training_database/Example-Detection` | 02 Part 4 분류기 학습 |
| 사전학습 분류기 7종 | Zenodo `classifier.zip` | 02 Part 1~3 |

## 로컬 실행 (이 서버)

이 폴더에 `.venv`(Python 3.10, torch 2.2.2 cu121, cellvit 1.0.9)가 있고, Jupyter 커널 `Python (cellvit)`로 등록되어 있습니다.
모델과 데이터는 이미 `models/`, `test_database/`, `cellvitpp_repo/`에 받아 두었습니다.

기존 Jupyter나 VS Code에서 노트북을 열고 커널을 `Python (cellvit)`로 선택하면 됩니다.

## 검증

- 로컬(RTX 3090 Ti, py3.10/torch 2.2.2)과 Colab 유사 환경(py3.12/torch 2.12)에서 두 노트북 모두 오류 없이 끝까지 실행했습니다.
- 실제 Colab(T4, 2코어)에서는 실행해 보지 않았습니다. CLI는 2코어 설정으로 로컬에서 정상 동작을 확인했습니다.

## 노트북 수정

노트북은 `build_notebooks.py`로 생성합니다. 셀 내용을 고친 뒤 `.venv/bin/python build_notebooks.py`를 실행하세요.

## 라이선스 주의

CellViT 코드와 가중치는 Apache 2.0 + **Commons Clause**입니다(상업적 이용은 저자 허가 필요).
HIPT 기반 가중치는 Mahmood Lab 조건도 따릅니다.
