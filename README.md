# CellViT / CellViT++ / HNE2Cell 튜토리얼 (Colab용)

| 노트북 | 내용 |
|---|---|
| `01_CellViT_tasks.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/01_CellViT_tasks.ipynb) | CellViT의 5가지 task: 조직 분류, 세포핵 검출, 인스턴스 분할, 세포 분류(PanNuke 5종), 세포 임베딩. 마지막에 QuPath용 GeoJSON 내보내기 |
| `02_CellViT_plusplus.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/02_CellViT_plusplus.ipynb) | CellViT++: 사전학습 분류기 7종, 같은 세포에 분류기 바꿔 끼우기, WSI 전체 추론(CLI), **점 라벨로 나만의 분류기 학습** 후 WSI에 적용 |
| `03_HNE2Cell.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fourmodern/cellvit-tutorial/blob/main/03_HNE2Cell.ipynb) | HNE2Cell: 공간전사체 라벨로 학습한 15종 세포 분류. 색 정규화, 패치/영역 추론, 계통 그룹, 종양 침윤·TLS 후보 공간 분석, CSV/GeoJSON 내보내기 |

## Colab에서 실행

1. 위 표의 **Open in Colab** 배지를 누릅니다.
2. `런타임 → 런타임 유형 변경 → T4 GPU`를 선택합니다.
3. 첫 셀을 실행하면 패키지 설치 후 런타임이 **자동으로 재시작**됩니다. 재시작 후 처음부터 다시 실행하세요.
4. 모델(SAM-H 약 2.8GB)과 예제 데이터(약 600MB)는 노트북이 Zenodo에서 자동으로 받습니다.
   Zenodo 속도에 따라 10~30분 걸릴 수 있습니다. `USE_DRIVE_CACHE = True`로 두면 Google Drive에 캐시됩니다.

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
