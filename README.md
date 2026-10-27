# Subject 9 — Machine Vision 실습 시리즈

> 컨베이어 벨트 이물질 검출을 위한 Machine Vision 기초부터 Edge AI 배포까지

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 제조 라인 병뚜껑 외관 검사 자동화 (OK/NG 판정) |
| **환경** | Edge AI — NVIDIA Jetson Orin Nano |
| **스택** | Python, OpenCV, YOLO, FastAPI, PostgreSQL, Docker |

---

## 📁 폴더 구조

```
subject9_machinevision/
├── samples/              # 실습용 원본 이미지/영상
│   └── day02/            # Day 02 전처리 실습 샘플
├── src/                  # 실습 스크립트
│   ├── 00_check_env.py
│   ├── ...
│   └── 17_day02_challenge.py
├── outputs/              # 처리 결과 이미지 (git 제외)
├── reports/              # 실험 보고서
│   ├── day02_parameter_table.csv
│   └── day02_preprocess_reason.md
└── requirements.txt
```

---

## 📅 Day별 실습 내용

### ✅ Day 01 — 영상 입력 및 ROI 추출
- OpenCV 기본 이미지/영상 입력 (`cv2.imread`, `cv2.VideoCapture`)
- ROI(Region of Interest) 수동 크롭 및 배치 처리
- 웹캠 / 스마트폰 카메라 실시간 입력

**핵심 파일:** `src/01_read_image.py` ~ `src/08_phone_camera_roi.py`

---

### ✅ Day 02 — Image Enhancement 및 이진화

**목표:** 저대비/어두운 검사 영상을 전처리하여 Otsu 이진화 품질 향상

| 스크립트 | 내용 |
|----------|------|
| `09_color_space.py` | BGR / Gray / HSV 색공간 비교 |
| `10_brightness_contrast.py` | 선형 밝기/대비 조절 (α, β) |
| `11_gamma.py` | 비선형 감마 보정 |
| `12_clahe.py` | 국소 히스토그램 균일화 (CLAHE) |
| `13_filter_compare.py` | Gaussian / Median / Bilateral 비교 |
| `14_sharpen.py` | 경계선 샤프닝 |
| `15_threshold_otsu.py` | 고정 임계값 vs Otsu 자동 이진화 |
| `16_preprocess_compare.py` | 전처리별 Otsu 이진화 종합 비교 |
| `17_day02_challenge.py` | **미니 챌린지 — 최종 해결 코드** |

**챌린지 핵심 결과:**

```
문제: 흰색 병뚜껑 vs 회색 배경 → 저대비로 단순 Otsu 실패
1차 시도 (실패): Gamma 보정 → 그림자로 인해 뚜껑 상단 경계 소실
2차 시도 (성공): CLAHE(clipLimit=3.0) + Sharpening(3x3) + Otsu
결과: 뚜껑 외곽선 및 나사산 홈 완벽 분리
```

> 상세 실험 기록: [`reports/day02_preprocess_reason.md`](reports/day02_preprocess_reason.md)
> 파라미터 테이블: [`reports/day02_parameter_table.csv`](reports/day02_parameter_table.csv)

---

### 🔲 Day 03 — Morphology & Contour 기반 OK/NG 판정 (예정)

```
이진화 결과
  → Morphology(Opening → Closing)으로 노이즈 제거 및 홀 메우기
  → findContours로 외곽선 추출
  → 면적/위치/종횡비 기반 OK/NG 자동 판정
```

---

## ⚙️ 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧠 핵심 사고방식

> "전처리는 목적이 아니라 수단이다.
> 항상 '이 전처리가 이진화/검출 품질을 높이는가?'를 기준으로 선택한다."
