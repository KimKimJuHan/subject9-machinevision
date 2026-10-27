"""
Day 03 — Morphology & Contour 기반 OK/NG 판정
================================================
목표:
  - Day 02 Otsu 이진화 결과의 잔존 문제 해결
    1) 뚜껑 내부 홀(Holes) 제거 → Closing
    2) 배경 노이즈 제거          → Opening
  - findContours로 외곽선 추출
  - 면적/위치/종횡비 기반 OK/NG 판정

입력: samples/day02/challenge.jpg
출력: outputs/day03/
"""

import cv2
import numpy as np
import os

# ── 경로 설정 ──────────────────────────────────────────────
INPUT_PATH  = "samples/day02/challenge.jpg"
OUTPUT_DIR  = "outputs/day03"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── OK/NG 판정 기준 (실제 환경에 맞게 튜닝 필요) ──────────
MIN_AREA       = 50_000   # 최소 뚜껑 면적 (픽셀²)
MAX_AREA       = 500_000  # 최대 뚜껑 면적 (픽셀²)
MIN_ASPECT     = 0.6      # 최소 종횡비 (w/h)
MAX_ASPECT     = 1.8      # 최대 종횡비 (w/h)


def preprocess_to_binary(gray: np.ndarray) -> np.ndarray:
    """Day 02 챌린지 최종 전처리 파이프라인 재사용."""
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Otsu 이진화
    _, binary = cv2.threshold(sharpened, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_morphology(binary: np.ndarray) -> np.ndarray:
    """
    Opening → Closing 순서로 적용.
    - Opening (Erosion → Dilation): 작은 노이즈 점 제거
    - Closing (Dilation → Erosion): 내부 홀 메우기
    """
    # Opening: 배경 노이즈 제거
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Closing: 뚜껑 내부 홀 메우기
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return closed


def find_cap_contour(morphed: np.ndarray):
    """가장 큰 Contour = 뚜껑 본체로 추정."""
    contours, _ = cv2.findContours(
        morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0
    # 면적 기준 최대 Contour 선택
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    return largest, area


def judge_ok_ng(contour, area: float) -> tuple[str, dict]:
    """
    OK/NG 판정 로직.
    Returns:
        verdict (str): "OK" or "NG"
        info    (dict): 판정에 사용된 측정값
    """
    if contour is None or area == 0:
        return "NG", {"reason": "뚜껑 Contour 미검출"}

    # Bounding Box로 종횡비 계산
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0

    info = {
        "area":         int(area),
        "bounding_box": (x, y, w, h),
        "aspect_ratio": round(aspect_ratio, 3),
    }

    # 판정
    if not (MIN_AREA <= area <= MAX_AREA):
        info["reason"] = f"면적 이상: {int(area)} (기준 {MIN_AREA}~{MAX_AREA})"
        return "NG", info

    if not (MIN_ASPECT <= aspect_ratio <= MAX_ASPECT):
        info["reason"] = f"종횡비 이상: {aspect_ratio:.3f} (기준 {MIN_ASPECT}~{MAX_ASPECT})"
        return "NG", info

    info["reason"] = "정상"
    return "OK", info


def draw_result(original: np.ndarray, contour, verdict: str, info: dict) -> np.ndarray:
    """결과 시각화 — Contour + 판정 텍스트 오버레이."""
    result = original.copy()
    if len(result.shape) == 2:  # grayscale → BGR 변환
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    color = (0, 255, 0) if verdict == "OK" else (0, 0, 255)  # 초록 / 빨강

    if contour is not None:
        cv2.drawContours(result, [contour], -1, color, 2)
        x, y, w, h = info["bounding_box"]
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)

    # 판정 텍스트
    label = f"{verdict} | area={info['area']} | AR={info.get('aspect_ratio', 'N/A')}"
    cv2.putText(result, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(result, info.get("reason", ""), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result


def main():
    # 1. 이미지 로드
    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {INPUT_PATH}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 전처리 → 이진화 (Day 02 파이프라인 재사용)
    binary = preprocess_to_binary(gray)

    # 3. Morphology (Opening → Closing)
    morphed = apply_morphology(binary)

    # 4. Contour 추출
    contour, area = find_cap_contour(morphed)

    # 5. OK/NG 판정
    verdict, info = judge_ok_ng(contour, area)

    # 6. 결과 출력
    print("=" * 40)
    print(f"  판정 결과: {verdict}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 40)

    # 7. 결과 이미지 저장
    result_img = draw_result(gray, contour, verdict, info)

    cv2.imwrite(f"{OUTPUT_DIR}/01_binary.jpg", binary)
    cv2.imwrite(f"{OUTPUT_DIR}/02_morphed.jpg", morphed)
    cv2.imwrite(f"{OUTPUT_DIR}/03_result_{verdict}.jpg", result_img)

    # 8. 비교 화면 출력 (선택)
    compare = np.hstack([
        cv2.cvtColor(binary,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR),
        result_img
    ])
    cv2.imshow("Binary | Morphed | Result", compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"\n결과 저장 완료: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
