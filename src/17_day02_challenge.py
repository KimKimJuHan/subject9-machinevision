from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------
image_path = Path("samples/day02/challenge.jpg")
output_dir = Path("outputs/day02/challenge")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# 2. 전처리: CLAHE (지역 대비 강조) + Sharpening
# --------------------------------------------------
# 타일별로 명암 대비를 크게 끌어올려 뚜껑 상단 경계를 강조
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)

# 경계선 강조를 위한 샤프닝 커널
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(clahe_img, -1, kernel)

result = sharpened.copy()

# --------------------------------------------------
# 3. 결과 저장
# --------------------------------------------------
cv2.imwrite(str(output_dir / "challenge_result.jpg"), result)

# 전처리 전 Otsu
_, original_otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 전처리 후 Otsu
_, result_otsu = cv2.threshold(
    result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    str(output_dir / "challenge_original_otsu.jpg"), original_otsu
)
cv2.imwrite(str(output_dir / "challenge_result_otsu.jpg"), result_otsu)

print("전처리 개선 시도 완료")