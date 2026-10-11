import sys
import cv2
from pathlib import Path
from detection.yolo_detector import YOLODetector

image_path = sys.argv[1]
input_path = Path(image_path)
output_file = Path("results/figures") / f"{input_path.stem}_detected.png"

img = cv2.imread(image_path)

detector = YOLODetector()

results = detector.detect(img)

r = results[0]

for d in results:
    label = d["label"]
    confidence = d["confidence"]
    bbox = d["bbox"]

    print( f"{label:10s} "
           f"{confidence:.3f} "
           f"{bbox}" )

annotated = detector.model(img)[0].plot()
cv2.imwrite(str(output_file), annotated)
print(f"Saved visualization to {output_file}")
