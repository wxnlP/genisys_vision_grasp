from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_FOLDER = Path(ROOT / "test_datasets")

model = YOLO(f"{ROOT}/../cfg/models/object-block-11n-seg.engine")

images = []
for img in TEST_FOLDER.rglob("*.jpg"):
    images.append(str(img))

results = model(images)

for result in results:
    boxes = result.boxes            # 用于边界框输出的 Boxes 对象
    print(f"Boxes: {boxes}")
    masks = result.masks            # 用于分割掩码输出的 Masks 对象
    # print(f"Masks: {masks}")
    result.show()                   # 显示到屏幕上
