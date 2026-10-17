from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


CLASS_SEQ = { 0: "cylinder", 1: "cuboid" }


class YOLOSeg:
    def __init__(self, model, class_seq: dict =CLASS_SEQ, task="segment", verbose=False):
        self.class_seq = class_seq
        self.model = YOLO(model, task=task, verbose=verbose)

        self.result = None
        self.color = None

    def predict(self, color, verbose=False):
        self.color = color
        self.result = self.model(color, verbose=verbose)[0]

        return self.result

    def select_target(self, target_class: str):
        if self.result is None: 
            return None
        if target_class not in self.class_seq.values():
            return None

        idx = next((k for k, v in self.class_seq.items() if v == target_class), None)

        cand = None
        for i in range(len(self.result.boxes.cls)):
            if int(self.result.boxes.cls[i]) != idx:
                continue
            if cand is not None:
                if self.result.boxes.conf[cand] > self.result.boxes.conf[i]:
                    continue
            cand = i
        return cand

    def draw_detections(self, target_idx=None):
        if self.color is None:
            return None
        display = self.color.copy()
        if self.result is None:
            return display

        ann = Annotator(display)
        base_lw = ann.lw

        for i in range(len(self.result.boxes.cls)):
            is_selected = i == target_idx
            class_id = int(self.result.boxes.cls[i])

            # 目标：绿色 + 双倍线宽；其他：按类别取调色板
            box_color = (68, 243, 0) if is_selected else colors(class_id, bgr=True)
            ann.lw = base_lw * 2 if is_selected else base_lw

            label = f"{self.class_seq[class_id]} {float(self.result.boxes.conf[i]):.2f}"
            if is_selected:
                label = f"TARGET {label}"

            x1, y1, x2, y2 = [int(v) for v in self.result.boxes.xyxy[i]]
            ann.box_label((x1, y1, x2, y2), label, color=box_color)

        return ann.result()
        