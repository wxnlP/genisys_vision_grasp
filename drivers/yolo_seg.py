import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


CLASS_SEQ = { 0: "cylinder", 1: "cuboid" }

# 三种目标颜色在 OpenCV HSV 空间中的取值范围：H 为 0-180，S、V 为 0-255。
COLOR_HSV_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "green": [((35, 70, 50), (85, 255, 255))],
    "blue": [((90, 40, 90), (130, 255, 255))],
}

# 某种颜色的像素在检测区域内所占的最小比例；低于该比例视为颜色无法判定。
MIN_COLOR_RATIO = 0.3


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

    def get_boxes_xyxy(self):
        return self.result.boxes.xyxy.cpu().numpy()

    def select_target(self, target_class: str, target_color: str = None):
        if self.result is None: 
            return None
        if target_class not in self.class_seq.values():
            return None
        if target_color is not None and target_color not in COLOR_HSV_RANGES:
            return None

        idx = next((k for k, v in self.class_seq.items() if v == target_class), None)

        # 指定颜色指标时，先把当前帧转换为 HSV，供下面逐个候选区域统计颜色。
        hsv = None
        if target_color is not None:
            if self.color is None:
                return None
            hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)

        cand = None
        for i in range(len(self.result.boxes.cls)):
            if int(self.result.boxes.cls[i]) != idx:
                continue
            # 颜色指标：掩码内主色与 target_color 不一致的检测框不参与候选。
            if target_color is not None and self._mask_color(hsv, i) != target_color:
                continue
            if cand is not None:
                if self.result.boxes.conf[cand] > self.result.boxes.conf[i]:
                    continue
            cand = i
        return cand

    def _mask_color(self, hsv, i):
        """统计第 i 个检测掩码内各颜色的像素占比，返回占比最高且不低于 MIN_COLOR_RATIO 的颜色名。"""
        h, w = self.color.shape[:2]

        # 分割掩码的分辨率与网络输入一致，最近邻放大到原图尺寸后与原图逐像素对应。
        mask = (self.result.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        if not mask.any():
            return None

        # 取出掩码内所有像素，形状整理为 inRange 需要的 (N, 1, 3)。
        pixels = hsv[mask].reshape(-1, 1, 3)

        best_color = None
        best_ratio = 0.0
        for name, ranges in COLOR_HSV_RANGES.items():
            count = 0
            for lo, hi in ranges:
                # 每段 HSV 范围统计一次命中像素数，red 的两段相加。
                count += int(cv2.countNonZero(
                    cv2.inRange(pixels, np.array(lo, np.uint8), np.array(hi, np.uint8))
                ))
            ratio = count / len(pixels)
            if ratio > best_ratio:
                best_color = name
                best_ratio = ratio

        return best_color if best_ratio >= MIN_COLOR_RATIO else None

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
        
