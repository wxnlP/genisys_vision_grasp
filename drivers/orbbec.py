import os
import numpy as np
import cv2
from pathlib import Path
from pyorbbecsdk import (
                Pipeline, Config,
                OBSensorType, OBFormat, OBAlignMode,
                Context, OBPropertyID,
            )


class OrbbecCamera:
    def __init__(self, width=848, height=530, fps=30):
        self._w = width
        self._h = height
        self._fps = fps
        self._depth_scale_mm = 0.1 # gemini305

        self._pipeline = None

    def open(self):
        self._pipeline = Pipeline()
        cfg = Config()

        plist = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            cp = plist.get_video_stream_profile(self._w, self._h, OBFormat.RGB, self._fps)
        except Exception:
            cp = plist.get_default_video_stream_profile()
        cfg.enable_stream(cp)

        dplist = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            dp = dplist.get_video_stream_profile(self._w, self._h, OBFormat.Y16, self._fps)
        except Exception:
            dp = dplist.get_default_video_stream_profile()
        cfg.enable_stream(dp)

        cfg.set_align_mode(OBAlignMode.HW_MODE)

        # 防止有些stream profile不支持HW_MODE
        try:
            self._pipeline.start(cfg)
        except Exception:
            print("[OrbbecCamera] HW depth-color alignment unavailable; using SW mode")
            self._pipeline.stop()
            cfg.set_align_mode(OBAlignMode.SW_MODE)
            self._pipeline.start(cfg)

        try:
            self._pipeline.get_device().set_int_property(
                OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT, 75
            )
        except Exception as e:
            print(f"[OrbbecCamera] set color sharpness failed: {e}")

        # Intrinsics（内参）
        intr = cp.get_intrinsic()
        self._K = np.array([
            [intr.fx, 0,       intr.cx],
            [0,       intr.fy, intr.cy],
            [0,       0,       1      ],
        ], dtype=np.float64)

        # Distortion（畸变）
        self._D = cp.get_distortion()

    def close(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def get_frame(self):
        if self._pipeline is None:
            print("[OrbbecCamera] Pipeline is not started. Call open() first.")
            return None, None

        frames = self._pipeline.wait_for_frames(500)
        if frames is None:
            print("[OrbbecCamera] No frames received.")
            return None, None

        # 彩色帧
        c_frame = frames.get_color_frame()
        if c_frame is None:
            return None, None
        w, h = c_frame.get_width(), c_frame.get_height()
        raw = np.asanyarray(c_frame.get_data(), dtype=np.uint8)
        fmt = c_frame.get_format()
        if fmt == OBFormat.MJPG:
            color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        elif fmt == OBFormat.RGB:
            color_bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)

        # 深度帧
        d_frame = frames.get_depth_frame()
        if d_frame is None:
            return None, None
        w, h = d_frame.get_width(), d_frame.get_height()
        depth = np.frombuffer(d_frame.get_data(), dtype=np.uint16).reshape(h, w)
        depth_scale = self._depth_scale_mm
        try:
            depth_scale = float(d_frame.get_depth_scale())
        except Exception:
            pass

        # 针对非1.0的depth_scale进行处理
        depth_mm = np.clip(
            np.rint(depth.astype(np.float32) * depth_scale),
            np.iinfo(np.uint16).min,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

        return color_bgr, depth_mm
