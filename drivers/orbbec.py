import os
import numpy as np
import cv2
from pathlib import Path
from pyorbbecsdk import (
                Pipeline, Config,
                OBSensorType, OBFormat, OBAlignMode,
                Context, OBPropertyID,
            )


import numpy as np

def _rotation_matrix_to_quat_wxyz(R):
    """旋转矩阵 -> Hamilton 四元数 (w, x, y, z)。"""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])  # s = 4 * x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])  # s = 4 * y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])  # s = 4 * z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    
    if q[0] < 0.0:
        q = -q
        
    return q


class OrbbecCamera:
    def __init__(self, width=848, height=530, fps=15, hand_eye_path=None):
        self._w = width
        self._h = height
        self._fps = fps
        self._depth_scale_mm = 0.1 # gemini305

        self._pipeline = None

        # 手眼标定：T_result = T_ee_cam（相机系 -> 末端系）
        self._hand_eye_T = None
        if hand_eye_path is not None:
            with np.load(hand_eye_path, allow_pickle=False) as data:
                T = np.asarray(data["T_result"], dtype=np.float64)
            self._hand_eye_T = T

    def transform_grasp_by_hand_eye(self, T_cam_grasp):
        """输入 GraspNet 的 grasp，输出末端系 xyz + 四元数 (w, x, y, z)。

        变换关系：T_ee_grasp = T_ee_cam @ T_cam_grasp。
        """
        if self._hand_eye_T is None:
            raise RuntimeError("hand-eye calibration is not loaded")

        T_ee_grasp = self._hand_eye_T @ T_cam_grasp
        return T_ee_grasp[:3, 3].copy(), _rotation_matrix_to_quat_wxyz(T_ee_grasp[:3, :3])

    def open(self):
        self._pipeline = Pipeline()
        cfg = Config()

        plist = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            cp = plist.get_video_stream_profile(self._w, self._h, OBFormat.MJPG, self._fps)
        except Exception:
            cp = plist.get_default_video_stream_profile()
            print("[OrbbecCamera] Select default color stream profile, please make sure it's currently USB 3.0")
        cfg.enable_stream(cp)

        dplist = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            dp = dplist.get_video_stream_profile(self._w, self._h, OBFormat.Y16, self._fps)
        except Exception:
            dp = dplist.get_default_video_stream_profile()
            print("[OrbbecCamera] Select default depth stream profile, please make sure it's currently USB 3.0")
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
        color_bgr = None
        if fmt == OBFormat.MJPG:
            color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        elif fmt == OBFormat.YUYV:
            color_bgr = cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)

        if color_bgr is None:
            print("[OrbbecCamera] Unsupported color format: {fmt}")
            return None, None

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

    def get_intrinsic(self):
        if self._K is not None:
            return self._K