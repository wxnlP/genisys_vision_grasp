from tabnanny import verbose
from turtle import width

import aimrt_py as aimrt
from protocols.joint_state_pb2 import JointStateArray
from protocols.pose_pb2 import PoseStamped
import yaml

from drivers.http_rpc import RpcClient
from drivers.orbbec import OrbbecCamera
from drivers.utils.snapshot import save_color_snapshot, snapshot_requested
from ultralytics import YOLO

import time
import datetime
import numpy as np
import cv2


class VisionGraspModule(aimrt.ModuleBase):
    def __init__(self):
        super().__init__()
        # AimRT 对象
        self.core = aimrt.CoreRef()
        self.logger = aimrt.LoggerRef()
        self.work_executor = aimrt.ExecutorRef()

        self.camera = None
        self.rpc = None
        self.model = None

        self.width = 1280
        self.height = 800

        self.running = False
        self.loop_exit = False

    def Info(self) -> aimrt.ModuleInfo:
        info = aimrt.ModuleInfo()
        info.name = "VisionGraspModule"
        return info

    def Initialize(self, core: aimrt.CoreRef) -> bool:
        self.core = core
        self.logger = self.core.GetLogger()

        # yaml 初始化
        file_path = self.core.GetConfigurator().GetConfigFilePath()
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            work_executor_name = str(config["work_executor_name"])
            yolo_model = str(config["yolo_model"])
        
        # 创建任务执行器
        self.work_executor = self.core.GetExecutorManager().GetExecutor(work_executor_name)

        # 初始化 Orbbec 相机
        self.width = int(config["camera"]["width"])
        self.height = int(config["camera"]["height"])
        fps = int(config["camera"]["fps"])
        self.camera = OrbbecCamera(self.width, self.height, fps)

        # rpc 初始化
        self.rpc = RpcClient()

        # 初始化 YOLO 模型
        self.model = YOLO(yolo_model, task= "segment", verbose=False)

        return True

    def Start(self) -> bool:
        try:
            self.camera.open()
            self.running = True
            self.work_executor.ExecuteAfter(datetime.timedelta(milliseconds=500), self.main_loop)
        except Exception as e:
            aimrt.error(self.logger, f"VisionGraspModule Start failed: {e}")
            return False
        aimrt.info(self.logger, "VisionGraspModule started.")
        return True

    def Shutdown(self):
        try:
            self.running = False
            while not self.loop_exit:
                time.sleep(0.1)
            aimrt.info(self.logger, "VisionGraspModule shutdown.")
        except Exception as e:
            aimrt.error(self.logger, f"VisionGraspModule shutdown failed: {e}")
    
    def main_loop(self):
        error_count  = 0
        try:
            cv2.namedWindow("YOLO11-seg", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO11-seg", self.width, self.height)
            while self.running:
                color, depth = self.camera.get_frame()
                if color is None or depth is None:
                    error_count += 1
                    if error_count > 10:
                        aimrt.error(self.logger, "Failed to get frame from camera.")
                    continue
                else:
                    error_count = 0
                results = self.model(color, verbose=False)
                cv2.imshow("YOLO11-seg", results[0].plot())
                # if snapshot_requested():
                #     path = save_color_snapshot(color)
                #     if path:
                #         aimrt.info(self.logger, f"snapshot saved: {path}")
                #     else:
                #         aimrt.error(self.logger, "snapshot save failed")
                cv2.imshow("color", color)
                # cv2.imshow("depth", depth)
                cv2.waitKey(1)
            self.loop_exit = True
            aimrt.info(self.logger, "VisionGraspModule main loop exited.")
        except Exception as e:
            self.loop_exit = True
            aimrt.error(self.logger, f"VisionGraspModule main loop error: {e}")
        finally:
            cv2.destroyAllWindows()
            self.camera.close()

