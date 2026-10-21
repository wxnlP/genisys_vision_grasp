import aimrt_py as aimrt
from protocols.joint_state_pb2 import JointStateArray
from protocols.pose_pb2 import PoseStamped
import yaml

from drivers.http_rpc import RpcClient
from drivers.orbbec import OrbbecCamera
from drivers.utils.snapshot import save_color_snapshot, snapshot_requested
from drivers.gnet import Gnet, Open3DGraspWindow
from drivers.yolo_seg import YOLOSeg

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
        self.yolo_seg = None
        self.gnet = None

        self.width = 848
        self.height = 530

        self.snapshot_events = []
        self.target_class = "cuboid"
        self.target_color = "blue"
        self.gnet_checkpoint = ""
        self.target_margin_px = 12
        self.target_expand_ratio = 1.35
        self.max_grasp_width_m = 0.09
        self.roi_expand_ratio = 1.35
        self.roi_margin_px = 20

        # Open3D 可视化窗口
        self.grasp_vis = None
        self.grasp_vis_broken = False

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

        self.snapshot_events = list(config["snapshot_event"])
        self.gnet_checkpoint = str(config["graspnet"]["gnet_checkpoint"])
        self.target_margin_px = int(config["graspnet"]["target_margin_px"])
        self.target_expand_ratio = float(config["graspnet"]["target_expand_ratio"])
        self.max_grasp_width_m = float(config["graspnet"]["max_grasp_width_m"])
        self.roi_expand_ratio = float(config["graspnet"]["roi_expand_ratio"])
        self.roi_margin_px = int(config["graspnet"]["roi_margin_px"])
        
        # 创建任务执行器
        self.work_executor = self.core.GetExecutorManager().GetExecutor(work_executor_name)

        # 初始化 Orbbec 相机
        self.width = int(config["camera"]["width"])
        self.height = int(config["camera"]["height"])
        fps = int(config["camera"]["fps"])
        hand_eye_path = str(config["hand_eye_path"])
        self.camera = OrbbecCamera(self.width, self.height, fps, hand_eye_path=hand_eye_path)

        # rpc 初始化
        self.rpc = RpcClient()

        # 初始化 YOLO 模型
        self.yolo_seg = YOLOSeg(yolo_model)

        return True

    def Start(self) -> bool:
        try:
            self.running = True
            self.camera.open()
            if "graspnet" in self.snapshot_events:
                self.gnet = Gnet(self.gnet_checkpoint, self.camera.get_intrinsic())
                aimrt.info(self.logger, f"GraspNet loaded: {self.gnet_checkpoint}")

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
            cv2.namedWindow("YOLO-seg", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO-seg", self.width, self.height)
            while self.running:
                # 相机帧获取
                color, depth = self.camera.get_frame()
                if color is None or depth is None:
                    error_count += 1
                    if error_count > 10:
                        aimrt.error(self.logger, "Failed to get frame from camera.")
                    continue
                else:
                    error_count = 0
                # YOLO 推理
                result = self.yolo_seg.predict(color)
                idx = self.yolo_seg.select_target(self.target_class, self.target_color)
                display = self.yolo_seg.draw_detections(idx)

                cv2.imshow("YOLO-seg", display)
                # 处理 Enter Event
                if snapshot_requested():
                    for event in self.snapshot_events:
                        if event == "screenshot":
                            aimrt.info(self.logger, "screenshot requested")
                            self.do_screenshot(display)
                        elif event == "graspnet":
                            aimrt.info(self.logger, "graspnet requested")
                            self.do_grasp(color, depth, result.boxes.xyxy, idx)
                        else:
                            aimrt.warn(self.logger, f"unknown snapshot_event: {event}")
                cv2.waitKey(1)
                if self.grasp_vis is not None and not self.grasp_vis.poll():
                    self.grasp_vis.close()
                    self.grasp_vis = None
            self.loop_exit = True
            aimrt.info(self.logger, "VisionGraspModule main loop exited.")
        except Exception as e:
            self.loop_exit = True
            aimrt.error(self.logger, f"VisionGraspModule main loop error: {e}")
        finally:
            if self.grasp_vis is not None:
                self.grasp_vis.close()
                self.grasp_vis = None
            cv2.destroyAllWindows()
            self.camera.close()

    def do_screenshot(self, display):
        path = save_color_snapshot(display, prefix="display")
        if path:
            aimrt.info(self.logger, f"screenshot saved: {path}")
        else:
            aimrt.error(self.logger, "screenshot save failed")

    def do_grasp(self, color, depth, box_xyxy, target_idx):
        """用当前帧跑一次 GraspNet，打印最终选出的那一个候选。"""
        if self.gnet is None:
            aimrt.error(self.logger, "grasp event disabled: GraspNet not loaded")
            return

        bbox = tuple(int(v) for v in box_xyxy[target_idx])

        boxes = box_xyxy.cpu().numpy()
        roi = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())

        try:
            self.gnet.build_end_points(color, depth, roi=roi,
                                       roi_expand_ratio=self.roi_expand_ratio,
                                       roi_margin_px=self.roi_margin_px)
            self.gnet.infer_grasp()
            self.gnet.filter_grasps_by_bbox(bbox, depth.shape, self.target_margin_px, self.target_expand_ratio)
            self.gnet.filter_grasps_by_width(self.max_grasp_width_m)
            best = self.gnet.select_best_grasp()
        except Exception as e:
            aimrt.error(self.logger, f"grasp inference failed: {e}")
            return

        self.update_grasp_vis()

        c = self.gnet.counts
        aimrt.info(self.logger,
                   "\n---\ngrasp stats: \n"
                   f"  depth_ok={c.get('depth_ok', 0)} \n"
                   f"  input={c.get('input', 0)} \n"
                   f"  decoded={c.get('decoded', 0)} \n"
                   f"  after_collision={c.get('after_collision', 0)} \n"
                   f"  after_bbox={c.get('after_bbox', 0)} \n"
                   f"  after_width={c.get('after_width', 0)} \n"
                   f"  final={c.get('final', 0)}\n---")

        if best is None:
            aimrt.warn(self.logger, "no valid grasp candidate")
            return

        aimrt.info(self.logger, 
                   f"\n---\ngrasp candidate: \n"
                   f"  score={best.score:.4f} \n"
                   f"  width={best.width:.4f} \n"
                   f"  translation = {[round(float(v), 4) for v in best.translation]} \n"
                   f"  rotation    = {np.round(best.rotation_matrix, 4).tolist()} \n---")

        # 构造齐次变换矩阵（相机 to 抓取点）
        T_cam_grasp = np.eye(4, dtype=np.float64)
        T_cam_grasp[:3, :3] = np.asarray(best.rotation_matrix, dtype=np.float64)
        T_cam_grasp[:3, 3] = np.asarray(best.translation, dtype=np.float64).reshape(3)

        xyz, quat = self.camera.transform_grasp_by_hand_eye(T_cam_grasp)

        aimrt.info(self.logger,
                   f"\n---\ngrasp in end-effector frame (hand-eye): \n"
                   f"  xyz        = {[round(float(v), 6) for v in xyz]}  (m) \n"
                   f"  quaternion = {[round(float(v), 6) for v in quat]}  (w, x, y, z) \n---")

    def update_grasp_vis(self):
        """刷新 Open3D 窗口；创建失败（无显示等）只禁用可视化，不影响主流程。"""
        if self.grasp_vis_broken:
            return
        if self.grasp_vis is None:
            try:
                self.grasp_vis = Open3DGraspWindow("GraspNet Grasps", top_k=50)
            except Exception as e:
                self.grasp_vis_broken = True
                aimrt.error(self.logger, f"Open3D window unavailable, visualization disabled: {e}")
                return
        self.grasp_vis.update(self.gnet.o3d_cloud, self.gnet.gg)

