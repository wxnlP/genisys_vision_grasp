from pathlib import Path
import sys
import torch
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
PATH_GRASPNET = ROOT_DIR / "../graspnet_baseline"
sys.path.append(str(PATH_GRASPNET / "utils"))
sys.path.insert(0, str(PATH_GRASPNET / "models"))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup
import open3d as o3d


class Gnet:
    def __init__(self, checkpoint, K, max_mm=1000, min_mm=50, num_view=300, num_point=20000, 
                 voxel_size=0.01, collision_thresh=0.01):
        self.device = torch.device("cuda:0")
        self.max_mm = max_mm
        self.min_mm = min_mm
        self.num_point = num_point
        self.voxel_size = voxel_size
        self.collision_thresh = collision_thresh
        self.K = K

        self.gnet = GraspNet(num_view=num_view, is_training=False)
        self.gnet.to(self.device)
        self.gnet.load_state_dict(torch.load(checkpoint, map_location=self.device)['model_state_dict'])
        self.gnet.eval()

        self.end_points = None
        self.o3d_cloud = None
        self.raw_cloud = None
        self.gg = None

        self.counts = {}

    def warmup(self):
        """用随机点跑一次前向与解码，提前完成 CUDA 初始化。"""
        points = torch.rand((1, self.num_point, 3), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.gnet({"point_clouds": points})
            pred_decode(out)
        torch.cuda.synchronize()

    def build_end_points(self, color_bgr, depth_mm, roi=None, roi_expand_ratio=1.35, roi_margin_px=20,
                         build_o3d=True):
        depth_mask = (depth_mm > self.min_mm) & (depth_mm < self.max_mm)
        if int(depth_mask.sum()) == 0:
            raise RuntimeError("No valid depth_mm pixels in the configured depth range.")
        # color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = depth_mm.shape
        camera = CameraInfo(w, h, float(self.K[0, 0]), float(self.K[1, 1]), float(self.K[0, 2]), float(self.K[1, 2]), 1000.0)
        cloud = create_point_cloud_from_depth_image(depth_mm, camera, organized=True)

        mask = depth_mask
        if roi is not None:
            x1, y1, x2, y2 = [float(v) for v in roi]
            expand_ratio = max(1.0, float(roi_expand_ratio))
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
            half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
            rx1 = int(np.clip(round(cx - half_w) - roi_margin_px, 0, w - 1))
            rx2 = int(np.clip(round(cx + half_w) + roi_margin_px, 0, w - 1))
            ry1 = int(np.clip(round(cy - half_h) - roi_margin_px, 0, h - 1))
            ry2 = int(np.clip(round(cy + half_h) + roi_margin_px, 0, h - 1))
            roi_mask = np.zeros((h, w), bool)
            roi_mask[ry1:ry2 + 1, rx1:rx2 + 1] = True
            mask = mask & roi_mask

        cloud_masked = cloud[mask]
        color_masked_bgr = color_bgr[mask]

        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs = np.concatenate(
                [
                    np.arange(len(cloud_masked)),
                    np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True),
                ],
                axis=0,
            )

        sampled_cloud = cloud_masked[idxs]
        sampled_bgr = color_masked_bgr[idxs]
        sampled_color = cv2.cvtColor(
            sampled_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB
        ).reshape(-1, 3).astype(np.float32) / 255.0

        end_points = {
            "point_clouds": torch.from_numpy(sampled_cloud.astype(np.float32)[np.newaxis]).cuda(non_blocking=True),
            "cloud_colors": sampled_color,
        }

        self.end_points = end_points
        if build_o3d:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(sampled_cloud.astype(np.float32))
            o3d_cloud.colors = o3d.utility.Vector3dVector(sampled_color.astype(np.float32))
            self.o3d_cloud = o3d_cloud
        else:
            self.o3d_cloud = None
        self.raw_cloud = cloud[depth_mask]

        self.counts = {"depth_ok": int(depth_mask.sum()), "input": int(mask.sum())}

    def decode_grasp(self):
        """前向推理并解码抓取，得到未过滤的候选。"""
        with torch.no_grad():
            self.end_points = self.gnet(self.end_points)
            grasp_preds = pred_decode(self.end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

        self.counts["decoded"] = len(gg)
        self.gg = gg
        return self.gg

    def detect_collision(self):
        """对当前候选做碰撞检测。"""
        if len(self.gg) == 0 or self.collision_thresh <= 0:
            self.counts["after_collision"] = len(self.gg)
            return self.gg

        detector = ModelFreeCollisionDetector(self.raw_cloud, voxel_size=self.voxel_size)
        collision_mask = detector.detect(self.gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        self.gg = self.gg[~collision_mask]
        self.counts["after_collision"] = len(self.gg)
        return self.gg

    def filter_grasps_by_bbox(self, bbox_xyxy, image_shape, margin_px, expand_ratio):
        if len(self.gg) == 0:
            self.counts["after_bbox"] = 0
            return self.gg
        # 抓取中心反投影到像素
        translations = np.asarray(self.gg.translations, dtype=np.float64)
        z = translations[:, 2]
        valid_z = z > 1e-6
        u = np.full(len(self.gg), np.nan, dtype=np.float64)
        v = np.full(len(self.gg), np.nan, dtype=np.float64)
        u[valid_z] = float(self.K[0, 0]) * translations[valid_z, 0] / z[valid_z] + float(self.K[0, 2])
        v[valid_z] = float(self.K[1, 1]) * translations[valid_z, 1] / z[valid_z] + float(self.K[1, 2])

        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        expand_ratio = max(1.0, float(expand_ratio))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
        half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
        x1 = int(round(cx - half_w)) - int(margin_px)
        y1 = int(round(cy - half_h)) - int(margin_px)
        x2 = int(round(cx + half_w)) + int(margin_px)
        y2 = int(round(cy + half_h)) + int(margin_px)

        if image_shape is not None:
            h, w = image_shape
            x1 = int(np.clip(x1, 0, max(0, w - 1)))
            x2 = int(np.clip(x2, 0, max(0, w - 1)))
            y1 = int(np.clip(y1, 0, max(0, h - 1)))
            y2 = int(np.clip(y2, 0, max(0, h - 1)))

        keep = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        self.gg = self.gg[keep]
        self.counts["after_bbox"] = len(self.gg)
        return self.gg

    def filter_grasps_by_width(self, max_width_m):
        if max_width_m is None or len(self.gg) == 0:
            self.counts["after_width"] = len(self.gg)
            return self.gg
        self.gg = self.gg[np.asarray(self.gg.widths, dtype=np.float64) <= float(max_width_m)]
        self.counts["after_width"] = len(self.gg)
        return self.gg

    def sort_grasps(self):
        """NMS 去重后按分数从高到低排序。"""
        if len(self.gg) == 0:
            self.counts["final"] = 0
            return self.gg
        self.gg = self.gg.nms()
        self.gg.sort_by_score()
        self.counts["final"] = len(self.gg)
        return self.gg

    def get_gg_length(self):
        return len(self.gg)

    def draw_best_grasp_projection(self, display, grasp):
        if grasp is None:
            return
        x, y, z = [float(v) for v in grasp.translation]
        if z <= 1e-6:
            return
        u = int(round(float(self.K[0, 0]) * x / z + float(self.K[0, 2])))
        v = int(round(float(self.K[1, 1]) * y / z + float(self.K[1, 2])))
        if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
            cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
            cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    
class Open3DGraspWindow:
    def __init__(self, title: str, top_k: int, width: int, height: int) -> None:
        self._top_k = top_k
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=width, height=height):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries = []
        self._initialized = False

    def update(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup) -> None:
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        cloud_vis = o3d.geometry.PointCloud(cloud)
        cloud_vis.transform(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64))
        geometries = [cloud_vis]

        if len(grasps) > 0:
            grasps_vis = GraspGroup(grasps.grasp_group_array.copy())
            try:
                grasps_vis = grasps_vis.nms()
            except Exception as exc:
                print(f"Grasp NMS skipped: {exc}")
            grasps_vis.sort_by_score()
            grasps_vis = grasps_vis[: self._top_k]
            grasps_vis.transform(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64))
            geometries.extend(grasps_vis.to_open3d_geometry_list())

        for geom in geometries:
            self._vis.add_geometry(geom, reset_bounding_box=not self._initialized)
        self._geometries = geometries
        self._initialized = True
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()