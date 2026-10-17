from drivers.orbbec import OrbbecCamera

import torch
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys


ROOT_DIR = Path(__file__).resolve().parent
PATH_GRASPNET = ROOT_DIR / "../graspnet_baseline"
PATH_YOLO_SEG = ROOT_DIR / "../cfg/models"

sys.path.append(str(PATH_GRASPNET / "models"))
sys.path.append(str(PATH_GRASPNET / "utils"))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')

config = parser.parse_args()

gnet = None
yolo_seg = None
camera = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init():
    global gnet, yolo_seg, camera
    # 1.初始化模型
    gnet = GraspNet(num_view=config.num_view, is_training=False)
    gnet.to(device)

    yolo_seg = YOLO(PATH_YOLO_SEG / "object-block-11n-seg.engine")
    camera = OrbbecCamera()
    camera.open()

    # 2.加载 checkpoint
    checkpoint = torch.load(PATH_GRASPNET / "checkpoint/checkpoint-rs.tar")
    gnet.load_state_dict(checkpoint['model_state_dict'])
    ## TODO: Only Print
    start_epoch = checkpoint['epoch']
    print(f"-> loaded checkpoint (epoch: {start_epoch})")
    # 把模型切换到评估（推理）模式
    gnet.eval()

def process_data():
    intr = camera.get_intrinsic()
    color_bgr, depth_mm = camera.get_frame()
    while color_bgr is None or depth_mm is None:
        color_bgr, depth_mm = camera.get_frame()

    # 物体并集掩码
    mask = None
    results = yolo_seg(color_bgr, verbose=False)
    cv2.imwrite("result.jpg", results[0].plot())
    if results[0].masks is not None:
        keep = results[0].boxes.conf.cpu().numpy() > 0.5
        masks_keep = results[0].masks.data.cpu().numpy()[keep]
        if len(masks_keep):
            mask = masks_keep.astype(bool).any(axis=0)
            if mask.shape != depth_mm.shape:             # 兜底：对齐到 depth
                mask = cv2.resize(mask.astype(np.uint8),
                                    (depth_mm.shape[1], depth_mm.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

    # 点云处理
    H, W = depth_mm.shape
    camera_info = CameraInfo(W, H, intr[0][0], intr[1][1], intr[0][2], intr[1][2], 1000.0)
    cloud = create_point_cloud_from_depth_image(depth_mm, camera_info, organized=True)

    depth_ok = depth_mm > 0
    cloud_ori  = cloud[depth_ok].astype(np.float32)

    # get valid points
    valid = depth_ok
    if mask is not None:
        valid = valid & mask.astype(bool)
    cloud_masked  = cloud[valid].astype(np.float32)
    color_masked = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)[valid].astype(np.float32)

    if len(cloud_masked) >= config.num_point:
        idxs = np.random.choice(len(cloud_masked), config.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), config.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked)
    cloud.colors = o3d.utility.Vector3dVector(color_masked)

    end_points = {
        'point_clouds': torch.from_numpy(cloud_sampled[np.newaxis]).to(device),  # (1, N, 3)
        'cloud_colors': color_sampled,
    }

    return end_points, cloud, cloud_ori

def get_grasps(end_points):
    # Forward pass
    with torch.no_grad():
        end_points = gnet(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg
    
def collision_detection(gg, cloud):
    # 碰撞检测
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=config.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=config.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    # 去掉位姿相近的重复抓取
    gg = gg.nms()
    # 按分数从高到低排
    gg.sort_by_score()
    # 只留分数最高的 50 个
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

if __name__ == "__main__":
    init()
    end_points, cloud, cloud_ori = process_data()
    gg = get_grasps(end_points)
    if config.collision_thresh > 0:
        collision_detection(gg, cloud_ori)
    vis_grasps(gg, cloud)

    