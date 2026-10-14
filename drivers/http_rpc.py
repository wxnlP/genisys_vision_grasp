import requests
import math


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """intrinsic ZYX 欧拉角（弧度）→ Hamilton 四元数 (w, x, y, z)。

    对应 R = Rz(yaw) · Ry(pitch) · Rx(roll)，与 rotation_matrix_to_euler_zyx 互逆。
    """
    r, p, y = roll / 2.0, pitch / 2.0, yaw / 2.0
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )

# ---------------- HTTP RPC 封装 ----------------

class RpcClient:
    """AimRT http+json RPC 客户端（MoveJ / StartRecord / StopRecord）。"""

    def __init__(self, base_url: str = "http://127.0.0.1:50080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_error = ""

    def _post(self, service: str, method: str, payload: dict) -> bool:
        """成功返回 True；失败返回 False，原因写入 self.last_error。"""
        url = f"{self.base_url}/rpc/{service}/{method}"
        self.last_error = ""
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            status_info = result.get("error") or result.get("common_rsp") or result
            code = status_info.get("code", -1)
            if int(code) == 0:
                return True
            self.last_error = f"code={code} {status_info.get('msg', '')}".strip()
        except requests.exceptions.ConnectionError:
            self.last_error = f"cannot connect to {self.base_url}"
        except requests.exceptions.Timeout:
            self.last_error = f"timeout after {self.timeout}s"
        except requests.exceptions.RequestException as e:
            self.last_error = str(e)
        return False


    # --- 机械臂运动（MoveJ 为阻塞语义：任务完成才返回） ---
    def movej(self, q: list, v: float, a: float):
        """MoveJ 到关节角 q（弧度），阻塞至到位后返回响应。"""
        return self._post(
            "genisys_msgs/srv", "MoveJ",
            {"position": {"position": [float(x) for x in q]},
             "v": float(v), "a": float(a)})

    # --- 机械臂运动（MoveJPose 为阻塞语义：任务完成才返回） ---
    def movej_pose(self, x: float, y: float, z: float,
                   roll: float, pitch: float, yaw: float,
                   v: float = 0.8, a: float = 0.4):
        """MoveJPose 到指定位姿（阻塞，返回 True / None）。

        rpy 为 intrinsic ZYX 欧拉角，单位弧度。
        v/a 是关节空间的速度与加速度，单位 rad/s 与 rad/s²。
        """
        w, qx, qy, qz = rpy_to_quat(roll, pitch, yaw)
        return self._post(
            "genisys_msgs/srv", "MoveJPose",
            {
                "pose": {"position": {"x": float(x), "y": float(y), "z": float(z)},
                         "orientation": {"x": qx, "y": qy, "z": qz, "w": w}},
                "v": float(v),
                "a": float(a)
            })

    # --- 机械臂运动（MoveJMulti 为阻塞语义：任务完成才返回） ---
    def movej_multi(self, q_list: list,
                    r: float = 0.1, v: float = 0.8, a: float = 0.4):
        """MoveJMulti 依次经过多个关节角路径点（阻塞，返回 True / None）。

        q_list 为关节角路径点序列 [[q1, ..., q6], ...]，单位弧度，至少 2 个点。
        r 为路径点之间的过渡半径，单位弧度。
        v/a 是关节空间的速度与加速度，单位 rad/s 与 rad/s²。
        """
        return self._post(
            "genisys_msgs/srv", "MoveJMulti",
            {
                "position": [{"position": [float(x) for x in q]} for q in q_list],
                "r": float(r),
                "v": float(v),
                "a": float(a)
            })

    # --- 机械臂运动（MoveJPoseMulti 为阻塞语义：任务完成才返回） ---
    def movej_pose_multi(self, poses: list,
                         r: float = 0.1, v: float = 0.8, a: float = 0.4):
        """MoveJPoseMulti 依次经过多个末端位姿路径点（阻塞，返回 True / None）。

        poses 为位姿路径点序列 [(x, y, z, roll, pitch, yaw), ...]，
        rpy 为 intrinsic ZYX 欧拉角，单位弧度，至少 2 个点。
        r 为路径点之间的过渡半径，单位弧度。
        v/a 是关节空间的速度与加速度，单位 rad/s 与 rad/s²。
        """
        pose_list = []
        for x, y, z, roll, pitch, yaw in poses:
            w, qx, qy, qz = rpy_to_quat(roll, pitch, yaw)
            pose_list.append(
                {"position": {"x": float(x), "y": float(y), "z": float(z)},
                 "orientation": {"x": qx, "y": qy, "z": qz, "w": w}})
        return self._post(
            "genisys_msgs/srv", "MoveJPoseMulti",
            {
                "pose": pose_list,
                "r": float(r),
                "v": float(v),
                "a": float(a)
            })

    # --- 机械臂运动（MoveL 为阻塞语义：任务完成才返回） ---
    def movel(self, x: float, y: float, z: float,
              roll: float, pitch: float, yaw: float,
              v: float = 0.3, a: float = 0.2):
        """MoveL 沿直线运动到指定位姿（阻塞，返回 True / None）。

        rpy 为 intrinsic ZYX 欧拉角，单位弧度。
        v/a 是末端笛卡尔速度与加速度，单位 m/s 与 m/s²。
        """
        w, qx, qy, qz = rpy_to_quat(roll, pitch, yaw)
        return self._post(
            "genisys_msgs/srv", "MoveL",
            {
                "pose": {"position": {"x": float(x), "y": float(y), "z": float(z)},
                         "orientation": {"x": qx, "y": qy, "z": qz, "w": w}},
                "v": float(v),
                "a": float(a)
            })

    # --- 机械臂运动（MoveLMulti 为阻塞语义：任务完成才返回） ---
    def movel_multi(self, poses: list,
                    r: float = 0.05, v: float = 0.3, a: float = 0.2):
        """MoveLMulti 依次经过多个末端位姿路径点（阻塞，返回 True / None）。

        poses 为位姿路径点序列 [(x, y, z, roll, pitch, yaw), ...]，
        rpy 为 intrinsic ZYX 欧拉角，单位弧度，至少 1 个点。
        r 为路径点之间的过渡半径，单位米。
        v/a 是末端笛卡尔速度与加速度，单位 m/s 与 m/s²。
        """
        pose_list = []
        for x, y, z, roll, pitch, yaw in poses:
            w, qx, qy, qz = rpy_to_quat(roll, pitch, yaw)
            pose_list.append(
                {"position": {"x": float(x), "y": float(y), "z": float(z)},
                 "orientation": {"x": qx, "y": qy, "z": qz, "w": w}})
        return self._post(
            "genisys_msgs/srv", "MoveLMulti",
            {
                "pose": pose_list,
                "r": float(r),
                "v": float(v),
                "a": float(a)
            })

     # --- 夹爪 ---
    def gripper_pos(self, position: float = 1.0, velocity: float = 1.0,
                    acceleration: float = 2.0, effort: float = 0.0):
        """GripperPos 位置控制（阻塞，返回 True / None）。

        position 为开合度，0.0 = 夹紧，1.0 = 全开。
        velocity / acceleration 为运动速度与加速度。
        effort 为力矩（电流）上限，0 表示不限制。
        """
        return self._post(
            "genisys_msgs/srv", "GripperPos",
            {
                "position": float(position),
                "velocity": float(velocity),
                "acceleration": float(acceleration),
                "effort": float(effort)
            })

    def gripper_force(self, target_torque: float = 0.5,
                      switch_velocity: float = 0.2,
                      approach_velocity: float = 1.0,
                      approach_accel: float = 1.0,
                      closed_release: bool = True):
        """GripperForce 力控抓取：接近 → 力矩保持，力到位后返回（力矩指令继续保持）。

        target_torque 为夹持力矩，单位 N·m。
        switch_velocity 为速度亏欠阈值，单位 rad/s（实际速度落后规划速度该值即判定碰到）。
        approach_velocity / approach_accel 为接近段的速度与加速度，单位 rad/s 与 rad/s²。
        closed_release 为位置接近闭合时是否释放力矩。

        返回 True 后不需要再发保持指令，也不要重复发送同一条力控指令。
        """
        return self._post(
            "genisys_msgs/srv", "GripperForce",
            {
                "target_torque": float(target_torque),
                "switch_velocity": float(switch_velocity),
                "approach_velocity": float(approach_velocity),
                "approach_accel": float(approach_accel),
                "closed_release": bool(closed_release)
            })

    # --- record 插件 ---
    def start_record(self, action_name: str):
        return self._post(
            "aimrt.protocols.record_playback_plugin.RecordPlaybackService",
            "StartRecord",
            {"action_name": action_name,
             "preparation_duration_s": 0, "record_duration_s": 0})

    def stop_record(self, action_name: str):
        return self._post(
            "aimrt.protocols.record_playback_plugin.RecordPlaybackService",
            "StopRecord",
            {"action_name": action_name})