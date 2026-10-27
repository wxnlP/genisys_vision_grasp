# protocols

protobuf 源在 `pb/`，ROS2 IDL 在 `ros2/genisys_msgs/`；生成物分别落到 `pb/_gen/` 和 `ros2/rpc_stubs/`。

## 生成

```bash
./protocols/gen_python_proto.sh   # *.proto -> pb/_gen（消息 + AimRT RPC 桩）
./protocols/build_ros2_msgs.sh    # ROS2 msg/srv -> ros2/install，并为每个 srv 生成 rpc_stubs
```

## 导入

### protobuf

```python
from protocols.pb._gen.pose_pb2 import Pose, PoseStamped
from protocols.pb._gen.graspnet_candidate_aimrt_rpc_pb2 import (
    GraspCandidateService,        # 服务端基类
    GraspCandidateServiceProxy,   # 客户端代理
)
```

### ROS2

先 source ROS2 和工作区：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source protocols/ros2/install/local_setup.bash
```

msg：

```python
from genisys_msgs.msg import TaskContext
```

RPC 桩（每个 srv 一个文件 `<Srv>_aimrt_rpc_ros2.py`）：

```python
from protocols.ros2.rpc_stubs.MoveJ_aimrt_rpc_ros2 import MoveJService, MoveJServiceProxy
```
