#!/usr/bin/env bash
# 构建 protocols/ros2/genisys_msgs（msg + srv）到 protocols/ros2/install，
# 再用 aimrt_py-gen-ros2-rpc 为每个 .srv 生成 AimRT Python RPC 桩到 protocols/ros2/rpc_stubs。
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$HERE/ros2"
PKG_NAME=genisys_msgs
SRV_DIR="$WORKSPACE/$PKG_NAME/srv"
STUB_DIR="$WORKSPACE/rpc_stubs"
ROS_PREFIX="${ROS2_PREFIX:-/opt/ros/${ROS_DISTRO:-humble}}"
# rosidl_adapter 用系统 Python
BUILD_PYTHON="${BUILD_PYTHON:-/usr/bin/python3}"
# AimRT Python RPC 桩生成器、
RPC_GEN="${AIMRT_ROS2_RPC_GEN:-$(command -v aimrt_py-gen-ros2-rpc || true)}"

if [ ! -x "$BUILD_PYTHON" ]; then
  echo "未找到构建用 Python：$BUILD_PYTHON" >&2
  exit 1
fi

if [ ! -f "$ROS_PREFIX/setup.bash" ]; then
  echo "未找到 ROS2：$ROS_PREFIX/setup.bash" >&2
  echo "请设置 ROS2_PREFIX（或 ROS_DISTRO）后重试。" >&2
  exit 1
fi

if [ -z "$RPC_GEN" ] || [ ! -x "$RPC_GEN" ]; then
  echo "未找到 aimrt_py-gen-ros2-rpc，请先 pip install aimrt_py（或设置 AIMRT_ROS2_RPC_GEN）。" >&2
  exit 1
fi

source "$ROS_PREFIX/setup.bash"

if ! command -v colcon >/dev/null 2>&1; then
  echo "未找到 colcon。" >&2
  exit 1
fi

cd "$WORKSPACE"

colcon build \
  --packages-select "$PKG_NAME" \
  --merge-install \
  --install-base install \
  --build-base build \
  --cmake-clean-cache \
  --cmake-args \
    -DPython3_EXECUTABLE="$BUILD_PYTHON" \
    -DPYTHON_EXECUTABLE="$BUILD_PYTHON"

# ---------------------------------------------------------------------------
# Ros2 srv -> AimRT Python RPC 桩
#   aimrt_py-gen-ros2-rpc --pkg_name=<pkg> --srv_file=<x.srv> --output_path=<dir>
# 输出 <ServiceName>_aimrt_rpc_ros2.py，内含 <ServiceName>Service / <ServiceName>ServiceProxy。
# ---------------------------------------------------------------------------
echo
echo "生成 AimRT Python RPC 桩 ..."
mkdir -p "$STUB_DIR"
# 先清掉旧桩，避免删除/重命名 srv 后残留
rm -f "$STUB_DIR"/*_aimrt_rpc_ros2.py

stub_count=0
for srv in "$SRV_DIR"/*.srv; do
  [ -e "$srv" ] || continue
  "$RPC_GEN" \
    --pkg_name="$PKG_NAME" \
    --srv_file="$srv" \
    --output_path="$STUB_DIR"
  stub_count=$((stub_count + 1))
done

touch "$STUB_DIR/__init__.py"

echo
echo "已构建：$WORKSPACE/install"
echo "已生成 RPC 桩（$stub_count 个）：$STUB_DIR"
echo "source $WORKSPACE/install/local_setup.bash"
