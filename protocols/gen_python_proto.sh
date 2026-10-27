#!/usr/bin/env bash
# 从 protocols/pb/{aimrt,genisys,vision_grasp}/*.proto 生成 Python 模块到 protocols/pb/_gen：
#   *_pb2.py / *_pb2.pyi   消息类型（--python_out / --pyi_out）
#   *_aimrt_rpc_pb2.py     服务桩（--aimrt_rpc_out，需 aimrt_py 提供的 protoc 插件）
set -euo pipefail

cd "$(dirname "$0")/.."

PROTO_ROOT=protocols/pb
GEN_DIR="$PROTO_ROOT/_gen"

if ! command -v protoc >/dev/null 2>&1; then
    echo "未找到 protoc，请先安装 protobuf 编译器。" >&2
    exit 1
fi

mkdir -p "$GEN_DIR"

# aimrt_py 安装的 python RPC 插件；找不到时只生成 pb2，并给出提示
AIMRT_RPC_PLUGIN="$(command -v protoc_plugin_py_gen_aimrt_py_rpc || true)"

AIMRT_RPC_ARGS=()
if [ -n "$AIMRT_RPC_PLUGIN" ]; then
    AIMRT_RPC_ARGS=(
        --aimrt_rpc_out="$GEN_DIR"
        --plugin=protoc-gen-aimrt_rpc="$AIMRT_RPC_PLUGIN"
    )
else
    echo "未找到 aimrt 插件 protoc_plugin_py_gen_aimrt_py_rpc" >&2
fi

protoc -I "$PROTO_ROOT/aimrt" \
       -I "$PROTO_ROOT/genisys" \
       -I "$PROTO_ROOT/vision_grasp" \
       --python_out="$GEN_DIR" \
       --pyi_out="$GEN_DIR" \
       "${AIMRT_RPC_ARGS[@]}" \
       "$PROTO_ROOT"/aimrt/*.proto \
       "$PROTO_ROOT"/genisys/*.proto \
       "$PROTO_ROOT"/vision_grasp/*.proto

# 把跨文件的绝对 import 改成相对 import，便于按包导入
sed -i -E 's/^import ([A-Za-z0-9_]+_pb2) as ([A-Za-z0-9_]+)$/from . import \1 as \2/' \
    "$GEN_DIR"/*_pb2.py "$GEN_DIR"/*_pb2.pyi

# RPC 桩里的依赖是裸 import（不带 as），单独处理
sed -i -E 's/^import ([A-Za-z0-9_]+_pb2)$/from . import \1/' \
    "$GEN_DIR"/*_aimrt_rpc_pb2.py

touch "$GEN_DIR/__init__.py"

echo "已生成到 $GEN_DIR"
