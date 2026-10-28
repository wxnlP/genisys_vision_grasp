"""GraspCandidateService 的 HTTP 客户端 demo。

服务端用 AimRT net_plugin 的 http RPC 后端（见 cfg 里的 net_plugin / rpc.http），
请求格式就是 AimRT 约定的：

    POST http://{ip}:{port}/rpc/{func_name}
    content-type: application/json

本脚本只用标准库，可以在没装 aimrt_py 的机器（例如 RK3588）上直接跑。

用法：
    python grasp_candidate_client.py \
        --url http://192.168.12.10:18080/rpc/vision_graspnet.graspnet_candidate.GraspCandidateService/GetGraspCandidates \
        --target_class cuboid --target_color blue --max_candidates 32
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = (
    "http://127.0.0.1:18080/rpc/"
    "vision_graspnet.graspnet_candidate.GraspCandidateService/GetGraspCandidates"
)


def post_json(url, payload, timeout_s):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.status, resp.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="GraspCandidateService HTTP client.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="AimRT http rpc url")
    parser.add_argument("--target_class", type=str, default="cuboid")
    parser.add_argument("--target_color", type=str, default="blue")
    parser.add_argument("--max_candidates", type=int, default=32)
    parser.add_argument("--timeout_s", type=float, default=10.0)
    args = parser.parse_args()

    payload = {
        "targetClass": args.target_class,
        "targetColor": args.target_color,
        "maxCandidates": max(0, args.max_candidates),
    }

    try:
        status, body = post_json(args.url, payload, args.timeout_s)
    except urllib.error.HTTPError as e:
        print(f"http error: {e.code} {e.reason}", file=sys.stderr)
        print(e.read().decode("utf-8", "replace"), file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"connect failed: {e.reason}", file=sys.stderr)
        return 1

    try:
        rsp = json.loads(body)
    except json.JSONDecodeError:
        print(f"invalid json response (HTTP {status}):", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1

    error = rsp.get("error") or {}
    code = int(error.get("code", 0))
    msg = error.get("msg", "")
    candidates = rsp.get("candidates") or []

    print(f"http_status: {status}")
    print(f"error: code={code} msg={msg!r}")
    print(f"candidates: {len(candidates)}")
    print(json.dumps(rsp, indent=2, ensure_ascii=False))
    return 0 if code == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
