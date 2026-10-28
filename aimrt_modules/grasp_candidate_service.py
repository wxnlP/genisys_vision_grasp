import aimrt_py

from protocols.pb._gen.graspnet_candidate_pb2 import GraspCandidateResponse
from protocols.pb._gen.graspnet_candidate_aimrt_rpc_pb2 import GraspCandidateService

# 业务错误码
ERR_OK = 0
ERR_NO_TARGET = 1
ERR_NO_CANDIDATE = 2
ERR_INFERENCE_FAILED = 3


class GraspServiceError(Exception):
    """可预期的推理失败，携带业务错误码和给客户端看的原因。"""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class GraspCandidateServiceImpl(GraspCandidateService):
    """GraspCandidateService 服务端实现。

    provider 签名:
        provider(target_class: str, target_color: str, max_candidates: int)
            -> list[dict]

    每个 dict 必须包含:
        score: float
        width: float
        position: (x, y, z)            末端坐标系，单位 m
        orientation: (w, x, y, z)      末端坐标系四元数

    失败时 provider 抛 GraspServiceError；其他异常统一按推理失败返回。
    """

    def __init__(self, provider, logger):
        super().__init__()
        self._provider = provider
        self._logger = logger

    def GetGraspCandidates(self, ctx_ref, req):
        rsp = GraspCandidateResponse()

        aimrt_py.info(
            self._logger,
            f"\n---\nGraspCandidateService request: \nclass={req.target_class!r} \n"
            f"color={req.target_color!r} \nmax_candidates={req.max_candidates}\n---",
        )

        try:
            candidates = self._provider(
                req.target_class, req.target_color, int(req.max_candidates)
            )
        except GraspServiceError as e:
            rsp.error.code = e.code
            rsp.error.msg = e.message
            aimrt_py.warn(self._logger, f"GraspCandidateService: {e.code} {e.message}")
            return aimrt_py.RpcStatus(), rsp
        except Exception as e:
            rsp.error.code = ERR_INFERENCE_FAILED
            rsp.error.msg = f"grasp inference failed: {e}"
            if self._logger is not None:
                aimrt_py.error(self._logger, f"GraspCandidateService failed: {e}")
            return aimrt_py.RpcStatus(), rsp

        for rank, c in enumerate(candidates, start=1):
            item = rsp.candidates.add()
            item.rank = rank
            item.score = float(c["score"])
            item.width = float(c["width"])

            x, y, z = c["position"]
            item.flange_grasp_pose.position.x = float(x)
            item.flange_grasp_pose.position.y = float(y)
            item.flange_grasp_pose.position.z = float(z)

            w, qx, qy, qz = c["orientation"]
            item.flange_grasp_pose.orientation.w = float(w)
            item.flange_grasp_pose.orientation.x = float(qx)
            item.flange_grasp_pose.orientation.y = float(qy)
            item.flange_grasp_pose.orientation.z = float(qz)

        aimrt_py.info(
            self._logger,
            f"GraspCandidateService response: {len(rsp.candidates)} candidates",
        )

        return aimrt_py.RpcStatus(), rsp
