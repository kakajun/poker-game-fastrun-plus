"""兼容层：会话管理已迁到 fastrun 引擎，见 `fast_service.py`。"""

from .fast_service import FastSession, SessionManager

__all__ = ["FastSession", "SessionManager", "GameSession"]


# 旧名字保留，避免历史脚本 import 失败
GameSession = FastSession
