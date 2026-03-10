import uuid
from typing import Dict, Optional
from src.core.game import Game

class GameSession:
    """
    单个游戏会话
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.game = Game()
        # 默认 Player 0 是人类，1, 2 是 AI
        self.human_player_idx = 0 
        self.ai_player_indices = [1, 2]

class SessionManager:
    """
    会话管理器
    """
    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}

    def create_session(self) -> str:
        """创建一个新游戏，返回 session_id"""
        session_id = str(uuid.uuid4())
        session = GameSession(session_id)
        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[GameSession]:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
