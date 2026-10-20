"""HTTP API（M5：会话与规则改用 fastrun 引擎，DTO 保持不变）。"""

import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.api.ai_service import AIService
from src.api.fast_service import SessionManager
from src.api.models import (ActionRequest, GameStartRequest, GameStateModel)

app = FastAPI(title="Poker Game API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = SessionManager()
ai_service = AIService()


@app.get("/models", response_model=List[str])
def list_models():
    return ai_service.list_models()


@app.post("/game/start", response_model=GameStateModel)
def start_game(request: GameStartRequest = None):
    if request and request.model_name:
        ai_service.load_model(request.model_name)
    session_id = session_manager.create_session()
    return session_manager.get_session(session_id).to_dto()


@app.get("/game/{game_id}/state", response_model=GameStateModel)
def get_state(game_id: str):
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.to_dto()


@app.post("/game/{game_id}/action", response_model=GameStateModel)
def player_action(game_id: str, request: ActionRequest):
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
    try:
        session.apply_human_cards(request.card_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return session.to_dto()


@app.post("/game/{game_id}/ai", response_model=GameStateModel)
def trigger_ai(game_id: str):
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
    if session.state.over:
        return session.to_dto()

    move, thinking = ai_service.predict(session.state, return_thinking=True)
    session.apply_ai_move(move)
    return session.to_dto(thinking=thinking)


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
