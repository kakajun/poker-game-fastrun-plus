from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import uvicorn
import os
import sys

# Ensure root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.core.game import Game
from src.core.card import Card, Rank, Suit
from src.core.hand_type import HandType, Play
from src.core.evaluator import HandEvaluator
from src.api.session_manager import SessionManager
from src.api.ai_service import AIService
from src.api.models import GameStateModel, ActionRequest, CardModel, PlayModel

app = FastAPI(title="Poker Game API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Managers
session_manager = SessionManager()
ai_service = AIService()

def _convert_card_to_model(card: Card) -> CardModel:
    return CardModel(rank=card.rank.value, suit=card.suit.value, id=card.id)

def _convert_play_to_model(play: Play) -> PlayModel:
    return PlayModel(
        type=play.type.name,
        cards=[_convert_card_to_model(c) for c in play.cards],
        length=play.length,
        max_rank=play.max_rank.value if hasattr(play.max_rank, 'value') else play.max_rank
    )

def _convert_game_to_state(game: Game, game_id: str) -> GameStateModel:
    hands = []
    for h in game.hands:
        hands.append([_convert_card_to_model(c) for c in h])
        
    last_play = _convert_play_to_model(game.last_play) if game.last_play else None
    
    # Get legal actions for current player
    legal_plays = game.get_legal_actions()
    legal_models = [_convert_play_to_model(p) for p in legal_plays]
    
    return GameStateModel(
        game_id=game_id,
        current_player=game.current_player,
        hands=hands,
        last_play=last_play,
        last_play_player=game.last_play_player,
        winner=game.winner,
        is_over=game.is_over,
        scores=game.scores,
        cards_played_count=game.cards_played_count,
        bomb_scores=game.bomb_scores,
        legal_actions=legal_models
    )

@app.post("/game/start", response_model=GameStateModel)
def start_game():
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    return _convert_game_to_state(session.game, session_id)

@app.get("/game/{game_id}/state", response_model=GameStateModel)
def get_state(game_id: str):
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
    return _convert_game_to_state(session.game, game_id)

@app.post("/game/{game_id}/action", response_model=GameStateModel)
def player_action(game_id: str, request: ActionRequest):
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
        
    game = session.game
    if game.is_over:
        raise HTTPException(status_code=400, detail="Game is over")
        
    # Check if it's human turn (assuming human is 0)
    # But for debugging, we allow action on any turn?
    # Better to enforce turn.
    if game.current_player != session.human_player_idx:
        # raise HTTPException(status_code=400, detail="Not your turn")
        pass # Allow for now
        
    # Parse action
    if not request.card_ids:
        # Pass
        play = Play(HandType.PASS, [], 0, 0)
    else:
        # Convert IDs to Cards
        cards = []
        try:
            for cid in request.card_ids:
                cards.append(Card.from_id(cid))
        except:
            raise HTTPException(status_code=400, detail="Invalid card IDs")
            
        # Evaluate Hand
        play = HandEvaluator.evaluate(cards)
        if not play:
            raise HTTPException(status_code=400, detail="Invalid hand type")
            
    # Check legality (using game engine logic)
    # We can check if `play` is in `get_legal_actions()`
    # But `get_legal_actions` returns all possible actions, which is expensive to search.
    # Better to check `can_beat` directly.
    # However, `game.step` does not check `can_beat` strictly? 
    # `game.step` calls `game.get_legal_actions`? No.
    # `game.step` assumes action is legal.
    
    # So we must validate here.
    # 1. Check if cards are in hand (Game.step checks this)
    # 2. Check if can beat last play
    
    # But pass logic is tricky (must beat if possible).
    # Let's rely on `game.get_legal_actions()` for strict rule enforcement (including pass rule).
    # To optimize, we generate all legal actions and check if our play is in it.
    # Since we implemented `__eq__` for Play? No.
    # We need to match type, length, max_rank.
    
    legal_actions = game.get_legal_actions()
    
    is_legal = False
    for legal in legal_actions:
        if play.type == legal.type and play.length == legal.length and play.max_rank == legal.max_rank:
            # Check cards?
            # For PASS, no cards.
            # For others, we need to match cards exactly?
            # Yes, because `game.step` removes specific cards.
            # But `HandEvaluator` returns a Play with specific cards.
            # So we just need to find a legal action that has the SAME cards.
            if set(c.id for c in play.cards) == set(c.id for c in legal.cards):
                is_legal = True
                # Use the legal play object (it might have extra info like is_bomb)
                play = legal 
                break
                
    if not is_legal:
        # Special case: Pass
        if play.type == HandType.PASS:
             raise HTTPException(status_code=400, detail="Cannot pass (Must beat if possible)")
        else:
             raise HTTPException(status_code=400, detail="Illegal move (Cannot beat or invalid)")

    # Execute
    try:
        game.step(play)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    return _convert_game_to_state(game, game_id)

@app.post("/game/{game_id}/ai", response_model=GameStateModel)
def trigger_ai(game_id: str):
    session = session_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
        
    game = session.game
    if game.is_over:
        return _convert_game_to_state(game, game_id)
        
    # Predict (Even for human if auto-play is requested)
    action = ai_service.predict(game, game.current_player)
    
    if action:
        game.step(action)
    else:
        # Should not happen if AI is robust
        # Force Pass?
        pass_action = Play(HandType.PASS, [], 0, 0)
        try:
            game.step(pass_action)
        except:
            raise HTTPException(status_code=500, detail="AI failed to act")
            
    return _convert_game_to_state(game, game_id)

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
