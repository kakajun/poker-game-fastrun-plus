import os
import sys
import numpy as np
from typing import Optional

# Ensure root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sb3_contrib import MaskablePPO
from src.core.game import Game
from src.core.hand_type import Play
from src.env.action_space import ActionSpace
from src.env.obs_encoder import ObsEncoder

class AIService:
    """
    AI 服务封装，用于加载模型并进行推理。
    """
    def __init__(self, model_path: str = "models/ppo_poker_final"):
        self.model_path = model_path
        self.model: Optional[MaskablePPO] = None
        self.action_space_manager = ActionSpace()
        self.obs_encoder = ObsEncoder()
        
        # Load model on init
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path + ".zip"):
            print(f"Loading model from {self.model_path}...")
            # MaskablePPO.load requires env=None for inference only if env is not used for normalization
            # We don't use VecNormalize, so env=None is fine.
            self.model = MaskablePPO.load(self.model_path)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model not found at {self.model_path}. AI will not work.")

    def predict(self, game: Game, player_idx: int) -> Optional[Play]:
        """
        根据当前游戏状态，预测最佳出牌动作。
        """
        if self.model is None:
            return None
            
        # 1. Encode Observation
        obs = self.obs_encoder.encode(game, player_idx)
        
        # 2. Get Action Mask
        # We need to manually compute mask here as we don't have the Gym Env wrapper
        legal_plays = game.get_legal_actions()
        mask = np.zeros(self.action_space_manager.size, dtype=bool) # Boolean mask for sb3
        
        valid_actions = []
        for play in legal_plays:
            aid = self.action_space_manager.get_id(play)
            if aid != -1 and 0 <= aid < self.action_space_manager.size:
                mask[aid] = True
                valid_actions.append((aid, play))
            elif play.type.name == "PASS": # Handle Pass special case if ID is 0
                mask[0] = True
                valid_actions.append((0, play))
                
        # 3. Predict
        # action is int (or array of int)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        
        if isinstance(action, np.ndarray):
            action = action.item()
            
        # 4. Convert back to Concrete Play
        # We need to map abstract action ID back to concrete Play
        # Since we have the list of valid concrete plays, we can just find the matching one.
        
        # abstract_play = self.action_space_manager.get_action(action)
        # But we need concrete cards.
        
        # Search in valid_actions for matching ID
        candidates = [p for aid, p in valid_actions if aid == action]
        
        if candidates:
            # If multiple (e.g. same type different suits), pick first
            return candidates[0]
            
        return None
