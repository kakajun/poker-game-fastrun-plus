import os
import sys
import numpy as np
from typing import Optional, List
from sb3_contrib import MaskablePPO
from src.core.game import Game
from src.core.hand_type import Play, HandType
from src.env.action_space import ActionSpace
from src.env.obs_encoder import ObsEncoder
from src.agent.heuristic_agent import HeuristicAgent

class AIService:
    """
    AI 服务封装，用于加载模型并进行推理。
    """
    def __init__(self, model_path: str = "models/ppo_poker_final"):
        self.model_path = model_path
        self.model: Optional[MaskablePPO] = None
        self.action_space_manager = ActionSpace()
        self.obs_encoder = ObsEncoder()
        self.heuristic_agent = HeuristicAgent()
        
        # Load model on init
        self._load_model()

    def _load_model(self):
        # 1. 尝试加载指定的模型
        target_path = self.model_path + ".zip"
        if os.path.exists(target_path):
            self._try_load_path(self.model_path)
        
        # 2. 如果加载失败，尝试加载 models 目录下最新的匹配模型
        if self.model is None:
            print("Target model not found or invalid. Searching for latest matching model in models/...")
            model_dir = "models"
            if os.path.exists(model_dir):
                files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
                # 按时间排序
                files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                
                for f in files:
                    path = os.path.join(model_dir, f.replace(".zip", ""))
                    if self._try_load_path(path):
                        print(f"Automatically loaded latest model: {f}")
                        break

        if self.model is None:
            print("Warning: No valid RL model found. Using HeuristicAgent as fallback.")

    def _try_load_path(self, path: str) -> bool:
        """尝试从路径加载模型并校验维度"""
        try:
            model = MaskablePPO.load(path)
            model_obs_shape = model.observation_space.shape
            current_obs_shape = self.obs_encoder.shape

            if model_obs_shape == current_obs_shape:
                self.model = model
                return True
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
        return False

    def predict(self, game: Game, player_idx: int) -> Optional[Play]:
        """
        根据当前游戏状态，预测最佳出牌动作。
        """
        # 1. Get Action Mask
        legal_plays = game.get_legal_actions()
        mask = np.zeros(self.action_space_manager.size, dtype=bool)
        
        valid_actions = []
        for play in legal_plays:
            aid = self.action_space_manager.get_id(play)
            if aid != -1 and 0 <= aid < self.action_space_manager.size:
                mask[aid] = True
                valid_actions.append((aid, play))
            elif play.type == HandType.PASS:
                mask[0] = True
                valid_actions.append((0, play))

        # 2. 如果模型存在，尝试进行预测
        if self.model is not None:
            try:
                obs = self.obs_encoder.encode(game, player_idx)
                action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
                
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                candidates = [p for aid, p in valid_actions if aid == action]
                if candidates:
                    return candidates[0]
            except Exception as e:
                print(f"RL Prediction failed: {e}. Falling back to heuristic.")

        # 3. 兜底方案：使用 HeuristicAgent
        print("Using Heuristic fallback...")
        obs = self.obs_encoder.encode(game, player_idx)
        h_mask = mask.astype(np.int8)
        action_id = self.heuristic_agent.act(obs, h_mask)
        
        candidates = [p for aid, p in valid_actions if aid == action_id]
        if candidates:
            return candidates[0]
            
        return legal_plays[0] if legal_plays else None
