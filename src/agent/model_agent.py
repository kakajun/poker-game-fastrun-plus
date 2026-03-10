import numpy as np
import os
import torch
from sb3_contrib import MaskablePPO
from typing import Optional, Union, Any
from src.algo.mccfr.deep_model import ModelWrapper
from src.algo.mccfr.model import MCCFRModel

class ModelAgent:
    """
    封装已训练的模型作为环境中的对手。
    支持 MaskablePPO (.zip), DeepMCCFR (.pth) 和 Tabular MCCFR (.pkl) 模型。
    """
    def __init__(self, model_path: str, deterministic: bool = True):
        self.model_path = model_path
        self.deterministic = deterministic
        self.ppo_model: Optional[MaskablePPO] = None
        self.mccfr_model: Optional[ModelWrapper] = None
        self.tabular_mccfr: Optional[MCCFRModel] = None
        
        self._load_model()

    def _load_model(self):
        if self.model_path.endswith(".zip"):
            self.ppo_model = MaskablePPO.load(self.model_path)
        elif self.model_path.endswith(".pth"):
            # Deep MCCFR 模型
            from src.env.action_space import ActionSpace
            action_space = ActionSpace()
            self.mccfr_model = ModelWrapper(42, action_space.size, device="cpu")
            self.mccfr_model.load(self.model_path)
        elif self.model_path.endswith(".pkl"):
            # Tabular MCCFR 模型
            self.tabular_mccfr = MCCFRModel(self.model_path)
        else:
            # 默认尝试作为 PPO 加载
            try:
                self.ppo_model = MaskablePPO.load(self.model_path)
            except:
                raise ValueError(f"Unsupported model format: {self.model_path}")

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, game: Optional[Any] = None) -> int:
        # 1. 如果是 PPO 模型
        if self.ppo_model is not None:
            # PPO 预测需要 obs 和 mask
            try:
                action, _ = self.ppo_model.predict(
                    obs, 
                    action_masks=action_mask.astype(bool) if action_mask is not None else None, 
                    deterministic=self.deterministic
                )
                if isinstance(action, np.ndarray):
                    action = action.item()
                return action
            except Exception as e:
                # 可能是维度不匹配
                return 0

        # 2. 如果是 Deep MCCFR 模型
        if self.mccfr_model is not None:
            regrets = self.mccfr_model.predict_regrets(obs)
            
            # 这里的 action_mask 是 0/1 数组
            if action_mask is not None:
                # 只考虑合法动作的后悔值
                legal_indices = np.where(action_mask > 0)[0]
                if len(legal_indices) == 0:
                    return 0 # Pass
                
                pos_regrets = np.maximum(regrets[legal_indices], 0)
                if np.sum(pos_regrets) > 0:
                    probs = pos_regrets / np.sum(pos_regrets)
                else:
                    probs = np.ones(len(legal_indices)) / len(legal_indices)
                
                if self.deterministic:
                    return legal_indices[np.argmax(probs)]
                else:
                    return np.random.choice(legal_indices, p=probs)
            else:
                return np.argmax(regrets)

        # 3. 如果是 Tabular MCCFR 模型
        if self.tabular_mccfr is not None and game is not None:
            # Tabular 需要 Game 对象来生成 InfoSet Key
            from src.env.action_space import ActionSpace
            action_space = ActionSpace()
            
            legal_plays = game.get_legal_actions()
            if not legal_plays: return 0
            
            # 使用模型获取最佳动作 Play
            best_play = self.tabular_mccfr.get_best_action(game) if self.deterministic else self.tabular_mccfr.get_action(game)
            
            # 转换为 ID
            return action_space.get_id(best_play)

        return 0 # Fallback
