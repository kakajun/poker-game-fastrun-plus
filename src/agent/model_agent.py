import numpy as np
from sb3_contrib import MaskablePPO
from typing import Optional

class ModelAgent:
    """
    封装已训练的 MaskablePPO 模型作为环境中的对手。
    """
    def __init__(self, model_path: str, deterministic: bool = True):
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        action, _ = self.model.predict(
            obs, 
            action_masks=action_mask, 
            deterministic=self.deterministic
        )
        if isinstance(action, np.ndarray):
            action = action.item()
        return action
