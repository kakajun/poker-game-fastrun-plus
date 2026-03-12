import numpy as np
import os
import torch
from typing import Optional, Union, Any
from src.algo.vad_cfr.model import VADCFRModel
from src.env.action_space import ActionSpace

class ModelAgent:
    """
    Agent Wrapper for VAD-CFR Model.
    Replaced all previous algorithms with VAD-CFR.
    """
    def __init__(self, model_path: str, deterministic: bool = True):
        self.model_path = model_path
        self.deterministic = deterministic
        self.vad_model: Optional[VADCFRModel] = None
        self.action_space = ActionSpace()
        
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            return

        # Try loading as VAD-CFR (.pth)
        try:
            self.vad_model = VADCFRModel(self.model_path)
            print(f"Loaded VAD-CFR Agent from {self.model_path}")
        except Exception as e:
            print(f"Failed to load VAD-CFR Agent: {e}")

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None, game: Optional[Any] = None) -> int:
        """
        Return action index (int).
        Needs 'game' object because VAD-CFR model uses Game state for encoding/inference.
        """
        if self.vad_model is not None and game is not None:
            # VAD-CFR get_action returns a Play object
            play = self.vad_model.get_action(game, deterministic=self.deterministic)
            if play:
                return self.action_space.get_id(play)
        
        # Fallback: Random legal action
        if action_mask is not None:
            legal_indices = np.where(action_mask > 0)[0]
            if len(legal_indices) > 0:
                return np.random.choice(legal_indices)
                
        return 0 # Pass or invalid
