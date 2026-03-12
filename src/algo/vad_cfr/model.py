import numpy as np
import os
from typing import Dict, Optional, List
from src.core.game import Game
from src.core.hand_type import Play
from src.env.obs_encoder import ObsEncoder
from src.env.action_space import ActionSpace
from .deep_model import ModelWrapper

class VADCFRModel:
    """
    VAD-CFR Inference Model.
    Uses Deep Neural Networks trained with Volatility-Adaptive Discounting.
    """

    def __init__(self, model_path: str):
        self.action_space = ActionSpace()
        self.encoder = ObsEncoder()
        # Default to CPU for inference
        state_dim = self.encoder.shape[0]
        self.wrapper = ModelWrapper(state_dim, self.action_space.size, device="cpu")
        self.load(model_path)

    def load(self, path: str):
        """Load model weights"""
        if os.path.exists(path):
            try:
                self.wrapper.load(path)
                print(f"VAD-CFR Model loaded from {path}")
            except Exception as e:
                print(f"Error loading VAD-CFR model: {e}")
        else:
            print(f"Warning: VAD-CFR Model file not found at {path}")

    def get_action(self, game: Game, deterministic: bool = True) -> Play:
        """
        Predict best action using the Strategy Net.
        """
        legal_plays = game.get_legal_actions()
        if not legal_plays:
            return None
        if len(legal_plays) == 1:
            return legal_plays[0]

        obs = self.encoder.encode(game, game.current_player)
        
        # Use Strategy Net directly (VAD-CFR average strategy)
        probs_all = self.wrapper.predict_strategy(obs)
        
        # Filter legal actions
        legal_ids = []
        id_to_play = {}
        for p in legal_plays:
            aid = self.action_space.get_id(p)
            if aid != -1:
                legal_ids.append(aid)
                id_to_play[aid] = p
        
        if not legal_ids:
            return legal_plays[0]

        # Normalize probabilities for legal actions
        probs_legal = probs_all[legal_ids]
        if np.sum(probs_legal) > 0:
            probs_legal /= np.sum(probs_legal)
        else:
            # Fallback to uniform if model is unsure
            probs_legal = np.ones(len(legal_ids)) / len(legal_ids)

        if deterministic:
            chosen_id = legal_ids[np.argmax(probs_legal)]
        else:
            chosen_id = np.random.choice(legal_ids, p=probs_legal)
            
        return id_to_play[chosen_id]
