import os
import sys
import numpy as np
from typing import Optional, List
from src.core.game import Game
from src.core.hand_type import Play, HandType
from src.env.action_space import ActionSpace
from src.env.obs_encoder import ObsEncoder
from src.algo.vad_cfr.model import VADCFRModel


class AIService:
    """
    AI Service using VAD-CFR (Volatility-Adaptive Discounted Counterfactual Regret Minimization).
    Replaced all previous algorithms with VAD-CFR as requested.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.vad_model: Optional[VADCFRModel] = None
        self.action_space_manager = ActionSpace()
        self.obs_encoder = ObsEncoder()

        # Load model on init if path provided, otherwise auto-search
        if model_path:
            self.load_model(model_path)
        else:
            self._load_auto()

    def list_models(self) -> List[str]:
        """List all available VAD-CFR models"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            return []
        files = os.listdir(model_dir)
        # Assuming VAD-CFR uses .pth format
        valid_extensions = ('.pth',)
        models = [f for f in files if f.endswith(valid_extensions)]
        models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        return models

    def load_model(self, model_name: str) -> bool:
        """Load specific VAD-CFR model"""
        model_dir = "models"
        if os.path.dirname(model_name):
            full_path = model_name
        else:
            full_path = os.path.join(model_dir, model_name)

        if not os.path.exists(full_path):
            print(f"Model file not found: {full_path}")
            return False

        print(f"Loading VAD-CFR model: {full_path}...")
        self.vad_model = None

        try:
            self.vad_model = VADCFRModel(full_path)
            print(f"Successfully loaded VAD-CFR model: {model_name}")
            return True
        except Exception as e:
            print(f"Failed to load VAD-CFR model: {e}")
            return False

    def _load_auto(self):
        """Auto-load latest VAD-CFR model"""
        print("Searching for latest VAD-CFR model...")
        models = self.list_models()
        for m in models:
            if self.load_model(m):
                return
        print("Warning: No valid VAD-CFR model found. AI will play randomly or fail.")

    def predict(self, game: Game, player_idx: int) -> Optional[Play]:
        """
        Predict best action using VAD-CFR.
        """
        # 1. Get legal actions
        legal_plays = game.get_legal_actions()
        if not legal_plays:
            return None

        # 2. Use VAD-CFR Model
        if self.vad_model is not None:
            try:
                # Use deterministic strategy for best performance
                return self.vad_model.get_action(game, deterministic=True)
            except Exception as e:
                print(f"VAD-CFR Prediction failed: {e}")

        # 3. Fallback (Random) - Should not happen if model is loaded
        print("Using Random fallback (Model not loaded or failed)...")
        import random
        return random.choice(legal_plays)
