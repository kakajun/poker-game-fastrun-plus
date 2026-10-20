import numpy as np
import os
from typing import Dict, Optional, List, Union, Tuple
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
        self.wrapper = ModelWrapper(
            state_dim, self.action_space.size, device="cpu")
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

    def get_action(self, game: Game, deterministic: bool = True, verbose: bool = False, return_thinking: bool = False) -> Union[Play, Tuple[Play, Dict]]:
        """
        Predict best action using the Strategy Net.
        Returns:
            - Play: if return_thinking is False
            - (Play, thinking_dict): if return_thinking is True
        """
        legal_plays = game.get_legal_actions()
        if not legal_plays:
            return (None, None) if return_thinking else None

        # We still want to see thinking if return_thinking is True, even for single play
        # But for now, let's keep it simple and just ensure it's None if not populated
        if len(legal_plays) == 1:
            return (legal_plays[0], None) if return_thinking else legal_plays[0]

        obs = self.encoder.encode(game, game.current_player)

        # 1. Get Strategy Probabilities (Average Strategy)
        probs_all = self.wrapper.predict_strategy(obs)

        # 2. Get Predicted Regrets (Current "Thinking" or Advantage)
        regrets_all = self.wrapper.predict_regrets(obs)

        # Filter legal actions
        legal_ids = []
        id_to_play = {}
        for p in legal_plays:
            aid = self.action_space.get_id(p)
            if aid != -1:
                legal_ids.append(aid)
                id_to_play[aid] = p

        if not legal_ids:
            return (legal_plays[0], None) if return_thinking else legal_plays[0]

        # Normalize probabilities for legal actions
        probs_legal = probs_all[legal_ids]
        sum_probs = np.sum(probs_legal)
        if sum_probs > 0:
            probs_legal /= sum_probs
        else:
            probs_legal = np.ones(len(legal_ids)) / len(legal_ids)

        if deterministic:
            idx = np.argmax(probs_legal)
            chosen_id = legal_ids[idx]
        else:
            idx = np.random.choice(len(legal_ids), p=probs_legal)
            chosen_id = legal_ids[idx]

        thinking = {}
        if verbose or return_thinking:
            # Combine info for all legal actions and group by unique play string
            unique_actions = {}
            for i, aid in enumerate(legal_ids):
                play_str = str(id_to_play[aid])
                if play_str not in unique_actions or regrets_all[aid] > unique_actions[play_str]['regret']:
                    unique_actions[play_str] = {
                        'play': play_str,
                        'regret': float(regrets_all[aid]),
                        'prob': float(probs_legal[i]),
                        'cards': [c.id for c in id_to_play[aid].cards] if id_to_play[aid] else []
                    }

            action_info = list(unique_actions.values())
            action_info.sort(key=lambda x: x['prob'], reverse=True)

            thinking = {
                'player_idx': int(game.current_player),
                'top_actions': action_info[:10],
                'chosen_play': str(id_to_play[chosen_id])
            }

        if verbose:
            print(f"\n[AI Thinking - Player {game.current_player}]")
            print(f"Hand: {[str(c) for c in game.hands[game.current_player]]}")
            last_play_str = str(
                game.last_play) if game.last_play else "New Round"
            print(f"Table: {last_play_str}")
            print(f"{'Action':<35} | {'Regret':<10} | {'Prob':<10}")
            print("-" * 60)

            for info in thinking['top_actions']:
                mark = ">> " if info['play'] == thinking['chosen_play'] else "   "
                print(
                    f"{mark}{info['play']:<32} | {info['regret']:>10.4f} | {info['prob']:>10.2%}")

            if len(unique_actions) > 10:
                print(
                    f"... and {len(unique_actions) - 10} more unique actions.")

        if return_thinking:
            return id_to_play[chosen_id], thinking
        return id_to_play[chosen_id]
