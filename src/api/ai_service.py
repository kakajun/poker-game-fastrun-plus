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
from src.algo.mccfr.model import DeepMCCFRModel


class AIService:
    """
    AI 服务封装，用于加载模型并进行推理。
    支持 MaskablePPO (SB3) 和 DeepMCCFR 模型。
    """

    def __init__(self, model_path: str = "models/deep_mccfr_gpu.pth"):
        self.model_path = model_path
        self.ppo_model: Optional[MaskablePPO] = None
        self.mccfr_model: Optional[DeepMCCFRModel] = None
        self.action_space_manager = ActionSpace()
        self.obs_encoder = ObsEncoder()
        self.heuristic_agent = HeuristicAgent()

        # Load model on init
        self._load_model()

    def _load_model(self):
        # 1. 尝试加载 Deep MCCFR 模型 (.pth)
        if self.model_path.endswith(".pth") and os.path.exists(self.model_path):
            try:
                self.mccfr_model = DeepMCCFRModel(self.model_path)
                print(
                    f"Successfully loaded Deep MCCFR model from {self.model_path}")
                return
            except Exception as e:
                print(f"Failed to load Deep MCCFR model: {e}")

        # 2. 尝试加载 PPO 模型 (.zip)
        ppo_path = self.model_path if self.model_path.endswith(
            ".zip") else self.model_path + ".zip"
        if os.path.exists(ppo_path):
            if self._try_load_ppo(self.model_path.replace(".zip", "")):
                return

        # 3. 自动搜索最新模型
        print("Searching for any valid model in models/...")
        model_dir = "models"
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            # 优先找 .pth, 其次 .zip
            files.sort(key=lambda x: os.path.getmtime(
                os.path.join(model_dir, x)), reverse=True)

            # 先找最新的 .pth
            for f in files:
                if f.endswith(".pth"):
                    try:
                        self.mccfr_model = DeepMCCFRModel(
                            os.path.join(model_dir, f))
                        print(
                            f"Automatically loaded latest Deep MCCFR model: {f}")
                        return
                    except:
                        continue

            # 再找最新的 .zip
            for f in files:
                if f.endswith(".zip"):
                    path = os.path.join(model_dir, f.replace(".zip", ""))
                    if self._try_load_ppo(path):
                        print(f"Automatically loaded latest PPO model: {f}")
                        return

        print("Warning: No valid AI model found. Using HeuristicAgent as fallback.")

    def _try_load_ppo(self, path: str) -> bool:
        """尝试从路径加载 PPO 模型并校验维度"""
        try:
            model = MaskablePPO.load(path)
            if model.observation_space.shape == self.obs_encoder.shape:
                self.ppo_model = model
                return True
        except Exception as e:
            print(f"Error loading PPO model from {path}: {e}")
        return False

    def predict(self, game: Game, player_idx: int) -> Optional[Play]:
        """
        根据当前游戏状态，预测最佳出牌动作。
        """
        # 1. 获取合法动作
        legal_plays = game.get_legal_actions()
        if not legal_plays:
            return None

        # 2. 优先使用 MCCFR 模型
        if self.mccfr_model is not None:
            try:
                return self.mccfr_model.get_action(game, deterministic=True)
            except Exception as e:
                print(f"MCCFR Prediction failed: {e}")

        # 3. 其次使用 PPO 模型
        if self.ppo_model is not None:
            try:
                obs = self.obs_encoder.encode(game, player_idx)

                # 构建 Action Mask
                mask = np.zeros(self.action_space_manager.size, dtype=bool)
                valid_actions = []
                for play in legal_plays:
                    aid = self.action_space_manager.get_id(play)
                    if aid != -1:
                        mask[aid] = True
                        valid_actions.append((aid, play))

                if valid_actions:
                    action, _ = self.ppo_model.predict(
                        obs, action_masks=mask, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = action.item()
                    candidates = [p for aid,
                                  p in valid_actions if aid == action]
                    if candidates:
                        return candidates[0]
            except Exception as e:
                print(f"PPO Prediction failed: {e}")

        # 4. 兜底方案：使用 HeuristicAgent
        print("Using Heuristic fallback...")
        try:
            obs = self.obs_encoder.encode(game, player_idx)
            # Heuristic 还是需要 mask
            mask = np.zeros(self.action_space_manager.size, dtype=np.int8)
            for play in legal_plays:
                aid = self.action_space_manager.get_id(play)
                if aid != -1:
                    mask[aid] = 1

            action_id = self.heuristic_agent.act(obs, mask)
            for play in legal_plays:
                if self.action_space_manager.get_id(play) == action_id:
                    return play
        except:
            pass

        return legal_plays[0]
