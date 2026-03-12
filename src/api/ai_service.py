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

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.ppo_model: Optional[MaskablePPO] = None
        self.mccfr_model: Optional[DeepMCCFRModel] = None
        self.action_space_manager = ActionSpace()
        self.obs_encoder = ObsEncoder()
        self.heuristic_agent = HeuristicAgent()

        # Load model on init if path provided, otherwise auto-search
        if model_path:
            self.load_model(model_path)
        else:
            self._load_auto()

    def list_models(self) -> List[str]:
        """列出 models/ 目录下的所有可用模型"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            return []
        files = os.listdir(model_dir)
        valid_extensions = ('.zip', '.pth', '.pkl')
        models = [f for f in files if f.endswith(valid_extensions)]
        # 按修改时间排序，最新的在前
        models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
        return models

    def load_model(self, model_name: str) -> bool:
        """加载指定模型"""
        model_dir = "models"
        # 处理全路径或文件名
        if os.path.dirname(model_name):
            full_path = model_name
        else:
            full_path = os.path.join(model_dir, model_name)

        if not os.path.exists(full_path):
            print(f"Model file not found: {full_path}")
            return False

        print(f"Loading model: {full_path}...")
        self.ppo_model = None
        self.mccfr_model = None

        # 1. Deep MCCFR (.pth)
        if full_path.endswith(".pth"):
            try:
                self.mccfr_model = DeepMCCFRModel(full_path)
                print(f"Successfully loaded Deep MCCFR model: {model_name}")
                return True
            except Exception as e:
                print(f"Failed to load Deep MCCFR model: {e}")
                return False

        # 2. PPO (.zip)
        if full_path.endswith(".zip"):
            # SB3 load 不需要扩展名? 不，load(path) 可以带或不带
            # 但是 MaskablePPO.load 内部可能会补 .zip
            # 我们传入完整路径去掉 .zip 试试，或者直接传
            load_path = full_path.replace(".zip", "")
            if self._try_load_ppo(load_path):
                print(f"Successfully loaded PPO model: {model_name}")
                return True

        print(f"Failed to load model: {model_name} (Unsupported format or load error)")
        return False

    def _load_auto(self):
        """自动加载最新的模型"""
        print("Searching for any valid model in models/...")
        models = self.list_models()
        for m in models:
            if self.load_model(m):
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
