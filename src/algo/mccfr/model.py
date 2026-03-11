import numpy as np
import pickle
import os
from typing import Dict, Optional, List
from src.core.game import Game
from src.core.hand_type import Play
from .info_set import InfoNode, InfoSetManager
from .deep_model import ModelWrapper
from src.env.obs_encoder import ObsEncoder
from src.env.action_space import ActionSpace


class MCCFRModel:
    """
    MCCFR 推理模型 (Tabular 版)。
    使用训练得到的平均策略进行决策。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.nodes: Dict[str, InfoNode] = {}
        if model_path:
            self.load(model_path)

    def load(self, path: str):
        """从文件加载模型节点"""
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    self.nodes = pickle.load(f)
                print(
                    f"MCCFR Model loaded from {path}, nodes: {len(self.nodes)}")
            except Exception as e:
                print(f"Error loading MCCFR model: {e}")
        else:
            print(f"Warning: MCCFR Model file not found at {path}")

    def get_action(self, game: Game) -> Play:
        """
        根据当前游戏状态，选择最优动作 (基于平均策略采样)。
        """
        actions = game.get_legal_actions()
        if not actions:
            return None

        if len(actions) == 1:
            return actions[0]

        info_key = InfoSetManager.get_key(game)

        if info_key in self.nodes:
            node = self.nodes[info_key]
            strategy = node.get_average_strategy()

            # 校验策略长度是否与当前动作匹配
            # 在 CFR 中，同一个信息集的动作集合应该是固定的
            if len(strategy) == len(actions):
                idx = np.random.choice(len(actions), p=strategy)
                return actions[idx]
            else:
                # 如果不匹配，可能是因为动作生成逻辑发生了微调
                # 这种情况下使用贪心或均匀随机作为 fallback
                print(
                    f"Warning: Action count mismatch for {info_key}. Expected {len(strategy)}, got {len(actions)}")
                return actions[np.random.choice(len(actions))]
        else:
            # 没见过的状态 (未覆盖的信息集)，随机选择
            return actions[np.random.choice(len(actions))]

    def get_best_action(self, game: Game) -> Play:
        """
        选择概率最大的动作 (Greedy)。
        """
        actions = game.get_legal_actions()
        if not actions:
            return None

        info_key = InfoSetManager.get_key(game)
        if info_key in self.nodes:
            node = self.nodes[info_key]
            strategy = node.get_average_strategy()
            if len(strategy) == len(actions):
                idx = np.argmax(strategy)
                return actions[idx]

        return self.get_action(game)


class DeepMCCFRModel:
    """
    深度 MCCFR 推理模型。
    使用神经网络预测遗憾值或策略。
    """

    def __init__(self, model_path: str):
        self.action_space = ActionSpace()
        self.encoder = ObsEncoder()
        # 默认使用 CPU 进行推理，防止后端服务占用过多 GPU 显存
        # 显式使用新的维度 57
        state_dim = self.encoder.shape[0]
        self.wrapper = ModelWrapper(state_dim, self.action_space.size, device="cpu")
        self.load(model_path)

    def load(self, path: str):
        """从文件加载模型权重"""
        if os.path.exists(path):
            try:
                self.wrapper.load(path)
                print(f"Deep MCCFR Model loaded from {path}")
            except Exception as e:
                print(f"Error loading Deep MCCFR model: {e}")
        else:
            print(f"Warning: Deep MCCFR Model file not found at {path}")

    def get_action(self, game: Game, deterministic: bool = True) -> Play:
        """
        根据当前局面预测最佳动作。
        优先使用 Strategy Net (平均策略)，如果效果不好可切换回 Regret Net。
        """
        legal_plays = game.get_legal_actions()
        if not legal_plays:
            return None
        if len(legal_plays) == 1:
            return legal_plays[0]

        obs = self.encoder.encode(game, game.current_player)
        
        # 方案 A: 使用 Strategy Net 直接预测概率 (纳什均衡平均策略)
        probs_all = self.wrapper.predict_strategy(obs)
        
        # 获取合法动作 ID
        legal_ids = []
        id_to_play = {}
        for p in legal_plays:
            aid = self.action_space.get_id(p)
            if aid != -1:
                legal_ids.append(aid)
                id_to_play[aid] = p
        
        if not legal_ids:
            return legal_plays[0]

        # 重新归一化合法动作的概率
        probs_legal = probs_all[legal_ids]
        if np.sum(probs_legal) > 0:
            probs_legal /= np.sum(probs_legal)
        else:
            # 如果策略网络没学到合法动作，回退到 Regret Matching 或随机
            probs_legal = np.ones(len(legal_ids)) / len(legal_ids)

        if deterministic:
            chosen_id = legal_ids[np.argmax(probs_legal)]
        else:
            chosen_id = np.random.choice(legal_ids, p=probs_legal)
            
        return id_to_play[chosen_id]
