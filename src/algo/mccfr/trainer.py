import random
import numpy as np
from typing import Dict, List, Optional
import os
import pickle
from src.core.game import Game
from src.core.hand_type import Play
from .info_set import InfoNode, InfoSetManager

class MCCFRTrainer:
    """
    MCCFR 训练器 (外部采样)。
    """
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.nodes: Dict[str, InfoNode] = {}
        self.iterations = 0

    def train(self, num_iterations: int):
        """进行若干次迭代训练"""
        for i in range(num_iterations):
            # 每轮迭代采样一个随机初始发牌情况
            game = Game()
            # 依次作为每个玩家的视角进行一次遍历更新
            for p in range(3):
                # 从该玩家视角采样 (实际上 Game 已经随机发牌，我们只需要 clone 即可)
                # shuffle_other_hands 用于确保该玩家不知道其他人的牌
                sim_game = game.clone()
                sim_game.shuffle_other_hands(p)
                self.cfr(sim_game, p)
            
            self.iterations += 1
            if self.iterations % 100 == 0:
                print(f"MCCFR: Iteration {self.iterations}, Nodes: {len(self.nodes)}")

    def cfr(self, game: Game, update_player_id: int) -> float:
        """核心递归算法 (外部采样)"""
        # Debug: 检查递归深度和状态
        # print(f"CFR: P{game.current_player} Depth {len(game.played_card_ids)}")
        
        # 只在第一次进入时打印
        if len(game.played_card_ids) == 0 and game.current_player == 0 and update_player_id == 0:
             print("CFR Started for Player 0")

        if game.is_over:
            # 返回该玩家的收益 (零和化处理: 自己的分 - 别人的平均分?)
            # 跑得快是多人博弈，简单的处理是直接返回其得分
            return float(game.scores[update_player_id])

        curr_player = game.current_player
        actions = game.get_legal_actions()
        
        # 异常处理：如果没有动作，则游戏结束逻辑可能有误
        if not actions:
            return 0.0

        info_key = InfoSetManager.get_key(game)
        
        # 获取或创建节点
        if info_key not in self.nodes:
            self.nodes[info_key] = InfoNode(len(actions))
        node = self.nodes[info_key]

        if curr_player == update_player_id:
            # 更新玩家：遍历所有动作以计算期望值
            # realization_weight 设为 1.0 (外部采样特点)
            # 在外部采样中，更新玩家不更新 strategy_sum
            # 我们只需要计算正遗憾以获取当前策略
            positive_regrets = np.maximum(node.regret_sum, 0)
            sum_pos_regret = np.sum(positive_regrets)
            if sum_pos_regret > 0:
                strategy = positive_regrets / sum_pos_regret
            else:
                strategy = np.ones(len(actions)) / len(actions)
            
            action_utils = np.zeros(len(actions))
            
            # 为了加速训练，对于动作空间过大的节点进行采样遍历 (Sampled CFR)
            # 尤其是首出节点 (actions > 20)
            # 我们只遍历概率较高的动作 + 随机探索
            # 但为了简单起见，这里我们只遍历前 N 个动作 + 随机几个?
            # 或者：如果动作数 > 10，则随机采样 5 个动作进行更新?
            # 注意：如果只更新部分动作，regret_sum 的更新公式需要调整 (Importance Sampling)
            # 但在 External Sampling 中，通常需要遍历所有。
            # 这里我们采用一种近似：只遍历部分动作，未遍历的动作遗憾不更新 (视为0或保持不变)
            
            sampled_indices = range(len(actions))
            if len(actions) > 8:
                # 随机采样 8 个动作
                sampled_indices = np.random.choice(len(actions), 8, replace=False)
            
            for a_idx in sampled_indices:
                action = actions[a_idx]
                # 使用 step 模拟动作。注意 step 会修改状态，因此需要 clone
                next_game = game.clone()
                next_game.step(action)
                action_utils[a_idx] = self.cfr(next_game, update_player_id)
            
            # 计算 node_util (基于全概率公式，但只用采样动作近似?)
            # node_util = sum(strategy * action_utils)
            # 如果我们只计算了部分 action_utils，那么 node_util 只能基于这些计算
            # 或者：未计算的 action_utils 设为当前 node_util (基准)?
            # 简单做法：只在采样动作上计算 regrets
            
            # 重新归一化策略 (只考虑采样动作)
            sampled_strategy = strategy[sampled_indices]
            if np.sum(sampled_strategy) > 0:
                sampled_strategy /= np.sum(sampled_strategy)
            else:
                sampled_strategy = np.ones(len(sampled_indices)) / len(sampled_indices)
                
            node_util = np.sum(sampled_strategy * action_utils[sampled_indices])
            
            # 更新累积遗憾 (Regret Matching)
            # 只更新采样的动作
            for a_idx in sampled_indices:
                node.regret_sum[a_idx] += (action_utils[a_idx] - node_util)
                
            return node_util
        else:
            # 对手玩家：根据当前策略采样一个动作执行
            # 这里需要累积平均策略 (Average Strategy)
            # 在外部采样中，我们只在对手节点采样动作并累积策略
            positive_regrets = np.maximum(node.regret_sum, 0)
            sum_pos_regret = np.sum(positive_regrets)
            if sum_pos_regret > 0:
                strategy = positive_regrets / sum_pos_regret
            else:
                strategy = np.ones(len(actions)) / len(actions)
            
            # 累积策略 (Strategy Sum) 用于最终推理
            # realization_weight 为到达概率，在外部采样中，由于是对等采样，通常设为 1
            node.strategy_sum += strategy
            
            # 随机采样一个动作
            a_idx = np.random.choice(len(actions), p=strategy)
            
            next_game = game.clone()
            next_game.step(actions[a_idx])
            return self.cfr(next_game, update_player_id)

    def save_model(self, path: str):
        """保存模型 (全量节点)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            # 为了减小存储体积，只保存策略和遗憾
            pickle.dump(self.nodes, f)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.nodes = pickle.load(f)
            print(f"Model loaded from {path}, nodes: {len(self.nodes)}")
        else:
            print(f"No model found at {path}")
