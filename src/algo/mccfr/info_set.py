import numpy as np
from typing import List, Dict, Tuple
from src.core.game import Game
from src.core.hand_type import Play

class InfoNode:
    """
    CFR 节点，存储特定信息集的累积遗憾和累积策略。
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)

    def get_strategy(self, realization_weight: float) -> np.ndarray:
        """
        使用 Regret Matching 算法从累积遗憾中获取当前策略。
        """
        # 只考虑正遗憾
        positive_regrets = np.maximum(self.regret_sum, 0)
        sum_pos_regret = np.sum(positive_regrets)

        if sum_pos_regret > 0:
            self.strategy = positive_regrets / sum_pos_regret
        else:
            # 如果没有正遗憾，则均匀分布
            self.strategy = np.ones(self.num_actions) / self.num_actions

        # 累积策略用于最终的平均策略输出
        self.strategy_sum += realization_weight * self.strategy
        return self.strategy

    def get_average_strategy(self) -> np.ndarray:
        """
        获取最终的训练结果：平均策略。
        """
        sum_strategy = np.sum(self.strategy_sum)
        if sum_strategy > 0:
            return self.strategy_sum / sum_strategy
        else:
            return np.ones(self.num_actions) / self.num_actions

class InfoSetManager:
    """
    信息集管理与抽象。
    """
    @staticmethod
    def get_key(game: Game) -> str:
        """
        生成信息集的唯一 key。
        """
        player_id = game.current_player
        
        # 1. 我的手牌 (Rank 序列)
        hand = game.hands[player_id]
        hand_ranks = sorted([c.rank.value for c in hand])
        hand_str = ",".join(map(str, hand_ranks))
        
        # 2. 其他人张数 (下家, 下下家)
        others_count = []
        for i in range(1, 3):
            p = (player_id + i) % 3
            others_count.append(len(game.hands[p]))
        others_count_str = "|".join(map(str, others_count))
        
        # 3. 桌面最后出牌
        if game.last_play:
            lp = game.last_play
            lp_str = f"{lp.type.value}:{lp.max_rank}:{lp.length}"
        else:
            lp_str = "FREE"
            
        # 4. 关键已出牌 (2 和 A 在跑得快中非常关键)
        # 这里为了减小空间，只记录 2 和 A 是否已出
        # Rank.TWO = 15, Rank.ACE = 14
        # 我们检查 played_card_ids
        # 注意：这里需要根据 Card.id 来判断，或者直接根据 Rank
        # 简单起见，暂不加入已出牌，待优化时再加。
        
        return f"P{player_id}:{hand_str}#{others_count_str}#{lp_str}"
