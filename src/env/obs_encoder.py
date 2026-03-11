import numpy as np
from typing import List, Optional
from src.core.game import Game
from src.core.card import Card, Rank
from src.core.hand_type import HandType, Play

class ObsEncoder:
    """
    状态编码器 (Observation Encoder - 去花色精简版)
    将 GameState 转换为神经网络可输入的 numpy 数组。
    Observation Vector (42 dim):
    - [0:15]: My Hand (Rank count: 3-2)
    - [15:17]: Others Card Count (Normalized / 15)
    - [17:32]: Last Play Cards (Rank count: 3-2)
    - [32:42]: Last Play Type (One-hot)
    """
    
    def __init__(self):
        # 15(手牌) + 2(对手张数) + 15(上家出牌) + 10(牌型) + 15(已出牌统计) = 57
        self.shape = (57,)

    def encode(self, game: Game, player_idx: int) -> np.ndarray:
        """
        编码当前局面 (以 player_idx 视角)
        """
        # 1. 我的手牌 (15维: 3..2 各点数张数)
        my_hand = np.zeros(15, dtype=np.float32)
        for c in game.hands[player_idx]:
            # Rank 3..15 -> index 0..12. index 13,14 reserved for extra if needed
            idx = c.rank.value - 3
            if 0 <= idx < 15:
                my_hand[idx] += 1.0
        
        # 2. 其他两家剩余牌数 (2维, 归一化 / 15)
        others_count = []
        for i in range(1, 3):
            other_idx = (player_idx + i) % 3
            count = len(game.hands[other_idx])
            others_count.append(count / 15.0)
        others_vec = np.array(others_count, dtype=np.float32)
        
        # 3. 当前桌面牌 (Last Play)
        last_play_ranks = np.zeros(15, dtype=np.float32)
        last_play_type = np.zeros(10, dtype=np.float32)
        
        is_new_round = (game.last_play_player == player_idx) or (game.last_play is None)
        
        if not is_new_round and game.last_play:
            # Cards Ranks
            for c in game.last_play.cards:
                idx = c.rank.value - 3
                if 0 <= idx < 15:
                    last_play_ranks[idx] += 1.0
            # Type (One-hot)
            type_idx = game.last_play.type.value - 1
            if 0 <= type_idx < 10:
                last_play_type[type_idx] = 1.0
        
        # 4. 已出牌统计 (15维) - 极其重要，AI 必须知道哪些大牌已经出了
        played_cards_vec = np.zeros(15, dtype=np.float32)
        for card_id in game.played_card_ids:
            # 还原出 rank (Card.from_id 的逻辑)
            rank_val = (card_id // 4) + 3
            played_cards_vec[rank_val - 3] += 0.25 # 归一化，每张占 0.25
            
        # Concatenate: 15 + 2 + 15 + 10 + 15 = 57
        return np.concatenate([
            my_hand,
            others_vec,
            last_play_ranks,
            last_play_type,
            played_cards_vec
        ])

