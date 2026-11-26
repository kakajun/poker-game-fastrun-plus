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
    
    # 每人起手 16 张（48 张牌 / 3 人）
    HAND_SIZE = 16

    def __init__(self):
        # 15(手牌) + 2(对手张数) + 15(上家出牌) + 10(牌型) + 15(已出牌统计) + 3(额外特征) = 60
        self.shape = (60,)

    def encode(self, game: Game, player_idx: int) -> np.ndarray:
        """
        编码当前局面 (以 player_idx 视角)
        """
        # 1. 我的手牌 (15维: 3..2 各点数张数)
        my_hand = np.zeros(15, dtype=np.float32)
        hand_ranks = [c.rank.value for c in game.hands[player_idx]]
        for r_val in hand_ranks:
            idx = r_val - 3
            if 0 <= idx < 15:
                my_hand[idx] += 1.0
        
        # 2. 其他两家剩余牌数 (2维, 归一化 / 16)
        others_count = []
        for i in range(1, 3):
            other_idx = (player_idx + i) % 3
            count = len(game.hands[other_idx])
            others_count.append(count / self.HAND_SIZE)
        others_vec = np.array(others_count, dtype=np.float32)
        
        # 3. 当前桌面牌 (Last Play)
        last_play_ranks = np.zeros(15, dtype=np.float32)
        last_play_type = np.zeros(10, dtype=np.float32)
        
        is_new_round = (game.last_play_player == player_idx) or (game.last_play is None)
        
        if not is_new_round and game.last_play:
            for c in game.last_play.cards:
                idx = c.rank.value - 3
                if 0 <= idx < 15:
                    last_play_ranks[idx] += 1.0
            type_idx = game.last_play.type.value - 1
            if 0 <= type_idx < 10:
                last_play_type[type_idx] = 1.0
        
        # 4. 已出牌统计 (15维)
        played_cards_vec = np.zeros(15, dtype=np.float32)
        for card_id in game.played_card_ids:
            rank_val = (card_id // 4) + 3
            played_cards_vec[rank_val - 3] += 0.25 
            
        # 5. 额外特征 (3维)
        extra_features = np.zeros(3, dtype=np.float32)
        # - 是否由我发牌 (1/0)
        extra_features[0] = 1.0 if is_new_round else 0.0
        # - 手牌中最大牌的 Rank (归一化 / 15)
        if hand_ranks:
            extra_features[1] = max(hand_ranks) / 15.0
        # - 对手是否报单 (报警)
        next_player = (player_idx + 1) % 3
        if len(game.hands[next_player]) == 1:
            extra_features[2] = 1.0
            
        # Concatenate: 15 + 2 + 15 + 10 + 15 + 3 = 60
        return np.concatenate([
            my_hand,
            others_vec,
            last_play_ranks,
            last_play_type,
            played_cards_vec,
            extra_features
        ])

