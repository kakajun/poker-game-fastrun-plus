import numpy as np
from typing import List, Optional
from src.core.game import Game
from src.core.card import Card, Rank, Suit
from src.core.hand_type import HandType, Play

class ObsEncoder:
    """
    状态编码器 (Observation Encoder)
    将 GameState 转换为神经网络可输入的 numpy 数组。
    Observation Vector (168 dim):
    - [0:52]: My Hand (One-hot)
    - [52:54]: Others Card Count (Normalized / 15)
    - [54:106]: Last Play Cards (One-hot)
    - [106:116]: Last Play Type (One-hot)
    - [116:168]: Already Played Cards (One-hot history)
    """
    
    def __init__(self):
        self.shape = (168,)

    def encode(self, game: Game, player_idx: int) -> np.ndarray:
        """
        编码当前局面 (以 player_idx 视角)
        """
        # 1. 我的手牌 (52维 0/1)
        my_hand = np.zeros(52, dtype=np.float32)
        for c in game.hands[player_idx]:
            if 0 <= c.id < 52:
                my_hand[c.id] = 1.0
        
        # 2. 其他两家剩余牌数 (2维, 归一化 / 15)
        others_count = []
        for i in range(1, 3):
            # i=1: next player, i=2: prev player
            other_idx = (player_idx + i) % 3
            count = len(game.hands[other_idx])
            others_count.append(count / 15.0)
        others_vec = np.array(others_count, dtype=np.float32)
        
        # 3. 当前桌面牌 (Last Play)
        last_play_cards = np.zeros(52, dtype=np.float32)
        last_play_type = np.zeros(10, dtype=np.float32)
        
        # 判断是否是新的一轮 (自由出牌)
        # 如果 last_play_player == current_player，说明其他人都过了，或者这是首轮
        is_new_round = (game.last_play_player == player_idx) or (game.last_play is None)
        
        if not is_new_round and game.last_play:
            # Cards
            for c in game.last_play.cards:
                if 0 <= c.id < 52:
                    last_play_cards[c.id] = 1.0
            # Type (One-hot)
            # HandType value starts from 1. 
            # Map 1..10 to 0..9
            type_idx = game.last_play.type.value - 1
            if 0 <= type_idx < 10:
                last_play_type[type_idx] = 1.0
        
        # 4. 已出牌历史 (52维 0/1)
        played_history = np.zeros(52, dtype=np.float32)
        for card_id in game.played_card_ids:
            if 0 <= card_id < 52:
                played_history[card_id] = 1.0
        
        # Concatenate
        return np.concatenate([
            my_hand,
            others_vec,
            last_play_cards,
            last_play_type,
            played_history
        ])

