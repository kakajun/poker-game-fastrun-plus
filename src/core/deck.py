from .card import Card, Rank
import random
from typing import Optional


class Deck:
    """牌堆管理类"""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.cards = []
        self.build()

    def build(self):
        """
        构建跑得快专用的牌堆 (45张)
        3-Q: 各 4 张 (共 40 张)
        K: 3 张
        A: 1 张
        2: 1 张
        """
        self.cards = []

        # 3 - Q (3-12)
        for r in range(3, 13):
            rank = Rank(r)
            for i in range(4):
                self.cards.append(Card(rank, i))

        # K (13) - 3 张
        for i in range(3):
            self.cards.append(Card(Rank.KING, i))

        # A (14) - 1 张
        self.cards.append(Card(Rank.ACE, 0))

        # 2 (15) - 1 张
        self.cards.append(Card(Rank.TWO, 0))

        # 校验数量
        assert len(self.cards) == 45, f"Deck size should be 45, but got {len(self.cards)}"

    def shuffle(self):
        """洗牌"""
        random.shuffle(self.cards)

    def deal(self, num_players: int = 3, cards_per_player: int = 15):
        """发牌"""
        if len(self.cards) < num_players * cards_per_player:
            raise ValueError("Not enough cards in deck")

        hands = []
        for i in range(num_players):
            start = i * cards_per_player
            end = start + cards_per_player
            # 排序手牌
            hand = sorted(self.cards[start:end])
            hands.append(hand)

        return hands
