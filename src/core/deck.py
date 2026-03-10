import random
from typing import List, Tuple
from src.core.card import Card, Rank, Suit


class Deck:
    """扑克牌堆管理"""

    def __init__(self, seed: int = None):
        self.cards: List[Card] = []
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.build()

    def build(self):
        """
        构建跑得快专用的牌堆 (45张)
        去掉大小王
        3-Q (4张): 全保留
        K (3张): 去掉方块K (Suit=0) -> 保留 Club(1), Heart(2), Spade(3)
        A (1张): 保留黑桃A (Suit=3)
        2 (1张): 保留红桃2 (Suit=2)
        """
        self.cards = []

        # 3 - Q (3-12)
        for r in range(3, 13):
            rank = Rank(r)
            for s in Suit:
                self.cards.append(Card(rank, s))

        # K (13) - 去掉方块K
        self.cards.append(Card(Rank.KING, Suit.CLUB))
        self.cards.append(Card(Rank.KING, Suit.HEART))
        self.cards.append(Card(Rank.KING, Suit.SPADE))

        # A (14) - 保留黑桃A
        self.cards.append(Card(Rank.ACE, Suit.SPADE))

        # 2 (15) - 保留红桃2
        self.cards.append(Card(Rank.TWO, Suit.HEART))

        # 校验数量
        assert len(
            self.cards) == 45, f"Deck size should be 45, but got {len(self.cards)}"

    def shuffle(self):
        """洗牌"""
        random.shuffle(self.cards)

    def deal(self) -> Tuple[List[Card], List[Card], List[Card]]:
        """发牌，均分给3个玩家，每人15张"""
        if len(self.cards) != 45:
            raise ValueError("Deck must have 45 cards before dealing")

        hand1 = sorted(self.cards[0:15])
        hand2 = sorted(self.cards[15:30])
        hand3 = sorted(self.cards[30:45])

        return hand1, hand2, hand3

    def __repr__(self):
        return f"Deck({len(self.cards)} cards)"
