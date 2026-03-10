from enum import IntEnum, unique
from typing import List, Optional


@unique
class Suit(IntEnum):
    """花色枚举"""
    DIAMOND = 0  # 方块
    CLUB = 1     # 梅花
    HEART = 2    # 红桃
    SPADE = 3    # 黑桃

    def __str__(self):
        return {
            Suit.DIAMOND: '♦',
            Suit.CLUB: '♣',
            Suit.HEART: '♥',
            Suit.SPADE: '♠'
        }[self]


@unique
class Rank(IntEnum):
    """点数枚举
    3-9: 对应实际数字
    10: 10
    11: J
    12: Q
    13: K
    14: A
    15: 2 (跑得快中最大)
    """
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15

    def __str__(self):
        if self.value <= 10:
            return str(self.value)
        return {
            11: 'J',
            12: 'Q',
            13: 'K',
            14: 'A',
            15: '2'
        }[self.value]


class Card:
    """扑克牌类"""

    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.suit}{self.rank}"

    def __str__(self):
        return f"{self.suit}{self.rank}"

    def __lt__(self, other):
        """小于比较，用于排序
        先比点数，点数相同比花色
        """
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit

    def __eq__(self, other):
        """等于比较"""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    @property
    def id(self) -> int:
        """
        返回全局唯一ID (0-51)
        ID = (Rank - 3) * 4 + Suit
        注意：这个ID空间包含被剔除的牌
        """
        return (self.rank - 3) * 4 + self.suit

    @staticmethod
    def from_id(card_id: int) -> 'Card':
        """从全局ID还原卡牌"""
        rank = Rank(card_id // 4 + 3)
        suit_val = card_id % 4
        return Card(rank, Suit(suit_val))
