from enum import IntEnum, unique
from typing import List, Optional


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
    """扑克牌类 (已移除花色，仅保留 rank 和用于区分的 instance_id)"""

    def __init__(self, rank: Rank, instance_id: int = 0):
        self.rank = rank
        self.instance_id = instance_id  # 0-3, 用于区分同点数的不同张牌

    def __repr__(self):
        return f"{self.rank}"

    def __str__(self):
        return f"{self.rank}"

    def __lt__(self, other):
        """小于比较，用于排序"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.instance_id < other.instance_id

    def __eq__(self, other):
        """等于比较 (逻辑上点数相同即视为相同)"""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank

    def __hash__(self):
        return hash(self.rank)

    @property
    def id(self) -> int:
        """
        返回全局唯一ID (0-51)
        ID = (Rank - 3) * 4 + instance_id
        """
        return (self.rank - 3) * 4 + self.instance_id

    @staticmethod
    def from_id(card_id: int) -> 'Card':
        """从全局ID还原卡牌"""
        rank = Rank(card_id // 4 + 3)
        instance_id = card_id % 4
        return Card(rank, instance_id)
