from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class HandType(Enum):
    """牌型枚举"""
    SINGLE = auto()             # 单张
    PAIR = auto()               # 对子
    TRIPLE = auto()             # 三张 (通常不能单独出，除非最后一手)
    TRIPLE_WITH_SINGLE = auto()  # 三带一
    TRIPLE_WITH_TWO = auto()     # 三带二
    STRAIGHT = auto()           # 顺子 (>=5)
    DOUBLE_SEQUENCE = auto()    # 连对 (>=2对, 或 >=3对? 跑得快通常 >=2对)
    # 规则: 连对 >= 3对 吗?
    # mainRule.md: "连对: 3对及以上连续对子" -> YES
    AIRPLANE = auto()           # 飞机 (>=2个三张)
    AIRPLANE_WITH_WINGS = auto()  # 飞机带翅膀
    BOMB = auto()               # 炸弹 (4张)
    PASS = auto()               # 过


@dataclass
class Play:
    """出牌动作描述"""
    type: HandType
    cards: List['Card']
    length: int = 0      # For straights, sequences, airplanes (e.g., 5-card straight has length 5)
    max_rank: int = 0    # The primary rank for comparison (e.g., rank of the triple in a trio)
    kicker_rank: int = 0 # Rank of the kicker card(s) (e.g., the single in a TRIPLE_WITH_SINGLE)
    is_bomb: bool = False

    def __repr__(self):
        if self.kicker_rank > 0:
            return f"Play({self.type.name}, {self.cards}, len={self.length}, max={self.max_rank}, kicker={self.kicker_rank})"
        else:
            return f"Play({self.type.name}, {self.cards}, len={self.length}, max={self.max_rank})"
