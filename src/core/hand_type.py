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
    length: int = 0      # 顺子/连对/飞机的长度 (如顺子5张length=5, 连对3对length=3)
    max_rank: int = 0    # 比较大小的关键点数 (如顺子最大牌, 三带一的三张点数)
    is_bomb: bool = False

    def __repr__(self):
        return f"Play({self.type.name}, {self.cards}, len={self.length}, max={self.max_rank})"
