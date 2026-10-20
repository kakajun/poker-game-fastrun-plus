"""牌型定义、判定与比牌（纯规则，无状态）。"""

from enum import IntEnum
from typing import NamedTuple, Optional

from . import cards as C
from .config import DEFAULT_RULES, RuleConfig


class MoveType(IntEnum):
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    TRIPLE_WITH_SINGLE = 4      # 三带一
    TRIPLE_WITH_TWO = 5         # 三带二
    STRAIGHT = 6                # 顺子
    DOUBLE_SEQ = 7              # 连对
    AIRPLANE = 8                # 飞机
    AIRPLANE_WITH_WINGS = 9     # 飞机带翅膀
    BOMB = 10


class Move(NamedTuple):
    """一个具体出牌。

    length:   顺子=张数, 连对=对子数, 飞机/飞机带翅膀=三张组数, 其它=1
    max_rank: 比大小的主点数
    cards:    要打出的牌（rank 计数打包），PASS 时为 0
    """

    type: MoveType
    length: int
    max_rank: int
    cards: int = 0

    @property
    def is_pass(self) -> bool:
        return self.type == MoveType.PASS

    def __str__(self) -> str:
        if self.is_pass:
            return "PASS"
        return f"{self.type.name}({C.to_str(self.cards)})"


PASS = Move(MoveType.PASS, 0, 0, 0)


# ---------------------------------------------------------------- 判定

def _present(cnt) -> list:
    return [i for i, c in enumerate(cnt) if c]


def _consecutive(present) -> bool:
    return all(present[i + 1] - present[i] == 1 for i in range(len(present) - 1))


def _no_two(present) -> bool:
    return present[-1] <= C.MAX_HAND_IDX


def _is_straight(cnt) -> bool:
    p = _present(cnt)
    if len(p) < 5 or sum(cnt) < 5:
        return False
    return all(cnt[i] == 1 for i in p) and _consecutive(p) and _no_two(p)


def _is_pair_seq(cnt, min_len: int) -> bool:
    p = _present(cnt)
    if len(p) < min_len:
        return False
    return all(cnt[i] == 2 for i in p) and _consecutive(p) and _no_two(p)


def _is_airplane(cnt) -> bool:
    p = _present(cnt)
    if len(p) < 2:
        return False
    return all(cnt[i] == 3 for i in p) and _consecutive(p) and _no_two(p)


def _apw_candidates(cnt):
    """产出 (三张主体窗口, 去掉主体后的余牌计数)。"""
    n_core = [i for i in range(C.MAX_HAND_IDX + 1) if cnt[i] >= 3]
    for start in range(0, len(n_core)):
        for end in range(start + 2, len(n_core) + 1):
            win = n_core[start:end]
            if not _consecutive(win):
                break
            rem = list(cnt)
            for i in win:
                rem[i] -= 3
            yield win, rem


def classify(cards: int, cfg: RuleConfig = DEFAULT_RULES) -> Optional[Move]:
    """把一手具体的牌判定为 Move；不合法返回 None。"""
    if cards <= 0:
        return None
    tot = C.total(cards)
    cnt = C.counts(cards)
    p = _present(cnt)

    if tot == 1:
        return Move(MoveType.SINGLE, 1, p[0] + C.RANK_MIN, cards)
    if tot == 2 and len(p) == 1:
        return Move(MoveType.PAIR, 1, p[0] + C.RANK_MIN, cards)
    if tot == 3 and len(p) == 1:
        return Move(MoveType.TRIPLE, 1, p[0] + C.RANK_MIN, cards)
    if tot == 4 and len(p) == 1:
        return Move(MoveType.BOMB, 1, p[0] + C.RANK_MIN, cards)
    if tot == 4 and len(p) == 2:
        trips = [i for i in p if cnt[i] == 3]
        if len(trips) == 1:
            return Move(MoveType.TRIPLE_WITH_SINGLE, 1, trips[0] + C.RANK_MIN, cards)
        return None
    if tot == 5:
        trips = [i for i in p if cnt[i] == 3]
        if len(trips) == 1:
            return Move(MoveType.TRIPLE_WITH_TWO, 1, trips[0] + C.RANK_MIN, cards)

    if _is_straight(cnt):
        return Move(MoveType.STRAIGHT, tot, p[-1] + C.RANK_MIN, cards)
    if _is_pair_seq(cnt, cfg.min_pair_seq):
        return Move(MoveType.DOUBLE_SEQ, len(p), p[-1] + C.RANK_MIN, cards)
    if _is_airplane(cnt):
        return Move(MoveType.AIRPLANE, len(p), p[-1] + C.RANK_MIN, cards)

    # 飞机带翅膀: 4n 张(带单) 或 5n 张(带对)
    if tot % 4 == 0:
        n = tot // 4
        if n >= 2:
            for win, rem in _apw_candidates(cnt):
                if len(win) != n:
                    continue
                if sum(rem) == n and (cfg.apw_single_dup or all(v <= 1 for v in rem)):
                    return Move(MoveType.AIRPLANE_WITH_WINGS, n,
                                win[-1] + C.RANK_MIN, cards)
    if cfg.airplane_pair_wings and tot % 5 == 0:
        n = tot // 5
        if n >= 2:
            for win, rem in _apw_candidates(cnt):
                if len(win) != n:
                    continue
                if sum(rem) == 2 * n and all(v % 2 == 0 for v in rem):
                    return Move(MoveType.AIRPLANE_WITH_WINGS, n,
                                win[-1] + C.RANK_MIN, cards)
    return None


def can_beat(cur: Move, target: Move) -> bool:
    if cur is None or target is None:
        return False
    if cur.type == MoveType.PASS or target.type == MoveType.PASS:
        return False
    if cur.type == MoveType.BOMB:
        return target.type != MoveType.BOMB or cur.max_rank > target.max_rank
    if target.type == MoveType.BOMB:
        return False
    return (cur.type == target.type
            and cur.length == target.length
            and cur.max_rank > target.max_rank)
