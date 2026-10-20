"""控制权估计（M1 近似，M2 会换成约束采样）。

规则是"必压"，所以"对手管不起"= 我继续首出。估计这个概率时必须按牌型算：

  * 单张: 对手要有更大的点数（或炸弹）；
  * 对子/三张: 要有更大的同点数 2/3 张；
  * 顺子: 要有更长的连续同型窗口（5 连 → 需要 5 个连续点数各至少 1 张）；
  * 连对/飞机/飞机带翅膀: 同理，步长 2/3；
  * 炸弹: 要更大的炸弹。

按点数独立近似（超几何），对 GreedyMin 这类对手足够用；M2 用约束采样替换。

（早期版本把"任意比它大的单张"当成能管顺子，导致低估组合牌的控制力，
 策略因此不敢出顺子——这是一次实测踩出来的坑，注释留档。）
"""

import math

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState


def unseen(state: GameState, player: int) -> int:
    return C.FULL_DECK - state.played - state.hands[player]


def _windows(un: int, max_rank: int, length: int, copies: int):
    """所有"能管住"的点数窗口 [(rank_idx, 需要张数), ...]。"""
    out = []
    for start in range(C.NUM_RANKS):
        end = start + length - 1
        if end > C.MAX_HAND_IDX:
            break
        if end + C.RANK_MIN <= max_rank:
            continue
        window = [(i, copies) for i in range(start, end + 1)]
        if all(C.n_at(un, i) >= copies for i, _ in window):
            out.append(window)
    return out


def _beat_windows(un: int, move: Move, cfg: RuleConfig):
    t = move.type
    if t == MoveType.SINGLE:
        ws = _windows(un, move.max_rank, 1, 1)
    elif t == MoveType.PAIR:
        ws = _windows(un, move.max_rank, 1, 2)
    elif t == MoveType.TRIPLE:
        ws = _windows(un, move.max_rank, 1, 3)
    elif t in (MoveType.TRIPLE_WITH_SINGLE, MoveType.TRIPLE_WITH_TWO):
        ws = _windows(un, move.max_rank, 1, 3)
    elif t == MoveType.STRAIGHT:
        ws = _windows(un, move.max_rank, move.length, 1)
    elif t == MoveType.DOUBLE_SEQ:
        ws = _windows(un, move.max_rank, move.length, 2)
    elif t in (MoveType.AIRPLANE, MoveType.AIRPLANE_WITH_WINGS):
        ws = _windows(un, move.max_rank, move.length, 3)
    elif t == MoveType.BOMB:
        return _windows(un, move.max_rank, 1, 4)
    else:
        ws = []

    if t != MoveType.BOMB and cfg.must_beat_bomb:
        # 任意 4 张同点数（炸弹）都能压非炸弹
        for i in range(C.NUM_RANKS):
            if C.n_at(un, i) == 4:
                ws.append([(i, 4)])
    return ws


def _p_hold(c: int, n: int, u: int, k: int) -> float:
    if n <= 0 or u <= 0 or c < k:
        return 0.0
    if u - c < n - k:
        return 1.0
    try:
        return (math.comb(c, k) * math.comb(u - c, n - k)) / math.comb(u, n)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _p_beat(un: int, n: int, u: int, windows) -> float:
    """对手 n 张牌里能凑出任意一个 windows 的概率（按点数独立近似）。"""
    if n <= 0 or not windows:
        return 0.0
    p_none = 1.0
    for window in windows:
        p_hold = 1.0
        for i, k in window:
            p_hold *= _p_hold(C.n_at(un, i), n, u, k)
            if p_hold <= 0.0:
                break
        p_none *= (1.0 - p_hold)
    return 1.0 - p_none


def keep_lead_prob(state: GameState, player: int, move: Move,
                   cfg: RuleConfig = DEFAULT_RULES) -> float:
    """打出 move 后所有对手都管不起（我继续首出）的概率。"""
    if move.is_pass:
        return 0.0
    un = unseen(state, player)
    u = C.total(un)
    if u <= 0:
        return 1.0
    windows = _beat_windows(un, move, cfg)
    if not windows:
        return 1.0

    p_all = 1.0
    for q in range(3):
        if q == player:
            continue
        n = C.total(state.hands[q])
        if n <= 0:
            continue
        p_all *= (1.0 - _p_beat(un, n, u, windows))
    return max(0.0, min(1.0, p_all))


def threat_level(state: GameState, player: int) -> float:
    worst = 0.0
    for q in range(3):
        if q == player:
            continue
        n = C.total(state.hands[q])
        if n <= 0:
            continue
        worst = max(worst, 1.0 / (n + 0.5))
    return worst


# 大牌的机会成本：越大的牌越该留到关键回合
BURN_WEIGHTS = {12: 8.0, 11: 6.0, 10: 3.5, 9: 2.0, 8: 1.2, 7: 0.6, 6: 0.3}


def burn_cost(move: Move) -> float:
    cost = 0.0
    for i, w in BURN_WEIGHTS.items():
        cost += C.n_at(move.cards, i) * w
    return cost


def bomb_cost(state: GameState, player: int, move: Move) -> float:
    """炸弹的机会成本: 早炸通常亏牌权，残局炸可能直接赢。"""
    if move.type != MoveType.BOMB:
        return 0.0
    mine = C.total(state.hands[player])
    if mine <= 4:
        return 0.0
    opp_min = min(C.total(state.hands[q]) for q in range(3) if q != player)
    if opp_min <= 2:
        return 1.0
    return 5.0
