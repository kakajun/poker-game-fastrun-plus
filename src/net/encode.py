"""状态特征编码（固定视角 p）。

设计原则:
  * 全部从 p 的视角，相对座位（下家/上家）而不是绝对座位号，避免位置偏置；
  * 只喂"规则允许知道"的信息（自己的牌、已出牌、各家张数、过牌证据），
    绝不喂对手手牌 —— 否则价值网会学到作弊特征，上线即崩；
  * 维度固定 80，全部归一化到 0~1，推理端可批量。
"""

from typing import List, Tuple

import numpy as np

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import MoveType
from ..fastrun.state import GameState
from ..eval.decompose import min_steps

DIM = 80
MOVE_DIM = 19

_TYPES = list(MoveType)


def encode_move(state: GameState, move, cfg: RuleConfig = DEFAULT_RULES) -> np.ndarray:
    """动作特征（与状态特征拼接后送 Q 头）。

    关键几维是"这手牌让局面变好多少"（progress）、"打出去的大牌多贵"（burn）、
    "打完还能不能继续首出"（keep_lead_prob）——这三条是 M1/M2 实测出来的有效成分。
    """
    from ..eval.control import burn_cost, keep_lead_prob
    from ..eval.decompose import min_steps

    f: List[float] = []
    for t in _TYPES:
        f.append(1.0 if move.type == t else 0.0)          # 11
    f.append(min(move.length, 12) / 12.0)                 # 1
    f.append(move.max_rank / 15.0)                        # 1
    f.append(C.total(move.cards) / 15.0)                  # 1
    f.append(1.0 if move.type == MoveType.BOMB else 0.0)  # 1

    hand = state.hand()
    before = min_steps(hand, cfg)
    after = min_steps(hand - move.cards, cfg)
    f.append(min(burn_cost(move), 20.0) / 10.0)           # 1
    f.append(max(-2.0, min(5.0, before - after)) / 5.0)   # 1
    f.append(keep_lead_prob(state, state.current, move, cfg))  # 1
    f.append(min(after, 15) / 15.0)                       # 1

    arr = np.asarray(f, dtype=np.float32)
    assert arr.shape[0] == MOVE_DIM, arr.shape
    return arr


def encode(state: GameState, p: int,
           cfg: RuleConfig = DEFAULT_RULES) -> np.ndarray:
    f: List[float] = []

    own = C.counts(state.hands[p])
    played = C.counts(state.played)
    unseen = C.counts(C.FULL_DECK - state.played - state.hands[p])

    f.extend(c / 4.0 for c in own)                    # 0..12
    f.extend(c / 4.0 for c in played)                 # 13..25
    f.extend(c / 4.0 for c in unseen)                 # 26..38

    nxt = (p + 1) % 3
    prv = (p + 2) % 3
    f.append(C.total(state.hands[nxt]) / 15.0)        # 39
    f.append(C.total(state.hands[prv]) / 15.0)        # 40

    for q in (p, nxt, prv):                            # 41..43
        f.append(1.0 if state.current == q else 0.0)

    leading = state.last_play is None
    f.append(1.0 if leading else 0.0)                  # 44

    if leading:
        f.extend([0.0] * 17)     # 11 牌型 + 长度 + 主点数 + 炸弹 + 3 相对座位
    else:
        lp = state.last_play
        for t in _TYPES:                               # 45..55
            f.append(1.0 if lp.type == t else 0.0)
        f.append(min(lp.length, 12) / 12.0)            # 56
        f.append(lp.max_rank / 15.0)                   # 57
        f.append(1.0 if lp.type == MoveType.BOMB else 0.0)  # 58
        for q in (p, nxt, prv):                        # 59..61
            f.append(1.0 if state.last_player == q else 0.0)

    f.extend(t / 15.0 for t in state.tricks)           # 62..64
    f.extend(max(-60, min(60, b)) / 60.0 for b in state.bombs)  # 65..67
    f.extend(min_steps(state.hands[q], cfg) / 15.0 for q in (p, nxt, prv))  # 68..70

    own_max = C.max_idx(state.hands[p])
    own_min = C.min_idx(state.hands[p])
    f.append((own_max + 3) / 15.0 if own_max >= 0 else 0.0)  # 71
    f.append((own_min + 3) / 15.0 if own_min >= 0 else 0.0)  # 72
    f.append(1.0 if C.total(state.hands[nxt]) == 1 else 0.0)  # 73 顶大压力

    ev_count = {p: 0, nxt: 0, prv: 0}
    ev_max = {p: 0, nxt: 0, prv: 0}
    ev_bomb = {p: 0, nxt: 0, prv: 0}
    for q, target in state.pass_log:
        if q == p:
            continue
        ev_count[q] = ev_count.get(q, 0) + 1
        if target.type == MoveType.SINGLE:
            ev_max[q] = max(ev_max.get(q, 0), target.max_rank)
        if target.type == MoveType.BOMB:
            ev_bomb[q] = 1

    f.append(min(ev_count[nxt], 8) / 8.0)              # 74
    f.append(min(ev_count[prv], 8) / 8.0)              # 75
    f.append(ev_max[nxt] / 15.0)                       # 76
    f.append(ev_max[prv] / 15.0)                       # 77
    f.append(float(ev_bomb[nxt]))                      # 78
    f.append(float(ev_bomb[prv]))                      # 79

    arr = np.asarray(f, dtype=np.float32)
    assert arr.shape[0] == DIM, arr.shape
    return arr
