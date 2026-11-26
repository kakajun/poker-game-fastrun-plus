"""v3 合法信息集编码：状态 + 全部合法动作 + 有序公开历史。

硬约束:
  * 只读取 p 自己手牌、公开出牌、公开动作/Pass 历史和各家剩余张数；
  * 绝不读取对手手牌或对手 `min_steps`；
  * 动作编码不依赖 M1 排名，部署时直接给全部合法动作打分。

历史序列固定为最近 `HISTORY_STEPS` 个事件，供 MLP/GRU 使用；更早事件仍以
全量计数特征保留，避免长局只在序列层被截断。
"""

from typing import List, Sequence

import numpy as np

from ..eval.control import burn_cost, keep_lead_prob
from ..eval.decompose import min_steps
from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState

FEATURE_VERSION = "v3"
ACTION_VERSION = "v3"
HISTORY_STEPS = 128
EVENT_DIM = 8
MOVE_DIM = 22
STATE_DIM = 92
_TYPES = list(MoveType)


def encode_state(state: GameState, p: int,
                 cfg: RuleConfig = DEFAULT_RULES) -> np.ndarray:
    """编码 p 的信息集；交换两个对手的隐藏手牌时结果必须完全相同。"""
    own = C.counts(state.hands[p])
    played = C.counts(state.played)
    unseen = C.counts(C.FULL_DECK - state.played - state.hands[p])

    f: List[float] = []
    f.extend(c / 4.0 for c in own)                    # 0..12
    f.extend(c / 4.0 for c in played)                 # 13..25
    f.extend(c / 4.0 for c in unseen)                 # 26..38

    nxt = (p + 1) % 3
    prv = (p + 2) % 3
    f.append(C.total(state.hands[nxt]) / C.HAND_SIZE)  # 39
    f.append(C.total(state.hands[prv]) / C.HAND_SIZE)  # 40

    for q in (p, nxt, prv):                            # 41..43
        f.append(1.0 if state.current == q else 0.0)

    leading = state.last_play is None
    f.append(1.0 if leading else 0.0)                  # 44
    if leading:
        f.extend([0.0] * 17)                          # 45..61
    else:
        lp = state.last_play
        for t in _TYPES:
            f.append(1.0 if lp.type == t else 0.0)     # 45..55
        f.append(min(lp.length, 12) / 12.0)            # 56
        f.append(lp.max_rank / 15.0)                   # 57
        f.append(1.0 if lp.type == MoveType.BOMB else 0.0)  # 58
        for q in (p, nxt, prv):                        # 59..61
            f.append(1.0 if state.last_player == q else 0.0)

    f.extend(t / C.HAND_SIZE for t in state.tricks)    # 62..64
    f.extend(max(-60, min(60, b)) / 60.0 for b in state.bombs)  # 65..67

    # 自己的拆牌手数合法；对手信息只能来自公开事件，不再填对手真实 min_steps。
    f.append(min_steps(state.hands[p], cfg) / C.HAND_SIZE)  # 68

    own_max = C.max_idx(state.hands[p])
    own_min = C.min_idx(state.hands[p])
    f.append((own_max + 3) / 15.0 if own_max >= 0 else 0.0)  # 69
    f.append((own_min + 3) / 15.0 if own_min >= 0 else 0.0)  # 70
    f.append(1.0 if C.total(state.hands[nxt]) == 1 else 0.0)  # 71

    ev_count = {p: 0, nxt: 0, prv: 0}
    ev_max = {p: 0, nxt: 0, prv: 0}
    ev_bomb = {p: 0, nxt: 0, prv: 0}
    for q, target in state.pass_log:
        if q not in ev_count:
            continue
        ev_count[q] += 1
        if target.type == MoveType.SINGLE:
            ev_max[q] = max(ev_max[q], target.max_rank)
        if target.type == MoveType.BOMB:
            ev_bomb[q] = 1

    f.append(min(ev_count[nxt], 8) / 8.0)              # 72
    f.append(min(ev_count[prv], 8) / 8.0)              # 73
    f.append(ev_max[nxt] / 15.0)                       # 74
    f.append(ev_max[prv] / 15.0)                       # 75
    f.append(float(ev_bomb[nxt]))                      # 76
    f.append(float(ev_bomb[prv]))                      # 77

    total_actions = max(1, len(state.action_log))
    type_counts = [0] * len(_TYPES)
    actor_counts = {p: 0, nxt: 0, prv: 0}
    for q, mv in state.action_log:
        type_counts[int(mv.type)] += 1
        if q in actor_counts:
            actor_counts[q] += 1
    f.extend(c / total_actions for c in type_counts)    # 78..88
    f.extend(actor_counts[q] / total_actions for q in (p, nxt, prv))  # 89..91

    arr = np.asarray(f, dtype=np.float32)
    assert arr.shape[0] == STATE_DIM, arr.shape
    return arr


def encode_move(state: GameState, move: Move,
                cfg: RuleConfig = DEFAULT_RULES) -> np.ndarray:
    """动作特征；不包含任何 M1 排名，避免网络只能重排旧策略候选。"""
    f: List[float] = []
    for t in _TYPES:
        f.append(1.0 if move.type == t else 0.0)       # 11
    f.append(min(move.length, 12) / 12.0)              # 1
    f.append(move.max_rank / 15.0)                     # 1
    f.append(C.total(move.cards) / C.HAND_SIZE)        # 1
    f.append(1.0 if move.type == MoveType.BOMB else 0.0)  # 1
    f.append(1.0 if move.is_pass else 0.0)             # 1

    hand = state.hand()
    if move.is_pass:
        burn = progress = after_steps = keep = 0.0
    else:
        before = min_steps(hand, cfg)
        after = min_steps(hand - move.cards, cfg)
        burn = min(burn_cost(move), 20.0) / 10.0
        progress = max(-2.0, min(5.0, before - after)) / 5.0
        after_steps = min(after, C.HAND_SIZE) / C.HAND_SIZE
        keep = keep_lead_prob(state, state.current, move, cfg)
    f.extend([burn, progress, keep, after_steps])      # 4
    f.append(1.0 if (not move.is_pass and move.cards == hand) else 0.0)  # 1

    nxt = (state.current + 1) % 3
    f.append(1.0 if C.total(state.hands[nxt]) == 1 else 0.0)  # 1

    arr = np.asarray(f, dtype=np.float32)
    assert arr.shape[0] == MOVE_DIM, arr.shape
    return arr


def encode_move_set(state: GameState, moves: Sequence[Move],
                    cfg: RuleConfig = DEFAULT_RULES) -> np.ndarray:
    if not moves:
        return np.zeros((0, MOVE_DIM), dtype=np.float32)
    return np.stack([encode_move(state, m, cfg) for m in moves])


def encode_history(state: GameState, p: int,
                   steps: int = HISTORY_STEPS) -> np.ndarray:
    """返回 (steps, EVENT_DIM)，右侧为最近事件，左侧 padding。"""
    events = state.action_log[-steps:]
    out = np.zeros((steps, EVENT_DIM), dtype=np.float32)
    nxt = (p + 1) % 3
    prv = (p + 2) % 3
    off = steps - len(events)
    for i, (q, mv) in enumerate(events):
        row = off + i
        out[row, 0] = 1.0 if q == p else 0.0
        out[row, 1] = 1.0 if q == nxt else 0.0
        out[row, 2] = 1.0 if q == prv else 0.0
        out[row, 3] = float(mv.type) / float(max(_TYPES))
        out[row, 4] = 1.0 if mv.is_pass else 0.0
        out[row, 5] = min(mv.length, 12) / 12.0
        out[row, 6] = mv.max_rank / 15.0
        out[row, 7] = C.total(mv.cards) / C.HAND_SIZE
    return out


def encode(state: GameState, p: int,
           cfg: RuleConfig = DEFAULT_RULES) -> np.ndarray:
    """兼容旧调用名；新代码应显式使用 encode_state。"""
    return encode_state(state, p, cfg)
