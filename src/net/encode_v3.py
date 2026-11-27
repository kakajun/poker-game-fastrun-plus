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
from ..eval.score_ev import zero_play_opponents
from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState

FEATURE_VERSION = "v4"
ACTION_VERSION = "v4"
HISTORY_STEPS = 128
EVENT_DIM = 8
MOVE_DIM = 28
STATE_DIM = 97
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

    # tricks/bombs 统一按 p/nxt/prv 相对座位编码，避免绝对座位偏置。
    f.extend(state.tricks[q] / C.HAND_SIZE for q in (p, nxt, prv))  # 62..64
    f.extend(max(-60, min(60, state.bombs[q])) / 60.0
             for q in (p, nxt, prv))                    # 65..67

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

    # v4：春天存活、可兑现基础分和控制点，均为公开信息。
    zero_opp = zero_play_opponents(state.tricks, p)
    opp_cards = sum(C.total(state.hands[q]) for q in (nxt, prv))
    f.append(zero_opp / 2.0)                           # 92
    f.append(1.0 if zero_opp == 2 else 0.0)            # 93
    f.append(opp_cards / (2.0 * C.HAND_SIZE))          # 94
    f.append(1.0 if C.total(state.hands[prv]) == 1 else 0.0)  # 95
    f.append(min(60, zero_opp * cfg.spring_score) / 60.0)  # 96

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
    p = state.current
    if move.is_pass:
        burn = progress = after_steps = keep = 0.0
        bomb_delta = 0.0
        finish = False
    else:
        before = min_steps(hand, cfg)
        after = min_steps(hand - move.cards, cfg)
        burn = min(burn_cost(move), 20.0) / 10.0
        progress = max(-2.0, min(5.0, before - after)) / 5.0
        after_steps = min(after, C.HAND_SIZE) / C.HAND_SIZE
        keep = keep_lead_prob(state, state.current, move, cfg)
        child = state.apply(move, cfg)
        bomb_delta = float(child.bombs[p] - state.bombs[p])
        finish = (hand - move.cards) == 0
    f.extend([burn, progress, keep, after_steps])      # 4
    f.append(1.0 if finish else 0.0)                   # 1

    nxt = (state.current + 1) % 3
    prv = (state.current + 2) % 3
    f.append(1.0 if C.total(state.hands[nxt]) == 1 else 0.0)  # 1

    # v4：炸弹立即分差、春天存活和兑现概率。
    zero_opp = zero_play_opponents(state.tricks, p)
    spring_finish = (zero_opp * cfg.spring_score) if finish else 0
    spring_potential = zero_opp * cfg.spring_score * keep
    opp_cards = sum(C.total(state.hands[q]) for q in (nxt, prv))
    f.append(max(-30.0, min(30.0, bomb_delta)) / 30.0)  # 1
    f.append(zero_opp / 2.0)                           # 1
    f.append(min(60, spring_finish) / 60.0)            # 1
    f.append(min(60.0, spring_potential) / 60.0)       # 1
    f.append(1.0 if (zero_opp > 0 and keep >= 0.8) else 0.0)  # 1
    f.append(opp_cards / (2.0 * C.HAND_SIZE))          # 1

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
