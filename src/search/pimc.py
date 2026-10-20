"""PIMC 雏形：只搜"这一轮牌"打完，叶子用拆牌手数估值。

为什么先搜一轮而不是整局:
  规则是必压 ⇒ 一轮之内所有人都在被迫加注，谁赢下这一轮谁就拿到下一轮的
  首出权。首出权 + 剩余手牌手数基本决定了后续走势，所以"本轮结果 + 静态估值"
  的性价比远高于完整 rollout。

成本: K 个确定性化 × 候选数 × 一轮内 2~6 步，实测 5~15ms/决策。
"""

import random
from dataclasses import replace
from typing import List, Optional, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState
from ..infer.sampler import sample_world
from ..eval.decompose import min_steps


def _greedy_choice(state: GameState, ms: List[Move]) -> Move:
    pool = [m for m in ms if not m.is_pass]
    if not pool:
        return ms[0]
    normal = [m for m in pool if m.type != MoveType.BOMB]
    return min(normal or pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


def _rollout_trick(state: GameState, cfg: RuleConfig) -> GameState:
    st = state
    guard = 0
    while not st.over and st.last_play is not None:
        guard += 1
        if guard > 30:
            break
        ms = st.legal_moves(cfg)
        st = st.apply(_greedy_choice(st, ms), cfg)
    return st


def _value(state: GameState, me: int, cfg: RuleConfig) -> float:
    if state.over:
        rem = C.total(state.hands[me])
        return 200.0 if state.winner == me else -200.0 - rem
    mine = min_steps(state.hands[me], cfg)
    opp = [min_steps(state.hands[q], cfg) for q in range(3) if q != me]
    v = -1.0 * mine + 0.5 * (sum(opp) / 2.0)
    if state.current == me and state.last_play is None:
        v += 1.0        # 我拿到首出权
    return v


def _candidates(state: GameState, ms: List[Move],
                cfg: RuleConfig, limit: int) -> List[Move]:
    hand = state.hand()
    if any(m.cards == hand and not m.is_pass for m in ms):
        return [m for m in ms if m.cards == hand and not m.is_pass]
    pool = [m for m in ms if not m.is_pass]
    if not pool:
        return ms
    base = min_steps(hand, cfg)
    scored = []
    for m in pool:
        prog = base - min_steps(hand - m.cards, cfg)
        scored.append((prog, -m.max_rank, C.total(m.cards), m))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    top_prog = scored[0][0]
    out = [x[3] for x in scored if x[0] == top_prog]
    return out[:limit] if out else pool[:limit]


def choose(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
           k: int = 8, limit: int = 8, seed: int = 0) -> Move:
    ms = state.legal_moves(cfg)
    if len(ms) <= 1:
        return ms[0]
    cands = _candidates(state, ms, cfg, limit)
    if len(cands) == 1:
        return cands[0]

    me = state.current
    rng = random.Random(seed)
    worlds: List[GameState] = []
    for _ in range(k):
        worlds.append(sample_world(state, me, rng))

    best_val = -1e18
    best: Optional[Move] = None
    for m in cands:
        total = 0.0
        for w in worlds:
            st = w.apply(m, cfg)
            st = _rollout_trick(st, cfg)
            total += _value(st, me, cfg)
        avg = total / k
        if avg > best_val:
            best_val = avg
            best = m
    return best if best is not None else cands[0]
