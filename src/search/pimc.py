"""PIMC 雏形：只搜"这一轮牌"打完，叶子用拆牌手数估值。

为什么先搜一轮而不是整局:
  规则是必压 ⇒ 一轮之内所有人都在被迫加注，谁赢下这一轮谁就拿到下一轮的
  首出权。首出权 + 剩余手牌手数基本决定了后续走势，所以"本轮结果 + 静态估值"
  的性价比远高于完整 rollout。

成本: K 个确定性化 × 候选数 × 一轮内 2~6 步，实测 5~15ms/决策。
"""

import math
import random
from dataclasses import replace
from typing import List, Optional, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState
from ..infer.sampler import SamplingError, sample_world
from ..eval.decompose import min_steps
from ..eval.objective import ObjectiveConfig
from ..eval.score_ev import zero_play_opponents


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


def _value(state: GameState, me: int, cfg: RuleConfig,
           objective: ObjectiveConfig) -> Tuple[float, float]:
    """返回 (胜率代理, 净分代理)；终局使用真实 scores。"""
    if state.over:
        return (1.0 if state.winner == me else 0.0,
                float(state.scores[me]))
    mine = min_steps(state.hands[me], cfg)
    opp_steps = [min_steps(state.hands[q], cfg)
                 for q in range(3) if q != me]
    opp = min(opp_steps)
    win = 1.0 / (1.0 + math.exp(
        max(-60.0, min(60.0, (mine - opp) / 1.5))))
    own_cards = C.total(state.hands[me])
    opp_cards = sum(C.total(state.hands[q]) for q in range(3) if q != me)
    base = win * opp_cards - (1.0 - win) * own_cards
    zero_opp = zero_play_opponents(state.tricks, me)
    spring = (objective.spring_weight * zero_opp * cfg.spring_score
              * win * win)
    bomb = objective.bomb_weight * float(state.bombs[me])
    if state.current == me and state.last_play is None:
        base += 1.0
    return win, base + spring + bomb


def _candidates(state: GameState, ms: List[Move],
                cfg: RuleConfig, limit: int,
                objective: ObjectiveConfig) -> List[Move]:
    hand = state.hand()
    if any(m.cards == hand and not m.is_pass for m in ms):
        return [m for m in ms if m.cards == hand and not m.is_pass]
    pool = [m for m in ms if not m.is_pass]
    if not pool:
        return ms
    base = min_steps(hand, cfg)
    scored = []
    for m in pool:
        after = min_steps(hand - m.cards, cfg)
        prog = base - after
        scored.append((after, prog, -m.max_rank, C.total(m.cards), m))
    min_steps_after = min(x[0] for x in scored)
    limit_steps = min_steps_after + objective.max_extra_steps
    safe = [x for x in scored
            if x[0] <= limit_steps
            or (x[4].type == MoveType.BOMB and x[0] <= limit_steps + 1)]
    if not safe:
        safe = [x for x in scored if x[0] == min_steps_after]
    safe.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [x[4] for x in safe[:limit]]


def choose(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
           k: int = 8, limit: int = 8, seed: int = 0,
           objective: Optional[ObjectiveConfig] = None) -> Move:
    objective = objective or ObjectiveConfig()
    ms = state.legal_moves(cfg)
    if len(ms) <= 1:
        return ms[0]
    cands = _candidates(state, ms, cfg, limit, objective)
    if len(cands) == 1:
        return cands[0]

    me = state.current
    rng = random.Random(seed)
    worlds: List[GameState] = []
    try:
        for _ in range(k):
            worlds.append(sample_world(state, me, rng))
    except SamplingError:
        # 不拿不满足 Pass 证据的世界做搜索；退回启发式候选首位。
        return cands[0]

    wins: List[float] = []
    scores: List[float] = []
    for m in cands:
        win_total = 0.0
        score_total = 0.0
        for w in worlds:
            st = w.apply(m, cfg)
            st = _rollout_trick(st, cfg)
            win, score = _value(st, me, cfg, objective)
            win_total += win
            score_total += score
        wins.append(win_total / k)
        scores.append(score_total / k)
    return cands[objective.select_index(wins, scores)]
