"""残局搜索（完美信息 max-n + 置换表）。

诚实性约定:
  * 节点预算耗尽时返回近似估值，但结果标记 `exact=False`；
  * 近似值绝不写入置换表；
  * 多世界比较时，每个根动作在每个世界都必须被估值，不能只统计“它曾经最优”的世界。

决策目标:
  * 每个值都是 `(win_prob, net_score)`；
  * 默认净分优先：在胜率不低于本节点最佳值 `win_guard` 的动作中取净分最高者；
  * `win_primary` 模式用于消融/回滚。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from ..eval.decompose import min_steps
from ..eval.objective import ObjectiveConfig
from ..eval.score_ev import zero_play_opponents

ENDGAME_CARDS = 8
NODE_BUDGET = 40_000

Pair = Tuple[float, float]          # (win_prob, net_score)
Values = Tuple[Pair, Pair, Pair]    # 三个玩家各自的 Pair


@dataclass
class SolveResult:
    move: Move
    values: Pair
    exact: bool
    nodes: int


@dataclass
class WorldSolveResult:
    move: Move
    action_values: Dict[Move, Pair]
    exact: bool
    nodes: int


def _key(st: GameState) -> tuple:
    return (st.hands, st.current, st.last_play, st.last_player,
            st.tricks, st.bombs)


def _terminal_values(st: GameState) -> Values:
    return tuple(
        (1.0 if st.winner == p else 0.0, float(st.scores[p]))
        for p in range(3)
    )


def _win_proxy(st: GameState, p: int, cfg: RuleConfig) -> float:
    mine = min_steps(st.hands[p], cfg)
    opp = min(min_steps(st.hands[q], cfg) for q in range(3) if q != p)
    # 手数差越大越可能先走完；1.5 手约对应一个明显的领先档位。
    return 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, (mine - opp) / 1.5))))


def _eval(st: GameState, cfg: RuleConfig,
          objective: ObjectiveConfig) -> Values:
    """非终局近似：同时给出胜率代理和净分代理，避免预算耗尽后退回纯手数。"""
    wins = [_win_proxy(st, p, cfg) for p in range(3)]
    out = []
    for p in range(3):
        win = wins[p]
        opp_cards = sum(C.total(st.hands[q]) for q in range(3) if q != p)
        own_cards = C.total(st.hands[p])
        base = win * opp_cards - (1.0 - win) * own_cards
        zero_opp = zero_play_opponents(st.tricks, p)
        spring = (objective.spring_weight * zero_opp * cfg.spring_score
                  * win * win)
        bomb = objective.bomb_weight * float(st.bombs[p])
        out.append((win, base + spring + bomb))
    return tuple(out)  # type: ignore[return-value]


def _pick(values: List[Values], p: int,
          objective: ObjectiveConfig) -> Values:
    """按当前玩家目标从候选值中选一个。"""
    if not values:
        raise ValueError("empty value list")
    if not objective.score_primary:
        return max(values, key=lambda v: (v[p][0], v[p][1]))
    best_win = max(v[p][0] for v in values)
    cutoff = best_win - objective.win_guard
    eligible = [v for v in values if v[p][0] >= cutoff]
    return max(eligible, key=lambda v: (v[p][1], v[p][0]))


def _order(ms: List[Move]) -> List[Move]:
    return sorted(ms, key=lambda m: (-C.total(m.cards), m.max_rank))


def _search_child(child: GameState, cfg: RuleConfig, budget: int,
                  tt: Dict[tuple, Values],
                  objective: ObjectiveConfig) -> Tuple[Values, bool, int]:
    """从 child 开始搜索；返回 (values, exact, nodes)。"""
    counter = [0]

    def rec(st: GameState) -> Tuple[Values, bool]:
        if st.over:
            return _terminal_values(st), True
        counter[0] += 1
        if counter[0] > budget:
            return _eval(st, cfg, objective), False
        k = _key(st)
        hit = tt.get(k)
        if hit is not None:
            return hit, True

        p = st.current
        children: List[Values] = []
        all_exact = True
        for m in _order(st.legal_moves(cfg)):
            val, ok = rec(st.apply(m, cfg))
            children.append(val)
            all_exact = all_exact and ok
            if counter[0] > budget:
                all_exact = False
                break
        if not children:
            return _eval(st, cfg, objective), False
        best = _pick(children, p, objective)
        if all_exact:
            tt[k] = best
        return best, all_exact

    vals, exact = rec(child)
    return vals, exact, counter[0]


def solve_values(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
                 budget: int = NODE_BUDGET,
                 objective: Optional[ObjectiveConfig] = None
                 ) -> Tuple[Dict[Move, Values], bool, int]:
    """给每个根动作独立预算估值，保证动作间可比较。"""
    objective = objective or ObjectiveConfig()
    ms = state.legal_moves(cfg)
    if len(ms) == 1:
        child = state.apply(ms[0], cfg)
        if child.over:
            return {ms[0]: _terminal_values(child)}, True, 0
        return {ms[0]: _eval(state, cfg, objective)}, False, 0

    tt: Dict[tuple, Values] = {}
    out: Dict[Move, Values] = {}
    exact = True
    nodes = 0
    for m in _order(ms):
        child = state.apply(m, cfg)
        vals, ok, n = _search_child(
            child, cfg, budget, tt, objective)
        out[m] = vals
        exact = exact and ok
        nodes += n
    return out, exact, nodes


def _select_action(values: Dict[Move, Values], p: int,
                   objective: ObjectiveConfig) -> Move:
    if not values:
        raise ValueError("empty action values")
    if objective.score_primary:
        best_win = max(v[p][0] for v in values.values())
        cutoff = best_win - objective.win_guard
        eligible = {m: v for m, v in values.items()
                    if v[p][0] >= cutoff}
        return max(eligible, key=lambda m: (
            eligible[m][p][1], eligible[m][p][0]))
    return max(values, key=lambda m: (values[m][p][0], values[m][p][1]))


def solve_with_status(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
                      budget: int = NODE_BUDGET,
                      objective: Optional[ObjectiveConfig] = None
                      ) -> SolveResult:
    objective = objective or ObjectiveConfig()
    values, exact, nodes = solve_values(state, cfg, budget, objective)
    p = state.current
    move = _select_action(values, p, objective)
    return SolveResult(move=move, values=values[move][p],
                       exact=exact, nodes=nodes)


def solve(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
          budget: int = NODE_BUDGET,
          objective: Optional[ObjectiveConfig] = None
          ) -> Tuple[Move, Pair]:
    """兼容旧调用：只返回动作和该动作估值，不暴露 exact 状态。"""
    r = solve_with_status(state, cfg, budget, objective)
    return r.move, r.values


def solve_worlds_with_status(state: GameState, worlds: List[GameState],
                             cfg: RuleConfig = DEFAULT_RULES,
                             budget: int = NODE_BUDGET,
                             objective: Optional[ObjectiveConfig] = None
                             ) -> WorldSolveResult:
    """跨世界对每个动作取均值；所有世界都精确时才标记 exact。"""
    objective = objective or ObjectiveConfig()
    ms = state.legal_moves(cfg)
    if not worlds:
        return WorldSolveResult(ms[0], {ms[0]: (0.0, 0.0)}, False, 0)

    win_totals: Dict[Move, float] = {}
    score_totals: Dict[Move, float] = {}
    counts: Dict[Move, int] = {}
    exact = True
    nodes = 0
    p = state.current
    for w in worlds:
        values, ok, n = solve_values(w, cfg, budget, objective)
        exact = exact and ok
        nodes += n
        for m, vals in values.items():
            win_totals[m] = win_totals.get(m, 0.0) + vals[p][0]
            score_totals[m] = score_totals.get(m, 0.0) + vals[p][1]
            counts[m] = counts.get(m, 0) + 1

    if not win_totals:
        return WorldSolveResult(ms[0], {ms[0]: (0.0, 0.0)}, False, nodes)
    action_values = {
        m: (win_totals[m] / counts[m], score_totals[m] / counts[m])
        for m in win_totals
    }
    if objective.score_primary:
        best_win = max(v[0] for v in action_values.values())
        cutoff = best_win - objective.win_guard
        eligible = {m: v for m, v in action_values.items()
                    if v[0] >= cutoff}
        move = max(eligible, key=lambda m: (eligible[m][1], eligible[m][0]))
    else:
        move = max(action_values,
                   key=lambda m: (action_values[m][0], action_values[m][1]))
    return WorldSolveResult(move, action_values, exact, nodes)


def solve_worlds(state: GameState, worlds: List[GameState],
                 cfg: RuleConfig = DEFAULT_RULES,
                 budget: int = NODE_BUDGET,
                 objective: Optional[ObjectiveConfig] = None) -> Move:
    return solve_worlds_with_status(
        state, worlds, cfg, budget, objective).move
