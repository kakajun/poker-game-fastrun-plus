"""残局搜索（完美信息 max-n + 置换表）。

诚实性约定:
  * 节点预算耗尽时返回近似估值，但结果标记 `exact=False`；
  * 近似值绝不写入置换表；
  * 多世界比较时，每个根动作在每个世界都必须被估值，不能只统计“它曾经最优”的世界。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from ..eval.decompose import min_steps

ENDGAME_CARDS = 8
NODE_BUDGET = 40_000

Values = Tuple[float, float, float]


@dataclass
class SolveResult:
    move: Move
    values: Values
    exact: bool
    nodes: int


@dataclass
class WorldSolveResult:
    move: Move
    action_values: Dict[Move, float]
    exact: bool
    nodes: int


def _key(st: GameState) -> tuple:
    return (st.hands, st.current, st.last_play, st.last_player,
            st.tricks, st.bombs)


def _eval(st: GameState, cfg: RuleConfig) -> Values:
    """非终局估值: 谁离走完更近谁更可能赢（零和化）。"""
    v = []
    for p in range(3):
        steps = min_steps(st.hands[p], cfg)
        x = -6.0 * steps
        if p == st.current and st.last_play is None:
            x += 1.5                      # 首出权
        v.append(x)
    m = sum(v) / 3.0
    return (v[0] - m, v[1] - m, v[2] - m)


def _order(ms: List[Move]) -> List[Move]:
    return sorted(ms, key=lambda m: (-C.total(m.cards), m.max_rank))


def _search_child(child: GameState, cfg: RuleConfig, budget: int,
                  tt: Dict[tuple, Values]) -> Tuple[Values, bool, int]:
    """从 child 开始搜索；返回 (values, exact, nodes)。"""
    counter = [0]

    def rec(st: GameState) -> Tuple[Values, bool]:
        if st.over:
            return tuple(float(x) for x in st.scores), True
        counter[0] += 1
        if counter[0] > budget:
            return _eval(st, cfg), False
        k = _key(st)
        hit = tt.get(k)
        if hit is not None:
            return hit, True

        p = st.current
        best: Optional[Values] = None
        all_exact = True
        for m in _order(st.legal_moves(cfg)):
            val, ok = rec(st.apply(m, cfg))
            all_exact = all_exact and ok
            if best is None or val[p] > best[p]:
                best = val
            if counter[0] > budget:
                all_exact = False
                break
        if best is None:
            return _eval(st, cfg), False
        if all_exact:
            tt[k] = best
        return best, all_exact

    vals, exact = rec(child)
    return vals, exact, counter[0]


def solve_values(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
                 budget: int = NODE_BUDGET
                 ) -> Tuple[Dict[Move, Values], bool, int]:
    """给每个根动作独立预算估值，保证动作间可比较。"""
    ms = state.legal_moves(cfg)
    if len(ms) == 1:
        return {ms[0]: _eval(state, cfg)}, True, 0

    tt: Dict[tuple, Values] = {}
    out: Dict[Move, Values] = {}
    exact = True
    nodes = 0
    for m in _order(ms):
        child = state.apply(m, cfg)
        vals, ok, n = _search_child(child, cfg, budget, tt)
        out[m] = vals
        exact = exact and ok
        nodes += n
    return out, exact, nodes


def solve_with_status(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
                      budget: int = NODE_BUDGET) -> SolveResult:
    values, exact, nodes = solve_values(state, cfg, budget)
    p = state.current
    move = max(values.items(), key=lambda kv: kv[1][p])[0]
    return SolveResult(move=move, values=values[move],
                       exact=exact, nodes=nodes)


def solve(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
          budget: int = NODE_BUDGET) -> Tuple[Move, Values]:
    """兼容旧调用：只返回动作和该动作估值，不暴露 exact 状态。"""
    r = solve_with_status(state, cfg, budget)
    return r.move, r.values


def solve_worlds_with_status(state: GameState, worlds: List[GameState],
                             cfg: RuleConfig = DEFAULT_RULES,
                             budget: int = NODE_BUDGET) -> WorldSolveResult:
    """跨世界对每个动作取均值；所有世界都精确时才标记 exact。"""
    if not worlds:
        ms = state.legal_moves(cfg)
        return WorldSolveResult(ms[0], {ms[0]: 0.0}, False, 0)

    totals: Dict[Move, float] = {}
    counts: Dict[Move, int] = {}
    exact = True
    nodes = 0
    p = state.current
    for w in worlds:
        values, ok, n = solve_values(w, cfg, budget)
        exact = exact and ok
        nodes += n
        for m, vals in values.items():
            totals[m] = totals.get(m, 0.0) + vals[p]
            counts[m] = counts.get(m, 0) + 1

    if not totals:
        ms = state.legal_moves(cfg)
        return WorldSolveResult(ms[0], {ms[0]: 0.0}, False, nodes)
    action_values = {m: totals[m] / counts[m] for m in totals}
    move = max(action_values.items(), key=lambda kv: kv[1])[0]
    return WorldSolveResult(move, action_values, exact, nodes)


def solve_worlds(state: GameState, worlds: List[GameState],
                 cfg: RuleConfig = DEFAULT_RULES,
                 budget: int = NODE_BUDGET) -> Move:
    return solve_worlds_with_status(state, worlds, cfg, budget).move
