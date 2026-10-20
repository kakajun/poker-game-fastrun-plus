"""残局精确解（完美信息 max-n 搜索 + 置换表）。

用法: 总剩牌 <= ENDGAME_CARDS 时，对采样出的世界做精确求解，取平均最好的动作。
为什么值得: 残局每一步的收益差距远大于中局启发式误差，而搜索空间随张数指数
下降（<=8 张时通常几千节点）。这是"能打"最直接的兑现方式。
"""

from typing import Dict, List, Optional, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from ..eval.decompose import min_steps

ENDGAME_CARDS = 8
NODE_BUDGET = 40_000


def _key(st: GameState) -> tuple:
    return (st.hands, st.current, st.last_play, st.last_player,
            st.tricks, st.bombs)


def _eval(st: GameState, cfg: RuleConfig) -> Tuple[float, float, float]:
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


def solve(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
          budget: int = NODE_BUDGET) -> Tuple[Move, Tuple[float, float, float]]:
    """返回 (对当前玩家最优的动作, 三人估值)。"""
    ms = state.legal_moves(cfg)
    if len(ms) == 1:
        return ms[0], _eval(state, cfg)

    tt: Dict[tuple, Tuple[float, float, float]] = {}
    counter = [0]

    def rec(st: GameState) -> Tuple[float, float, float]:
        if st.over:
            return tuple(float(x) for x in st.scores)
        counter[0] += 1
        if counter[0] > budget:
            return _eval(st, cfg)
        k = _key(st)
        hit = tt.get(k)
        if hit is not None:
            return hit

        p = st.current
        best: Optional[Tuple[float, float, float]] = None
        for m in _order(st.legal_moves(cfg)):
            val = rec(st.apply(m, cfg))
            if best is None or val[p] > best[p]:
                best = val
                if counter[0] > budget:
                    break
        if best is None:
            best = _eval(st, cfg)
        tt[k] = best
        return best

    best_move = ms[0]
    best_val: Optional[Tuple[float, float, float]] = None
    p = state.current
    for m in _order(ms):
        val = rec(state.apply(m, cfg))
        if best_val is None or val[p] > best_val[p]:
            best_val = val
            best_move = m
        if counter[0] > budget:
            break
    return best_move, (best_val if best_val is not None else _eval(state, cfg))


def solve_worlds(state: GameState, worlds: List[GameState],
                 cfg: RuleConfig = DEFAULT_RULES,
                 budget: int = NODE_BUDGET) -> Move:
    """多个确定性化世界里分别求解，取平均估值最高的动作（投票）。"""
    totals: Dict[Move, List[float]] = {}
    for w in worlds:
        mv, val = solve(w, cfg, budget)
        totals.setdefault(mv, []).append(val[w.current])
    if not totals:
        return state.legal_moves(cfg)[0]

    def key(item):
        mv, vals = item
        return (sum(vals) / len(vals), len(vals))

    return max(totals.items(), key=key)[0]
