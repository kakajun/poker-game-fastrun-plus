"""策略原型对比：搞清楚"什么打法真的赢"。

每个变体都是纯规则，不带权重，便于定位有效成分。
运行: python tools/probe_strategies.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import Agent, GreedyMinAgent
from src.arena.match import run_duplicate
from src.eval.control import burn_cost, keep_lead_prob
from src.eval.decompose import min_steps
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState

CFG = DEFAULT_RULES


class FnAgent(Agent):
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def act(self, state, moves):
        return self.fn(state, moves)


def _nonpass(moves):
    return [m for m in moves if not m.is_pass] or moves


def _progress(state, m):
    return min_steps(state.hand(), CFG) - min_steps(state.hand() - m.cards, CFG)


# ---- 变体 ----------------------------------------------------------

def greedy(state, moves):
    ms = _nonpass(moves)
    normal = [m for m in ms if m.type.name != "BOMB"] or ms
    return min(normal, key=lambda m: (m.max_rank, -C.total(m.cards)))


def progress_then_rank(state, moves):
    """先保证不拆牌（progress=1），再挑点数最小的。"""
    ms = _nonpass(moves)
    if len(ms) == 1:
        return ms[0]
    ps = [(m, _progress(state, m)) for m in ms]
    top = max(p for _, p in ps)
    pool = [m for m, p in ps if p == top]
    return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


def progress_then_burn(state, moves):
    """先保证不拆牌，再挑最省大牌的。"""
    ms = _nonpass(moves)
    if len(ms) == 1:
        return ms[0]
    ps = [(m, _progress(state, m)) for m in ms]
    top = max(p for _, p in ps)
    pool = [m for m, p in ps if p == top]
    return min(pool, key=lambda m: (burn_cost(m), m.max_rank, -C.total(m.cards)))


def progress_then_cards(state, moves):
    """先保证不拆牌，再挑一次出得多的（组合优先）。"""
    ms = _nonpass(moves)
    if len(ms) == 1:
        return ms[0]
    ps = [(m, _progress(state, m)) for m in ms]
    top = max(p for _, p in ps)
    pool = [m for m, p in ps if p == top]
    return min(pool, key=lambda m: (-C.total(m.cards), m.max_rank))


def lead_combo_beat_min(state, moves):
    """首出优先组合（出得多），跟牌只挑最小能管的。"""
    ms = _nonpass(moves)
    if len(ms) == 1:
        return ms[0]
    if state.last_play is None:
        pool = [m for m in ms if m.type.name != "BOMB"] or ms
        return max(pool, key=lambda m: (C.total(m.cards), -m.max_rank))
    normal = [m for m in ms if m.type.name != "BOMB"] or ms
    return min(normal, key=lambda m: (m.max_rank, -C.total(m.cards)))


def _progress_pool(state, moves):
    ms = _nonpass(moves)
    if len(ms) == 1:
        return ms
    ps = [(m, _progress(state, m)) for m in ms]
    top = max(p for _, p in ps)
    return [m for m, p in ps if p == top]


def prog_rank_ctrl(state, moves):
    """prog_rank + 若某候选"对手确定管不起"，优先用它把牌走掉。"""
    pool = _progress_pool(state, moves)
    safe = [m for m in pool if keep_lead_prob(state, state.current, m, CFG) >= 0.85]
    if safe and state.last_play is None:
        return max(safe, key=lambda m: (C.total(m.cards), -m.max_rank))
    return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


def prog_rank_hold(state, moves):
    """prog_rank + 首出时尽量不动 2/A/K（留作夺回牌权）。"""
    pool = _progress_pool(state, moves)
    if state.last_play is None:
        cheap = [m for m in pool if burn_cost(m) == 0.0]
        if cheap:
            pool = cheap
    return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


def prog_rank_ctrl_hold(state, moves):
    pool = _progress_pool(state, moves)
    if state.last_play is None:
        safe = [m for m in pool
                if keep_lead_prob(state, state.current, m, CFG) >= 0.85]
        if safe:
            return max(safe, key=lambda m: (C.total(m.cards), -m.max_rank))
        cheap = [m for m in pool if burn_cost(m) == 0.0]
        if cheap:
            pool = cheap
    return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


VARIANTS = [
    ("greedy", greedy),
    ("prog_rank", progress_then_rank),
    ("prog_ctrl", prog_rank_ctrl),
    ("prog_hold", prog_rank_hold),
    ("prog_ctrl_hold", prog_rank_ctrl_hold),
]


def main():
    n = 60
    print(f"{'agent':<12}{'vs greedy':>11}{'score_g':>9}")
    for name, fn in VARIANTS:
        a2 = FnAgent(name, fn)
        r2 = run_duplicate(a2, [GreedyMinAgent(), GreedyMinAgent()], n_deals=n, seed0=5)
        print(f"{name:<12}{r2.win_rate:>11.1%}{r2.avg_score:>+9.2f}")

    # 变体之间互相对打
    print("\nhead-to-head (row agent vs column agents, 3 座位轮转):")
    agents = [(n, FnAgent(n, f)) for n, f in VARIANTS[:4]]
    print(f"{'':<12}" + "".join(f"{n:>12}" for n, _ in agents))
    for name, ag in agents:
        row = f"{name:<12}"
        for oname, opp in agents:
            if name == oname:
                row += f"{'-':>12}"
                continue
            r = run_duplicate(FnAgent(name, dict(VARIANTS)[name]),
                              [FnAgent(oname, dict(VARIANTS)[oname]),
                               FnAgent(oname, dict(VARIANTS)[oname])],
                              n_deals=60, seed0=7)
            row += f"{r.win_rate:>11.1%} "
        print(row)


if __name__ == "__main__":
    main()
