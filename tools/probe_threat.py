"""威胁感知变体：对手只剩 1~3 张时，首出必须以"对手管不起"为第一优先。

运行: python tools/probe_threat.py [--deals 60]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import Agent, GreedyMinAgent, RandomAgent
from src.arena.match import run_duplicate
from src.eval.control import burn_cost, keep_lead_prob
from src.eval.decompose import min_steps
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES

CFG = DEFAULT_RULES


def _pool(state, moves):
    ms = [m for m in moves if not m.is_pass] or moves
    if len(ms) == 1:
        return ms
    base = min_steps(state.hand(), CFG)
    ps = [(m, base - min_steps(state.hand() - m.cards, CFG)) for m in ms]
    top = max(p for _, p in ps)
    return [m for m, p in ps if p == top]


def make_policy(threat_n: int):
    def act(state, moves):
        pool = _pool(state, moves)
        if len(pool) == 1:
            return pool[0]
        opp_min = min(C.total(state.hands[q]) for q in range(3)
                      if q != state.current)
        if state.last_play is None:
            if opp_min <= threat_n:
                return max(pool, key=lambda m: (
                    keep_lead_prob(state, state.current, m, CFG),
                    -m.max_rank, C.total(m.cards)))
            cheap = [m for m in pool if burn_cost(m) == 0.0]
            if cheap:
                pool = cheap
        return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=60)
    args = ap.parse_args()

    class Fn(Agent):
        def __init__(self, name, fn):
            self.name = name
            self.fn = fn

        def act(self, state, moves):
            return self.fn(state, moves)

    print(f"{'threat_n':<10}{'vs random':>11}{'vs greedy':>11}{'score_g':>9}")
    for t in (0, 1, 2, 3):
        a = Fn(f"t{t}", make_policy(t))
        r1 = run_duplicate(a, [RandomAgent(1), RandomAgent(2)],
                           n_deals=args.deals, seed0=13)
        a2 = Fn(f"t{t}", make_policy(t))
        r2 = run_duplicate(a2, [GreedyMinAgent(), GreedyMinAgent()],
                           n_deals=args.deals, seed0=13)
        print(f"{t:<10}{r1.win_rate:>10.1%}{r2.win_rate:>11.1%}{r2.avg_score:>+9.2f}")


if __name__ == "__main__":
    main()
