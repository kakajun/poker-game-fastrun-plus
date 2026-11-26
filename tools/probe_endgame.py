"""残局兑现型变体：只有当"保权"能把游戏直接结束时才为它付大牌成本。

规则: 若某候选打完只剩 <= N 手，且对手确定管不起（p_keep 高），优先兑现。

运行: python tools/probe_endgame.py [--deals 60]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import Agent, GreedyMinAgent
from src.arena.match import run_duplicate
from src.eval.control import burn_cost, keep_lead_prob
from src.eval.decompose import min_steps
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES

CFG = DEFAULT_RULES


def make_policy(finish_steps: int, safe_p: float = 0.85):
    def act(state, moves):
        ms = [m for m in moves if not m.is_pass] or moves
        if len(ms) == 1:
            return ms[0]
        hand = state.hand()
        base = min_steps(hand, CFG)
        ps = [(m, base - min_steps(hand - m.cards, CFG)) for m in ms]
        top = max(p for _, p in ps)
        pool = [m for m, p in ps if p == top]

        if finish_steps > 0:
            safe = []
            for m in pool:
                if min_steps(hand - m.cards, CFG) <= finish_steps:
                    if keep_lead_prob(state, state.current, m, CFG) >= safe_p:
                        safe.append(m)
            if safe:
                return max(safe, key=lambda m: (C.total(m.cards),
                                                -burn_cost(m), -m.max_rank))

        if state.last_play is None:
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

    print(f"{'finish/safe_p':<16}{'vs greedy':>11}{'score_g':>9}")
    for n, sp in [(0, 0.85), (1, 0.7), (1, 0.8), (1, 0.9),
                  (2, 0.8), (2, 0.9)]:
        a2 = Fn(f"e{n}", make_policy(n, sp))
        r2 = run_duplicate(a2, [GreedyMinAgent(), GreedyMinAgent()],
                           n_deals=args.deals, seed0=21)
        print(f"{f'{n}/{sp}':<16}{r2.win_rate:>11.1%}{r2.avg_score:>+9.2f}")


if __name__ == "__main__":
    main()
