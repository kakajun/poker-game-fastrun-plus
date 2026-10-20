""""竞速意识"变体：当我离走完比对手更近时，优先兑现牌权。

思路: 必压规则下，谁先手数领先谁就该抢牌权；落后时抢牌权反而烧牌。
运行: python tools/probe_race.py [--deals 80]
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


def make_policy(mode: int, safe_p: float = 0.85):
    def act(state, moves):
        ms = [m for m in moves if not m.is_pass] or moves
        if len(ms) == 1:
            return ms[0]
        hand = state.hand()
        base = min_steps(hand, CFG)
        ps = [(m, base - min_steps(hand - m.cards, CFG)) for m in ms]
        top = max(p for _, p in ps)
        pool = [m for m, p in ps if p == top]

        steps_after = {m: min_steps(hand - m.cards, CFG) for m in pool}
        opp_steps = min(min_steps(state.hands[q], CFG)
                        for q in range(3) if q != state.current)

        # 1) 一手可结束且能保权
        if mode >= 1:
            fin = [m for m in pool if steps_after[m] <= 1
                   and keep_lead_prob(state, state.current, m, CFG) >= safe_p]
            if fin:
                return max(fin, key=lambda m: (C.total(m.cards),
                                               -burn_cost(m), -m.max_rank))

        # 2) 竞速领先时优先保权
        if mode >= 2 and opp_steps >= steps_after[pool[0]]:
            safe = [m for m in pool
                    if keep_lead_prob(state, state.current, m, CFG) >= safe_p]
            if safe:
                return max(safe, key=lambda m: (C.total(m.cards),
                                                -burn_cost(m), -m.max_rank))

        # 3) 保权且便宜
        if mode >= 3:
            safe = [m for m in pool
                    if burn_cost(m) <= 2.0
                    and keep_lead_prob(state, state.current, m, CFG) >= safe_p]
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
    ap.add_argument("--deals", type=int, default=80)
    args = ap.parse_args()

    class Fn(Agent):
        def __init__(self, name, fn):
            self.name = name
            self.fn = fn

        def act(self, state, moves):
            return self.fn(state, moves)

    print(f"{'mode/safe_p':<14}{'vs random':>11}{'vs greedy':>11}{'score_g':>9}")
    for mode, sp in [(1, 0.8), (2, 0.8), (2, 0.7), (3, 0.8)]:
        a = Fn(f"m{mode}", make_policy(mode, sp))
        r1 = run_duplicate(a, [RandomAgent(1), RandomAgent(2)],
                           n_deals=args.deals, seed0=31)
        a2 = Fn(f"m{mode}", make_policy(mode, sp))
        r2 = run_duplicate(a2, [GreedyMinAgent(), GreedyMinAgent()],
                           n_deals=args.deals, seed0=31)
        print(f"{f'{mode}/{sp}':<14}{r1.win_rate:>10.1%}{r2.win_rate:>11.1%}"
              f"{r2.avg_score:>+9.2f}")


if __name__ == "__main__":
    main()
