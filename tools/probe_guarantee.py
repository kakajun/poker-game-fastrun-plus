""""确定性保权"变体: 有证据/已出牌证明对手管不起时，优先用它把牌走掉。

与"概率保权"的区别: 这里只在 p_keep 严格 == 1.0 时生效，不含猜测成分。

运行: python tools/probe_guarantee.py [--deals 80]
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


def make_policy(burn_cap: float, safe_p: float = 1.0):
    def act(state, moves):
        ms = [m for m in moves if not m.is_pass] or moves
        if len(ms) == 1:
            return ms[0]
        hand = state.hand()
        base = min_steps(hand, CFG)
        ps = [(m, base - min_steps(hand - m.cards, CFG)) for m in ms]
        top = max(p for _, p in ps)
        pool = [m for m, p in ps if p == top]

        # 1) 一手结束 + 保权
        fin = [m for m in pool
               if min_steps(hand - m.cards, CFG) <= 1
               and keep_lead_prob(state, state.current, m, CFG) >= 0.8]
        if fin:
            return max(fin, key=lambda m: (C.total(m.cards), -burn_cost(m), -m.max_rank))

        # 2) 确定性保权（含过牌证据）
        guaranteed = [m for m in pool
                      if burn_cost(m) <= burn_cap
                      and keep_lead_prob(state, state.current, m, CFG) >= safe_p]
        if guaranteed:
            return max(guaranteed, key=lambda m: (C.total(m.cards),
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

    print(f"{'burn_cap':<10}{'vs random':>11}{'vs greedy':>11}{'score_g':>9}")
    for cap in (0.0, 2.0, 3.6, 99.0):
        a = Fn(f"g{cap}", make_policy(cap))
        r1 = run_duplicate(a, [RandomAgent(1), RandomAgent(2)],
                           n_deals=args.deals, seed0=43)
        a2 = Fn(f"g{cap}", make_policy(cap))
        r2 = run_duplicate(a2, [GreedyMinAgent(), GreedyMinAgent()],
                           n_deals=args.deals, seed0=43)
        print(f"{cap:<10}{r1.win_rate:>10.1%}{r2.win_rate:>11.1%}{r2.avg_score:>+9.2f}")


if __name__ == "__main__":
    main()
