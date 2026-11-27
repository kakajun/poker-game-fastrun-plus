"""PIMC 雏形评测：搜索能否跨过"打不过双贪心"这道坎。

运行: python tools/probe_search.py [--deals 60]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import Agent, GreedyMinAgent, SearchAgent
from src.arena.match import run_duplicate
from src.eval.decompose import min_steps
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES

CFG = DEFAULT_RULES


class ProgRankAgent(Agent):
    name = "prog_rank"

    def act(self, state, moves):
        ms = [m for m in moves if not m.is_pass] or moves
        if len(ms) == 1:
            return ms[0]
        base = min_steps(state.hand(), CFG)
        prog = [(m, base - min_steps(state.hand() - m.cards, CFG)) for m in ms]
        top = max(p for _, p in prog)
        pool = [m for m, p in prog if p == top]
        return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    print(f"search: k={args.k}, deals={args.deals}")
    for label, hero, villains in [
        ("search vs greedy", SearchAgent(CFG, k=args.k), [GreedyMinAgent(), GreedyMinAgent()]),
        ("search vs prog_rank", SearchAgent(CFG, k=args.k), [ProgRankAgent(), ProgRankAgent()]),
        ("prog_rank vs greedy", ProgRankAgent(), [GreedyMinAgent(), GreedyMinAgent()]),
    ]:
        r = run_duplicate(hero, villains, n_deals=args.deals,
                          cfg=CFG, seed0=args.seed)
        print(f"{label:<22} n={r.n:4d} win={r.win_rate:6.1%} "
              f"[{r.ci[0]:.1%},{r.ci[1]:.1%}] score={r.avg_score:+6.2f} "
              f"{r.avg_ms:6.2f}ms")


if __name__ == "__main__":
    main()
