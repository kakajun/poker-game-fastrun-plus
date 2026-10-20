"""M1 竞技场: 同牌三座位对局 + Wilson 置信区间。

运行: python tools/arena_m1.py [--deals 200]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent, M1Agent, RandomAgent
from src.arena.match import run_duplicate
from src.fastrun.config import DEFAULT_RULES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = DEFAULT_RULES
    n = args.deals

    print(f"M1 arena | deals={n} (x3 seats) | rules={cfg.hash()}")
    print("-" * 96)

    reports = []
    reports.append(run_duplicate(M1Agent(cfg), [RandomAgent(1), RandomAgent(2)],
                                 n_deals=n, cfg=cfg, seed0=args.seed))
    print(reports[-1])

    reports.append(run_duplicate(M1Agent(cfg), [GreedyMinAgent(), GreedyMinAgent()],
                                 n_deals=n, cfg=cfg, seed0=args.seed))
    print(reports[-1])

    reports.append(run_duplicate(GreedyMinAgent(), [RandomAgent(1), RandomAgent(2)],
                                 n_deals=n, cfg=cfg, seed0=args.seed))
    print(reports[-1])

    reports.append(run_duplicate(RandomAgent(9), [RandomAgent(1), RandomAgent(2)],
                                 n_deals=n, cfg=cfg, seed0=args.seed))
    print(reports[-1])

    print("-" * 96)
    m1_vs_rand = reports[0].win_rate
    m1_vs_greedy = reports[1].win_rate
    print(f"GATE M1 vs random  >= 80.0% : {m1_vs_rand:.1%}  "
          f"{'PASS' if m1_vs_rand >= 0.80 else 'FAIL'}")
    print(f"GATE M1 vs greedy  >= 55.0% : {m1_vs_greedy:.1%}  "
          f"{'PASS' if m1_vs_greedy >= 0.55 else 'FAIL'}")


if __name__ == "__main__":
    main()
