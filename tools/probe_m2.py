"""M2 评测: 残局精确解 + 约束采样 是否值得。

运行: python tools/probe_m2.py [--deals 60] [--k 8] [--cards 8]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent, M1Agent, M2Agent
from src.arena.match import run_duplicate
from src.fastrun.config import DEFAULT_RULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    cfg = DEFAULT_RULES

    m2 = M2Agent(cfg, endgame_cards=args.cards, endgame_k=args.k, seed=1)
    r = run_duplicate(m2, [GreedyMinAgent(), GreedyMinAgent()],
                      n_deals=args.deals, cfg=cfg, seed0=args.seed)
    print(f"m2(cards={args.cards},k={args.k}) vs greedy  {r.win_rate:6.1%} "
          f"[{r.ci[0]:.1%},{r.ci[1]:.1%}]  score={r.avg_score:+6.2f}  {r.avg_ms:6.1f}ms")

    m2c = M2Agent(cfg, endgame_cards=args.cards, endgame_k=args.k, seed=3)
    r = run_duplicate(m2c, [M1Agent(cfg), M1Agent(cfg)],
                      n_deals=args.deals, cfg=cfg, seed0=args.seed)
    print(f"m2(cards={args.cards},k={args.k}) vs m1      {r.win_rate:6.1%} "
          f"[{r.ci[0]:.1%},{r.ci[1]:.1%}]  score={r.avg_score:+6.2f}  {r.avg_ms:6.1f}ms")


if __name__ == "__main__":
    main()
