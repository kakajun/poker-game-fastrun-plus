"""Q 网络评测（M3-A'）。

运行: python tools/arena_q.py --deals 150
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent, M1Agent, M2Agent, QAgent
from src.arena.match import run_duplicate
from src.fastrun.config import DEFAULT_RULES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=150)
    ap.add_argument("--model", type=str, default="models/q_v1.pt")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--prior", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=61)
    args = ap.parse_args()
    cfg = DEFAULT_RULES

    def run(label, hero, villains):
        r = run_duplicate(hero, villains, n_deals=args.deals, cfg=cfg,
                          seed0=args.seed)
        print(f"{label:<26} win={r.win_rate:6.1%} [{r.ci[0]:.1%},{r.ci[1]:.1%}] "
              f"score={r.avg_score:+6.2f}  {r.avg_ms:6.2f}ms")

    mk = lambda: QAgent(cfg, model_path=args.model, top_k=args.top_k,
                        prior=args.prior)
    run("q vs greedy", mk(), [GreedyMinAgent(), GreedyMinAgent()])
    run("q vs m2", mk(), [M2Agent(cfg, endgame_k=6), M2Agent(cfg, endgame_k=6)])
    run("q vs m1", mk(), [M1Agent(cfg), M1Agent(cfg)])
    run("m2 vs greedy (ref)", M2Agent(cfg, endgame_k=6),
        [GreedyMinAgent(), GreedyMinAgent()])


if __name__ == "__main__":
    main()
