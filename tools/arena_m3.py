"""M3-A 评测: 价值网 1-ply 重排。

运行（需要 torch 环境）:
  python tools/arena_m3.py --deals 200 --model models/value_v1.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import (GreedyMinAgent, M1Agent, M2Agent,
                              SearchValueAgent, ValueAgent)
from src.arena.match import run_duplicate
from src.fastrun.config import DEFAULT_RULES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--model", type=str, default="models/value_v1.pt")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--win-weight", type=float, default=1.0)
    ap.add_argument("--prior", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()
    cfg = DEFAULT_RULES

    def run(label, hero, villains):
        r = run_duplicate(hero, villains, n_deals=args.deals, cfg=cfg,
                          seed0=args.seed)
        print(f"{label:<26} win={r.win_rate:6.1%} [{r.ci[0]:.1%},{r.ci[1]:.1%}] "
              f"score={r.avg_score:+6.2f}  {r.avg_ms:6.2f}ms")
        return r

    mk = lambda: ValueAgent(cfg, model_path=args.model, top_k=args.top_k,
                            win_weight=args.win_weight, prior=args.prior)
    run("value vs m1", mk(), [M1Agent(cfg), M1Agent(cfg)])
    run("value vs greedy", mk(), [GreedyMinAgent(), GreedyMinAgent()])
    run("m2 vs greedy (ref)", M2Agent(cfg, endgame_k=6),
        [GreedyMinAgent(), GreedyMinAgent()])

    sv = lambda: SearchValueAgent(cfg, model_path=args.model, k=6,
                                  top_k=args.top_k, win_weight=args.win_weight,
                                  seed=7)
    run("svalue vs greedy", sv(), [GreedyMinAgent(), GreedyMinAgent()])
    run("svalue vs m2", sv(), [M2Agent(cfg, endgame_k=6),
                               M2Agent(cfg, endgame_k=6)])


if __name__ == "__main__":
    main()
