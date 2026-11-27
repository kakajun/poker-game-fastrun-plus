"""当前 fastrun 引擎的春天/炸弹分项评估。

示例:
  python tools/eval_score_components.py --agent m2 --opponent greedy \
      --deals 200 --test-data "data/cf_v4.test.w*.npz"
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent, M1Agent, M2Agent, PolicyAgent
from src.arena.match import paired_bootstrap_diff, run_duplicate_permutations
from src.fastrun.config import DEFAULT_RULES

CFG = DEFAULT_RULES


def _load_deals(pattern: str, limit: int):
    if not pattern or not glob.glob(pattern):
        return None
    deals = set()
    for f in sorted(glob.glob(pattern)):
        with np.load(f) as d:
            if "split" in d and np.any(d["split"] != 2):
                raise SystemExit(f"not a pure test split: {f}")
            deals.update(int(x) for x in np.unique(d["deal_id"]))
    out = sorted(deals)
    return out[:limit] if limit > 0 else out


def _agent(kind: str, model: str, seed: int):
    if kind == "m1":
        return M1Agent(CFG)
    if kind == "m2":
        return M2Agent(CFG, endgame_k=6, seed=seed)
    if kind == "greedy":
        return GreedyMinAgent()
    if kind == "policy":
        return PolicyAgent(CFG, model_path=model)
    raise ValueError(f"unknown agent: {kind}")


def _villains(kind: str, seed: int):
    if kind == "greedy":
        return [GreedyMinAgent(), GreedyMinAgent()]
    if kind == "m1":
        return [M1Agent(CFG), M1Agent(CFG)]
    if kind == "m2":
        return [M2Agent(CFG, endgame_k=6, seed=seed),
                M2Agent(CFG, endgame_k=6, seed=seed + 1)]
    raise ValueError(f"unknown opponent: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=("m1", "m2", "greedy", "policy"),
                    default="m2")
    ap.add_argument("--model", type=str, default="models/policy_v4.pt")
    ap.add_argument("--opponent", choices=("greedy", "m1", "m2"),
                    default="greedy")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--test-data", type=str, default="data/cf_v4.test.w*.npz")
    ap.add_argument("--compare", choices=("", "m1", "m2"), default="")
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()

    deals = _load_deals(args.test_data, args.deals)
    if deals is None:
        print(f"WARNING: no test data ({args.test_data}); use generated deals")
    cand = run_duplicate_permutations(
        _agent(args.agent, args.model, args.seed),
        _villains(args.opponent, args.seed + 10),
        n_deals=args.deals, cfg=CFG, seed0=args.seed, deal_ids=deals)
    print(cand)

    if args.compare:
        base = run_duplicate_permutations(
            _agent(args.compare, args.model, args.seed + 20),
            _villains(args.opponent, args.seed + 10),
            n_deals=args.deals, cfg=CFG, seed0=args.seed, deal_ids=deals)
        print(base)
        print(paired_bootstrap_diff(cand, base, seed=args.seed))


if __name__ == "__main__":
    main()
