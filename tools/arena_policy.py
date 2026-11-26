"""v3 策略竞技场：候选 vs M2/上一冠军，6 座位排列 + paired bootstrap。

运行:
  python tools/arena_policy.py --candidate models/policy_v3.pt \
      --baseline m2 --deals 2000 --opponents greedy,m2
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import (GreedyMinAgent, M1Agent, M2Agent, PolicyAgent)
from src.arena.match import (MatchReport, paired_bootstrap_diff,
                             run_duplicate_permutations)
from src.fastrun.config import DEFAULT_RULES

CFG = DEFAULT_RULES


def _villains(name: str, seed: int):
    if name == "greedy":
        return [GreedyMinAgent(), GreedyMinAgent()]
    if name == "m1":
        return [M1Agent(CFG), M1Agent(CFG)]
    if name == "m2":
        return [M2Agent(CFG, endgame_k=6, seed=seed),
                M2Agent(CFG, endgame_k=6, seed=seed + 1)]
    raise ValueError(f"unknown opponent: {name}")


def _baseline(kind: str, model: str):
    if kind == "m2":
        return M2Agent(CFG, endgame_k=6, seed=17)
    if kind == "policy":
        return PolicyAgent(CFG, model_path=model)
    raise ValueError(f"unknown baseline: {kind}")


def _pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def _load_test_deals(pattern: str, limit: int):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no test data: {pattern}")
    deals = set()
    for f in files:
        with np.load(f) as d:
            if "split" in d and np.any(d["split"] != 2):
                raise SystemExit(f"file is not pure test split: {f}")
            deals.update(int(x) for x in np.unique(d["deal_id"]))
    out = sorted(deals)
    if not out:
        raise SystemExit(f"test data has no deals: {pattern}")
    return out[:limit] if limit > 0 else out


def _report(label: str, r: MatchReport):
    ms = [x.ms for x in r.records]
    print(f"{label:<24} win={r.win_rate:6.1%} score={r.avg_score:+7.2f} "
          f"ms p50={_pct(ms, .50):5.2f} p95={_pct(ms, .95):5.2f} "
          f"p99={_pct(ms, .99):5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=str, default="models/policy_v3.pt")
    ap.add_argument("--baseline", choices=("m2", "policy"), default="m2")
    ap.add_argument("--baseline-model", type=str,
                    default="models/policy_prev.pt")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--test-data", type=str,
                    default="data/cf_v3.test.w*.npz",
                    help="冻结测试 deal；不存在则按 seed/deals 生成")
    ap.add_argument("--opponents", type=str, default="greedy,m2")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--gate", action="store_true",
                    help="任一对手未通过 CI 门槛则以非零状态退出")
    args = ap.parse_args()

    if glob.glob(args.test_data):
        test_deals = _load_test_deals(args.test_data, args.deals)
        print(f"using {len(test_deals)} frozen test deals")
    else:
        test_deals = None
        print(f"WARNING: no frozen test data ({args.test_data}); "
              f"falling back to {args.deals} generated deals")

    passed = True
    for name in [x.strip() for x in args.opponents.split(",") if x.strip()]:
        print(f"\n=== opponent: {name} ===")
        cand = run_duplicate_permutations(
            PolicyAgent(CFG, model_path=args.candidate),
            _villains(name, args.seed + 10),
            n_deals=args.deals, cfg=CFG, seed0=args.seed,
            deal_ids=test_deals)
        base = run_duplicate_permutations(
            _baseline(args.baseline, args.baseline_model),
            _villains(name, args.seed + 10),
            n_deals=args.deals, cfg=CFG, seed0=args.seed,
            deal_ids=test_deals)
        _report("candidate", cand)
        _report(f"baseline:{args.baseline}", base)
        diff = paired_bootstrap_diff(cand, base, n_boot=args.bootstrap,
                                     seed=args.seed)
        print(f"paired delta: {diff}")
        win_ok = diff.win_ci[0] > 0.0
        score_ok = diff.score_ci[0] > 0.0
        no_win_regress = diff.win_ci[1] > 0.0
        no_score_regress = diff.score_ci[1] > 0.0
        ok = (win_ok and no_score_regress) or (score_ok and no_win_regress)
        print(f"gate: {'PASS' if ok else 'FAIL'} "
              f"(win CI lower>0={win_ok}, score CI lower>0={score_ok}, "
              f"other-not-degraded={no_score_regress and no_win_regress})")
        passed = passed and ok

    if args.gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
