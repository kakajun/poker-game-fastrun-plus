import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.arena.agents import GreedyMinAgent, M1Agent
from src.arena.match import run_duplicate_permutations
from src.eval.objective import ObjectiveConfig
from src.fastrun.config import DEFAULT_RULES

CFG = DEFAULT_RULES
deals = set()
for f in glob.glob("data/cf_v4_r1.test.w*.npz"):
    with np.load(f) as d:
        deals.update(int(x) for x in np.unique(d["deal_id"]))
DEALS = sorted(deals)[:100]
for name, obj in [
    ("win_primary", ObjectiveConfig(mode="win_primary")),
    ("s1_b1", ObjectiveConfig(max_extra_steps=0)),
]:
    r = run_duplicate_permutations(
        M1Agent(CFG, objective=obj),
        [GreedyMinAgent(), GreedyMinAgent()],
        n_deals=0, seed0=83, deal_ids=DEALS)
    print(name, f"win={r.win_rate:.1%} score={r.avg_score:+.2f} "
          f"spring={r.avg_spring:+.2f} bomb={r.avg_bomb:+.2f}")
