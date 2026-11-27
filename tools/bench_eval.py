"""评估函数开销探针: min_steps / M1 决策耗时。

运行: python tools/bench_eval.py
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState
from src.eval.decompose import min_steps
from src.eval.policy_m1 import M1Policy


def main() -> None:
    cfg = DEFAULT_RULES
    rng = random.Random(3)
    policy = M1Policy(cfg)

    # 1) min_steps 单次耗时（按剩余张数分层）
    for n_cards in (15, 12, 10, 8, 6):
        hands = []
        for _ in range(60):
            deck = C.deck_list()
            rng.shuffle(deck)
            hands.append(C.make([deck[:n_cards].count(i) for i in range(C.NUM_RANKS)]))
        t0 = time.perf_counter()
        steps = [min_steps(h, cfg) for h in hands]
        dt = (time.perf_counter() - t0) / len(hands) * 1000
        print(f"min_steps {n_cards:2d} cards: {dt:7.3f} ms/call  "
              f"avg_steps={sum(steps)/len(steps):.1f}")

    # 2) M1 决策耗时（真实残局分布）
    t0 = time.perf_counter()
    decisions = 0
    for seed in range(40):
        st = GameState.new(seed)
        while not st.over:
            mv = policy.act(st)
            st = st.apply(mv)
            decisions += 1
    dt = time.perf_counter() - t0
    print(f"M1 decisions: {decisions} in {dt:.2f}s -> "
          f"{dt / decisions * 1000:.2f} ms/decision, {decisions/dt:.0f}/s")


if __name__ == "__main__":
    main()
