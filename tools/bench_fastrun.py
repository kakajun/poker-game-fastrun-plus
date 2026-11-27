"""M0 性能/分支度基线：随机对局吞吐、平均合法动作数、平均回合数。

运行: python tools/bench_fastrun.py [games]
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun.moves import gen_lead
from src.fastrun.state import GameState


def main(n_games: int = 500) -> None:
    rng = random.Random(7)
    t0 = time.perf_counter()
    decisions = 0
    plies = 0
    branch_sum = 0
    branch_max = 0

    for seed in range(n_games):
        st = GameState.new(seed)
        while not st.over:
            ms = st.legal_moves()
            decisions += 1
            branch_sum += len(ms)
            branch_max = max(branch_max, len(ms))
            st = st.apply(rng.choice(ms))
        plies += st.tricks[0] + st.tricks[1] + st.tricks[2]

    dt = time.perf_counter() - t0
    print(f"games           : {n_games}")
    print(f"wall            : {dt:.2f}s  ({n_games / dt:.0f} games/s)")
    print(f"decisions       : {decisions}  ({decisions / dt / 1000:.1f}k/s)")
    print(f"avg legal moves : {branch_sum / decisions:.1f}   max {branch_max}")
    print(f"avg plays/game  : {plies / n_games:.1f}")

    # 满手牌的首出分支度（决策最重的情形）
    worst = 0
    for seed in range(200):
        st = GameState.new(seed)
        worst = max(worst, len(gen_lead(st.hands[st.current])))
    print(f"max lead moves  : {worst}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
