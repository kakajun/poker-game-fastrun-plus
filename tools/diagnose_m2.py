"""M2 残局诊断: 触发次数、与 M1 决策的差异、耗时、以及在真手牌下的最优性。

运行: python tools/diagnose_m2.py
"""

import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent, M1Agent
from src.eval.policy_m1 import M1Policy
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState
from src.infer.sampler import sample_worlds
from src.search.endgame import solve_worlds

CFG = DEFAULT_RULES


def main():
    m1 = M1Agent(CFG)
    policy = M1Policy(CFG, endgame_cards=8, endgame_k=6, seed=1)
    trig = diff = 0
    total = 0
    ms_sum = 0.0
    for seed in range(40):
        st = GameState.new(seed)
        agents = [policy, m1, GreedyMinAgent()]
        while not st.over:
            ms = st.legal_moves(CFG)
            p = st.current
            if p == 0:
                total += 1
                if sum(st.remaining_cards()) <= 8:
                    trig += 1
                    t0 = time.perf_counter()
                    base = m1.act(st, ms)
                    got = policy.act(st, ms)
                    ms_sum += (time.perf_counter() - t0) * 1000
                    if got != base:
                        diff += 1
                    mv = got
                else:
                    mv = m1.act(st, ms)
            else:
                mv = agents[p].act(st, ms)
            st = st.apply(mv, CFG)

    print(f"P0 decisions={total}  endgame triggered={trig}  differs from M1={diff}")
    if trig:
        print(f"endgame 2x decision avg: {ms_sum/trig:.1f} ms")

    # 真手牌（上帝视角）残局解的上限：看它与 M1 的选择差异
    same = 0
    n = 0
    for seed in range(20):
        st = GameState.new(seed)
        agents = [M1Agent(CFG), M1Agent(CFG), M1Agent(CFG)]
        while not st.over:
            if st.current == 0 and sum(st.remaining_cards()) <= 8:
                ms = st.legal_moves(CFG)
                a = agents[0].act(st, ms)
                b = solve_worlds(st, [st], CFG)
                n += 1
                same += int(a == b)
            mv = agents[st.current].act(st, st.legal_moves(CFG))
            st = st.apply(mv, CFG)
    print(f"oracle endgame: n={n}  same_as_M1={same} ({same/max(1,n):.0%})")


if __name__ == "__main__":
    main()
