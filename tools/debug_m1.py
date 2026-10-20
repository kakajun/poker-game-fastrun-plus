"""M1 决策诊断：打印关键决策的候选排序，并统计行为特征。

运行: python tools/debug_m1.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent
from src.eval.policy_m1 import M1Policy
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState

cfg = DEFAULT_RULES
policy = M1Policy(cfg)


def trace(seed: int):
    st = GameState.new(seed)
    agents = [policy, GreedyMinAgent(), GreedyMinAgent()]
    print(f"\n===== deal {seed} =====")
    while not st.over:
        ms = st.legal_moves(cfg)
        p = st.current
        mv = agents[p].act(st, ms)
        if p == 0:
            print(policy.explain(st, ms))
            print(f"  -> {mv}")
        st = st.apply(mv, cfg)
    print(st.to_str())
    print(f"winner={st.winner} scores={st.scores}")


def behaviour(n: int = 40):
    losses_remain = []
    passes = bombs = wins = 0
    for seed in range(n):
        st = GameState.new(seed)
        agents = [policy, GreedyMinAgent(), GreedyMinAgent()]
        while not st.over:
            ms = st.legal_moves(cfg)
            p = st.current
            mv = agents[p].act(st, ms)
            if p == 0:
                passes += int(mv.is_pass)
                bombs += int(mv.type.name == "BOMB")
            st = st.apply(mv, cfg)
        if st.winner == 0:
            wins += 1
        else:
            losses_remain.append(C.total(st.hands[0]))
    print(f"\nM1 as P0: wins={wins}/{n} passes={passes} bombs={bombs} "
          f"avg_remain_on_loss={sum(losses_remain)/max(1,len(losses_remain)):.1f}")


if __name__ == "__main__":
    trace(3)
    behaviour(40)
