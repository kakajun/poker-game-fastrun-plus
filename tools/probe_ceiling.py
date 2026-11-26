"""上限诊断：上帝视角（能看到对手手牌）的 1-ply 搜索能赢到多少。

用途: 判断"52% 对贪心"是策略不够好，还是规则本身的天花板。
注意: 这是作弊 agent，只用于测量，不进 src/。

运行: python tools/probe_ceiling.py --deals 120
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import Agent, GreedyMinAgent, M2Agent
from src.arena.match import run_duplicate
from src.eval.decompose import min_steps
from src.eval.policy_m1 import M1Policy
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.search.pimc import _rollout_trick

CFG = DEFAULT_RULES


class OracleAgent(Agent):
    name = "oracle"

    def __init__(self, cfg=CFG, top_k=6):
        self.cfg = cfg
        self.policy = M1Policy(cfg)
        self.top_k = top_k

    def act(self, state, moves):
        me = state.current
        ms = list(moves)
        if len(ms) == 1:
            return ms[0]
        ranked = [m for m in self.policy.rank_moves(state, ms) if not m.is_pass]
        if not ranked:
            return ms[0]
        cands = ranked[:self.top_k]

        best, best_v = cands[0], -1e9
        for m in cands:
            st = _rollout_trick(state.apply(m, self.cfg), self.cfg)
            if st.over:
                v = 1000.0 if st.winner == me else -1000.0
            else:
                mine = min_steps(st.hands[me], self.cfg)
                opp = [min_steps(st.hands[q], self.cfg) for q in range(3) if q != me]
                v = -mine + 0.5 * (sum(opp) / 2.0)
                if st.current == me and st.last_play is None:
                    v += 1.0
            if v > best_v:
                best_v, best = v, m
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=120)
    args = ap.parse_args()

    for label, hero, villains in [
        ("oracle vs greedy", OracleAgent(), [GreedyMinAgent(), GreedyMinAgent()]),
        ("oracle vs m2", OracleAgent(), [M2Agent(CFG, endgame_k=6),
                                        M2Agent(CFG, endgame_k=6)]),
        ("m2 vs greedy (ref)", M2Agent(CFG, endgame_k=6),
         [GreedyMinAgent(), GreedyMinAgent()]),
    ]:
        r = run_duplicate(hero, villains, n_deals=args.deals, cfg=CFG, seed0=71)
        print(f"{label:<22} win={r.win_rate:6.1%} [{r.ci[0]:.1%},{r.ci[1]:.1%}] "
              f"score={r.avg_score:+6.2f}  {r.avg_ms:5.2f}ms")


if __name__ == "__main__":
    main()
