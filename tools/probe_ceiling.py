"""上限阶梯诊断：O0/O1/O2。

O0: 上帝视角 + M1 topK + 当前轮 Greedy + min_steps 静态叶子（旧 49.2% 口径）
O1: 上帝视角 + 全合法动作 + 当前轮 Greedy + min_steps
O2: 上帝视角 + 全合法动作 + 整局 continuation（hero M2，对手可配）

同时统计 M1 top1/topK 对 Oracle 最优动作的召回率，区分“候选剪枝”与“估值/horizon”。

运行:
  python tools/probe_ceiling.py --deals 120 --levels o0,o1
  python tools/probe_ceiling.py --deals 20 --levels o2 --opponent greedy
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import (Agent, GreedyMinAgent, M1Agent, M2Agent)
from src.arena.match import run_duplicate
from src.eval.decompose import min_steps
from src.eval.policy_m1 import M1Policy
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.search.pimc import _rollout_trick

CFG = DEFAULT_RULES


def _villain(name: str, seed: int):
    if name == "greedy":
        return GreedyMinAgent()
    if name == "m1":
        return M1Agent(CFG)
    if name == "m2":
        return M2Agent(CFG, endgame_cards=8, endgame_k=4, seed=seed)
    raise ValueError(f"unknown opponent: {name}")


def _full_playout(state, me, opponent, seed):
    st = state
    hero = M2Agent(CFG, endgame_cards=8, endgame_k=4, seed=seed)
    v1 = _villain(opponent, seed + 1)
    v2 = _villain(opponent, seed + 2)
    agents = [None, None, None]
    agents[me] = hero
    vi = 0
    for q in range(3):
        if q != me:
            agents[q] = v1 if vi == 0 else v2
            vi += 1
    for _ in range(400):
        if st.over:
            return (1.0 if st.winner == me else 0.0), float(st.scores[me])
        p = st.current
        st = st.apply(agents[p].act(st, st.legal_moves(CFG)), CFG)
    return 0.0, 0.0


class OracleAgent(Agent):
    """用于诊断，不是可部署策略（能看到所有手牌）。"""

    def __init__(self, level: str = "o0", top_k: int = 6,
                 opponent: str = "greedy", seed: int = 0):
        self.level = level
        self.policy = M1Policy(CFG)
        self.top_k = top_k
        self.opponent = opponent
        self.seed = seed
        self.calls = 0
        self.decisions = 0
        self.top1_hits = 0
        self.topk_hits = 0
        self.outside_topk = 0

    @property
    def name(self):
        return f"oracle_{self.level}"

    def reset(self):
        self.calls = 0

    def _static_value(self, st, me):
        if st.over:
            return 1000.0 if st.winner == me else -1000.0
        mine = min_steps(st.hands[me], CFG)
        opp = [min_steps(st.hands[q], CFG) for q in range(3) if q != me]
        v = -mine + 0.5 * (sum(opp) / 2.0)
        if st.current == me and st.last_play is None:
            v += 1.0
        return v

    def act(self, state, moves):
        ms = list(state.legal_moves(CFG) if moves is None else moves)
        if len(ms) <= 1:
            return ms[0]
        me = state.current
        ranked = [m for m in self.policy.rank_moves(state, ms) if not m.is_pass]
        if not ranked:
            return ms[0]

        if self.level == "o0":
            cands = ranked[:self.top_k]
        elif self.level in ("o1", "o2"):
            cands = ms
        else:
            raise ValueError(f"unknown level: {self.level}")

        self.calls += 1
        best, best_v = cands[0], float("-inf")
        for m in cands:
            child = state.apply(m, CFG)
            if self.level == "o2":
                win, score = _full_playout(
                    child, me, self.opponent,
                    self.seed * 100003 + self.calls)
                v = win * 1000.0 + score
            else:
                st = _rollout_trick(child, CFG)
                v = self._static_value(st, me)
            if v > best_v:
                best_v, best = v, m

        if self.level != "o0":
            self.decisions += 1
            if best == ranked[0]:
                self.top1_hits += 1
            if best in ranked[:self.top_k]:
                self.topk_hits += 1
            else:
                self.outside_topk += 1
        return best

    def report_recall(self):
        if self.decisions == 0:
            return ""
        return (f"  M1 recall@1={self.top1_hits/self.decisions:.1%} "
                f"recall@{self.top_k}={self.topk_hits/self.decisions:.1%} "
                f"outside_topK={self.outside_topk/self.decisions:.1%} "
                f"n={self.decisions}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=120)
    ap.add_argument("--levels", type=str, default="o0,o1")
    ap.add_argument("--opponent", choices=("greedy", "m1", "m2"),
                    default="greedy")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    for level in [x.strip() for x in args.levels.split(",") if x.strip()]:
        if level not in ("o0", "o1", "o2"):
            raise SystemExit(f"unknown level: {level}")
        hero = OracleAgent(level, args.top_k, args.opponent, args.seed)
        villains = [_villain(args.opponent, args.seed + 10),
                    _villain(args.opponent, args.seed + 11)]
        r = run_duplicate(hero, villains, n_deals=args.deals, cfg=CFG,
                          seed0=args.seed)
        print(f"{level} vs {args.opponent:<6} win={r.win_rate:6.1%} "
              f"[{r.ci[0]:.1%},{r.ci[1]:.1%}] score={r.avg_score:+6.2f} "
              f"{r.avg_ms:6.2f}ms")
        extra = hero.report_recall()
        if extra:
            print(extra)


if __name__ == "__main__":
    main()
