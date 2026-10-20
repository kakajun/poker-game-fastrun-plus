"""同牌多座位对局与报表。

为什么必须 duplicate-deal:
  一局跑得快的运气成分极大（手牌质量 + 首出权）。让 hero 在同一个发牌上坐
  0/1/2 三个座位各打一次，能把"牌运"对称掉，剩下的差异才反映策略强弱。

为什么必须报置信区间:
  200 局上 ±3% 的波动足以让"提升"变噪声。所有结论以 Wilson CI 为准。
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.state import GameState
from .agents import Agent


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - half) / d), min(1.0, (c + half) / d))


@dataclass
class MatchReport:
    label: str
    n: int
    wins: int
    losses: int
    avg_score: float
    spring_rate: float
    avg_ms: float
    ci: Tuple[float, float]

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    def __str__(self) -> str:
        lo, hi = self.ci
        return (f"{self.label:<28} n={self.n:4d}  win={self.win_rate:6.1%} "
                f"[{lo:.1%},{hi:.1%}]  score={self.avg_score:+6.2f}  "
                f"spring={self.spring_rate:5.1%}  {self.avg_ms:5.2f}ms/decision")


def play_game(agents: Sequence[Agent], cfg: RuleConfig,
              state: Optional[GameState] = None,
              hero_seat: int = 0,
              max_steps: int = 400) -> Tuple[GameState, float]:
    """跑完一局，返回 (终局状态, hero 平均决策毫秒)。"""
    st = state if state is not None else GameState.new()
    hero_ms = 0.0
    hero_moves = 0
    steps = 0
    while not st.over:
        steps += 1
        if steps > max_steps:
            raise RuntimeError("game did not terminate")
        ms = st.legal_moves(cfg)
        p = st.current
        t0 = time.perf_counter()
        mv = agents[p].act(st, ms)
        if p == hero_seat:
            hero_ms += (time.perf_counter() - t0) * 1000.0
            hero_moves += 1
        st = st.apply(mv, cfg)
    avg_ms = hero_ms / hero_moves if hero_moves else 0.0
    return st, avg_ms


def run_duplicate(hero: Agent, villains: Sequence[Agent],
                  n_deals: int = 200, cfg: RuleConfig = DEFAULT_RULES,
                  seed0: int = 1) -> MatchReport:
    """hero 在每副牌的 3 个座位各打一次，villains 填满其余两个座位。"""
    if len(villains) != 2:
        raise ValueError("need exactly 2 villains")
    wins = 0
    losses = 0
    springs = 0
    score_sum = 0
    ms_sum = 0.0
    n = 0
    t_start = time.perf_counter()

    for d in range(n_deals):
        base = GameState.new(seed0 + d)
        for seat in range(3):
            for a in [hero] + list(villains):
                a.reset()
            agents: List[Optional[Agent]] = [None, None, None]
            agents[seat] = hero
            vi = 0
            for s in range(3):
                if s != seat:
                    agents[s] = villains[vi]
                    vi += 1
            st = GameState.from_hands(base.hands, current=base.current)
            st, avg_ms = play_game(agents, cfg, st, hero_seat=seat)
            n += 1
            ms_sum += avg_ms
            score_sum += st.scores[seat]
            if st.winner == seat:
                wins += 1
                if any(st.tricks[q] == 0 for q in range(3) if q != seat):
                    springs += 1
            else:
                losses += 1

    return MatchReport(
        label=f"{hero.name} vs {'+'.join(v.name for v in villains)}",
        n=n, wins=wins, losses=losses,
        avg_score=score_sum / n,
        spring_rate=springs / wins if wins else 0.0,
        avg_ms=ms_sum / n,
        ci=wilson(wins, n),
    )
