"""同牌多座位对局与报表。

旧 `run_duplicate` 保留为 3 座位兼容接口；v3 验收使用
`run_duplicate_permutations`（hero/两个对手共 6 种排列）和
`paired_bootstrap_diff`（按 deal 聚类，不把同牌各座位当独立样本）。
"""

import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class GameRecord:
    deal_id: int
    layout: int
    hero_seat: int
    win: int
    score: float
    ms: float


@dataclass
class PairedDiff:
    n_deals: int
    n_games: int
    win_diff: float
    win_ci: Tuple[float, float]
    score_diff: float
    score_ci: Tuple[float, float]

    def __str__(self) -> str:
        return (f"deals={self.n_deals} games={self.n_games} "
                f"Δwin={self.win_diff:+.2%} "
                f"[{self.win_ci[0]:+.2%},{self.win_ci[1]:+.2%}] "
                f"Δscore={self.score_diff:+.3f} "
                f"[{self.score_ci[0]:+.3f},{self.score_ci[1]:+.3f}]")


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
    records: Tuple[GameRecord, ...] = ()

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


def run_duplicate_permutations(hero: Agent, villains: Sequence[Agent],
                               n_deals: int = 200,
                               cfg: RuleConfig = DEFAULT_RULES,
                               seed0: int = 1,
                               deal_ids: Optional[Sequence[int]] = None
                               ) -> MatchReport:
    """同一副牌跑 hero/两个对手的全部 6 种座位排列。"""
    if len(villains) != 2:
        raise ValueError("need exactly 2 villains")
    participants = [hero] + list(villains)
    layouts = list(itertools.permutations(range(3)))
    records: List[GameRecord] = []
    wins = losses = springs = 0
    score_sum = 0.0
    ms_sum = 0.0
    deals = list(deal_ids) if deal_ids is not None else [
        seed0 + d for d in range(n_deals)]

    for deal_id in deals:
        base = GameState.new(deal_id)
        for layout_i, perm in enumerate(layouts):
            for a in participants:
                a.reset()
            agents = [participants[perm[s]] for s in range(3)]
            hero_seat = perm.index(0)
            st = GameState.from_hands(base.hands, current=base.current)
            st, avg_ms = play_game(agents, cfg, st, hero_seat=hero_seat)
            win = 1 if st.winner == hero_seat else 0
            score = float(st.scores[hero_seat])
            records.append(GameRecord(
                deal_id=deal_id, layout=layout_i, hero_seat=hero_seat,
                win=win, score=score, ms=avg_ms))
            wins += win
            losses += 1 - win
            score_sum += score
            ms_sum += avg_ms
            if win and any(st.tricks[q] == 0 for q in range(3)
                           if q != hero_seat):
                springs += 1

    n = len(records)
    return MatchReport(
        label=f"{hero.name} vs {'+'.join(v.name for v in villains)}",
        n=n, wins=wins, losses=losses, avg_score=score_sum / n,
        spring_rate=springs / wins if wins else 0.0, avg_ms=ms_sum / n,
        ci=wilson(wins, n), records=tuple(records),
    )


def paired_bootstrap_diff(hero: MatchReport, baseline: MatchReport,
                          n_boot: int = 4000,
                          seed: int = 0) -> PairedDiff:
    """按 deal 聚类重采样；要求两份报表使用相同 deals/layout。"""
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    ha = {(r.deal_id, r.layout): r for r in hero.records}
    hb = {(r.deal_id, r.layout): r for r in baseline.records}
    if set(ha) != set(hb):
        raise ValueError("paired reports must share deals/layouts")

    per_deal_w: Dict[int, List[float]] = {}
    per_deal_s: Dict[int, List[float]] = {}
    for key, ra in ha.items():
        rb = hb[key]
        per_deal_w.setdefault(key[0], []).append(float(ra.win - rb.win))
        per_deal_s.setdefault(key[0], []).append(ra.score - rb.score)

    deals = sorted(per_deal_w)
    all_w = [x for d in deals for x in per_deal_w[d]]
    all_s = [x for d in deals for x in per_deal_s[d]]
    win_diff = sum(all_w) / len(all_w)
    score_diff = sum(all_s) / len(all_s)

    rng = random.Random(seed)
    boot_w, boot_s = [], []
    for _ in range(n_boot):
        sample = [deals[rng.randrange(len(deals))] for _ in deals]
        bw = [x for d in sample for x in per_deal_w[d]]
        bs = [x for d in sample for x in per_deal_s[d]]
        boot_w.append(sum(bw) / len(bw))
        boot_s.append(sum(bs) / len(bs))
    boot_w.sort()
    boot_s.sort()
    lo = int(0.025 * (n_boot - 1))
    hi = int(0.975 * (n_boot - 1))
    return PairedDiff(
        n_deals=len(deals), n_games=len(ha),
        win_diff=win_diff, win_ci=(boot_w[lo], boot_w[hi]),
        score_diff=score_diff, score_ci=(boot_s[lo], boot_s[hi]),
    )
