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

from ..eval.score_ev import score_breakdowns
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import MoveType
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
    base: float = 0.0
    spring: float = 0.0
    bomb: float = 0.0
    spring_opponents: int = 0
    full_spring: int = 0
    hero_bombs: int = 0
    opp_bombs: int = 0


@dataclass
class PairedDiff:
    n_deals: int
    n_games: int
    win_diff: float
    win_ci: Tuple[float, float]
    score_diff: float
    score_ci: Tuple[float, float]
    base_diff: float = 0.0
    base_ci: Tuple[float, float] = (0.0, 0.0)
    spring_diff: float = 0.0
    spring_ci: Tuple[float, float] = (0.0, 0.0)
    bomb_diff: float = 0.0
    bomb_ci: Tuple[float, float] = (0.0, 0.0)

    def __str__(self) -> str:
        return (f"deals={self.n_deals} games={self.n_games} "
                f"Δwin={self.win_diff:+.2%} "
                f"[{self.win_ci[0]:+.2%},{self.win_ci[1]:+.2%}] "
                f"Δscore={self.score_diff:+.3f} "
                f"[{self.score_ci[0]:+.3f},{self.score_ci[1]:+.3f}] "
                f"Δbase={self.base_diff:+.3f} "
                f"[{self.base_ci[0]:+.3f},{self.base_ci[1]:+.3f}] "
                f"Δspring={self.spring_diff:+.3f} "
                f"[{self.spring_ci[0]:+.3f},{self.spring_ci[1]:+.3f}] "
                f"Δbomb={self.bomb_diff:+.3f} "
                f"[{self.bomb_ci[0]:+.3f},{self.bomb_ci[1]:+.3f}]")


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
    avg_base: float = 0.0
    avg_spring: float = 0.0
    avg_bomb: float = 0.0
    partial_spring_rate: float = 0.0
    full_spring_rate: float = 0.0
    bomb_game_rate: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    def __str__(self) -> str:
        lo, hi = self.ci
        return (f"{self.label:<28} n={self.n:4d}  win={self.win_rate:6.1%} "
                f"[{lo:.1%},{hi:.1%}]  score={self.avg_score:+6.2f} "
                f"(base={self.avg_base:+.2f} spring={self.avg_spring:+.2f} "
                f"bomb={self.avg_bomb:+.2f})  spring={self.spring_rate:5.1%} "
                f"full={self.full_spring_rate:5.1%} "
                f"bomb_games={self.bomb_game_rate:5.1%}  "
                f"{self.avg_ms:5.2f}ms/decision")


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
    partial_springs = full_springs = bomb_games = 0
    score_sum = base_sum = spring_sum = bomb_sum = 0.0
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
            bd = score_breakdowns(
                st.hands, st.tricks, st.bombs, st.winner, cfg)[seat]
            score_sum += bd.total
            base_sum += bd.base
            spring_sum += bd.spring
            bomb_sum += bd.bomb
            hb = sum(1 for p, m in st.action_log
                     if p == seat and m.type == MoveType.BOMB)
            bomb_games += int(hb > 0)
            if st.winner == seat:
                wins += 1
                if bd.spring_opponents:
                    springs += 1
                if bd.full_spring:
                    full_springs += 1
                elif bd.spring_opponents == 1:
                    partial_springs += 1
            else:
                losses += 1

    return MatchReport(
        label=f"{hero.name} vs {'+'.join(v.name for v in villains)}",
        n=n, wins=wins, losses=losses,
        avg_score=score_sum / n,
        spring_rate=springs / wins if wins else 0.0,
        avg_ms=ms_sum / n,
        ci=wilson(wins, n),
        avg_base=base_sum / n,
        avg_spring=spring_sum / n,
        avg_bomb=bomb_sum / n,
        partial_spring_rate=partial_springs / n,
        full_spring_rate=full_springs / n,
        bomb_game_rate=bomb_games / n,
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
    partial_springs = full_springs = bomb_games = 0
    score_sum = base_sum = spring_sum = bomb_sum = 0.0
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
            bd = score_breakdowns(
                st.hands, st.tricks, st.bombs, st.winner, cfg)[hero_seat]
            score = float(bd.total)
            hero_bombs = sum(1 for p, m in st.action_log
                             if p == hero_seat and m.type == MoveType.BOMB)
            opp_bombs = sum(1 for p, m in st.action_log
                            if p != hero_seat and m.type == MoveType.BOMB)
            records.append(GameRecord(
                deal_id=deal_id, layout=layout_i, hero_seat=hero_seat,
                win=win, score=score, ms=avg_ms, base=float(bd.base),
                spring=float(bd.spring), bomb=float(bd.bomb),
                spring_opponents=bd.spring_opponents,
                full_spring=1 if bd.full_spring else 0,
                hero_bombs=hero_bombs, opp_bombs=opp_bombs))
            wins += win
            losses += 1 - win
            score_sum += score
            base_sum += bd.base
            spring_sum += bd.spring
            bomb_sum += bd.bomb
            ms_sum += avg_ms
            bomb_games += int(hero_bombs > 0)
            if win and bd.spring_opponents:
                springs += 1
                if bd.full_spring:
                    full_springs += 1
                else:
                    partial_springs += 1

    n = len(records)
    return MatchReport(
        label=f"{hero.name} vs {'+'.join(v.name for v in villains)}",
        n=n, wins=wins, losses=losses, avg_score=score_sum / n,
        spring_rate=springs / wins if wins else 0.0, avg_ms=ms_sum / n,
        ci=wilson(wins, n), records=tuple(records),
        avg_base=base_sum / n,
        avg_spring=spring_sum / n,
        avg_bomb=bomb_sum / n,
        partial_spring_rate=partial_springs / n,
        full_spring_rate=full_springs / n,
        bomb_game_rate=bomb_games / n,
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

    metrics = ("win", "score", "base", "spring", "bomb")
    per_deal: Dict[str, Dict[int, List[float]]] = {
        name: {} for name in metrics}
    for key, ra in ha.items():
        rb = hb[key]
        vals = {
            "win": float(ra.win - rb.win),
            "score": ra.score - rb.score,
            "base": ra.base - rb.base,
            "spring": ra.spring - rb.spring,
            "bomb": ra.bomb - rb.bomb,
        }
        for name, value in vals.items():
            per_deal[name].setdefault(key[0], []).append(value)

    deals = sorted(per_deal["win"])
    diffs = {}
    for name in metrics:
        all_v = [x for d in deals for x in per_deal[name][d]]
        diffs[name] = sum(all_v) / len(all_v)

    rng = random.Random(seed)
    boot = {name: [] for name in metrics}
    for _ in range(n_boot):
        sample = [deals[rng.randrange(len(deals))] for _ in deals]
        for name in metrics:
            vals = [x for d in sample for x in per_deal[name][d]]
            boot[name].append(sum(vals) / len(vals))
    lo = int(0.025 * (n_boot - 1))
    hi = int(0.975 * (n_boot - 1))
    cis = {}
    for name in metrics:
        boot[name].sort()
        cis[name] = (boot[name][lo], boot[name][hi])

    return PairedDiff(
        n_deals=len(deals), n_games=len(ha),
        win_diff=diffs["win"], win_ci=cis["win"],
        score_diff=diffs["score"], score_ci=cis["score"],
        base_diff=diffs["base"], base_ci=cis["base"],
        spring_diff=diffs["spring"], spring_ci=cis["spring"],
        bomb_diff=diffs["bomb"], bomb_ci=cis["bomb"],
    )
