"""M1 策略（规则版，全部结论来自竞技场实测，不是拍脑袋）。

决策顺序:
  1. 只有一个合法动作 / 只能过 -> 执行；
  2. 能一手走完 -> 走完；
  3. 在"不拆自己牌型"的候选里（progress 最大）挑；
  4. 首出时若存在不消耗 2/A/K 的候选，先排除消耗大牌的那些；
  5. 剩下的按 点数最小 -> 出牌张数多 排序。

对强基线的历史实测（同牌多座位，`tools/arena_m1.py`）:
    greedy_min   vs greedy 33.3%
    prog_rank    vs greedy 43.3%
    prog_hold    vs greedy 43.3%   <- 本实现
  Random 不作为验收基线。线性加权打分（手数 + 控制概率 + 大牌代价）已废弃，
  原因是它会把 2/A 当成"高控制概率"打出去，属于典型短视。
"""

from typing import List, Optional, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState
from .control import burn_cost, keep_lead_prob
from .decompose import min_steps
from .objective import ObjectiveConfig
from .score_ev import zero_play_opponents

# 残局兑现的保权阈值（实测 0.7~0.8 最好，0.9 开始过于保守）
SAFE_P = 0.8


class M1Policy:
    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 endgame_cards: int = 0, endgame_k: int = 8, seed: int = 0,
                 guaranteed_lead: bool = False,
                 objective: Optional[ObjectiveConfig] = None,
                 spring_endgame_cards: int = 12):
        self.cfg = cfg
        self.endgame_cards = endgame_cards
        self.endgame_k = endgame_k
        self.seed = seed
        self._calls = 0
        self.guaranteed_lead = guaranteed_lead
        self.objective = objective or ObjectiveConfig()
        self.spring_endgame_cards = spring_endgame_cards

    def reset(self) -> None:
        self._calls = 0

    # ------------------------------------------------------------------

    def _endgame_move(self, state: GameState,
                      moves: List[Move]) -> Optional[Move]:
        if not self.endgame_cards:
            return None
        total = sum(state.remaining_cards())
        zero_opp = zero_play_opponents(state.tricks, state.current)
        spring_live = (zero_opp > 0
                       and min_steps(state.hand(), self.cfg) <= 2)
        limit = max(self.endgame_cards,
                    self.spring_endgame_cards if spring_live else 0)
        if total > limit:
            return None
        from ..infer.sampler import SamplingError, sample_worlds
        from ..search.endgame import solve_worlds
        self._calls += 1
        try:
            worlds = sample_worlds(
                state, state.current, self.endgame_k, self.cfg,
                seed=self.seed * 7919 + self._calls)
        except SamplingError:
            # 证据约束过紧时退回普通 M1，而不是偷偷使用伪一致世界。
            return None
        mv = solve_worlds(state, worlds, self.cfg,
                          objective=self.objective)
        return mv if mv in moves else None

    # ------------------------------------------------------------------

    def _progress(self, state: GameState, move: Move) -> int:
        hand = state.hand()
        return (min_steps(hand, self.cfg)
                - min_steps(hand - move.cards, self.cfg))

    def _score_metrics(self, state: GameState, move: Move,
                       hand: int) -> Tuple[float, float, int, bool, float]:
        """返回 (净分代理, progress, 之后手数, 是否走完, 保权概率)。"""
        after = hand - move.cards
        steps_after = min_steps(after, self.cfg)
        progress = min_steps(hand, self.cfg) - steps_after
        child = state.apply(move, self.cfg)
        p = state.current
        bomb_delta = child.bombs[p] - state.bombs[p]
        keep = keep_lead_prob(state, state.current, move, self.cfg)
        zero_opp = zero_play_opponents(state.tricks, p)
        finish = after == 0
        spring = 0.0
        # 只有已经进入“一手内走完”的区间，春天才具有可兑现性；
        # 局中过早追逐 30 分会牺牲基础分和胜率，实测净分反而下降。
        if zero_opp and not move.is_pass and (finish or steps_after <= 1):
            spring = (keep if not finish else 1.0) * zero_opp \
                * self.cfg.spring_score
        score = (self.objective.bomb_weight * bomb_delta
                 + self.objective.spring_weight * spring
                 + 0.5 * progress
                 - 0.25 * burn_cost(move))
        return score, float(progress), steps_after, finish, keep

    def _rank_win_primary(self, state: GameState,
                          pool: List[Move], hand: int) -> List[Move]:
        progs = [(m, self._progress(state, m)) for m in pool]
        top = max(p for _, p in progs)
        pool = [m for m, p in progs if p == top]

        # 残局兑现: 打完只剩 1 手且对手确定管不起 => 直接锁定胜局。
        fin = []
        for m in pool:
            if min_steps(hand - m.cards, self.cfg) <= 1:
                if keep_lead_prob(state, state.current, m, self.cfg) >= SAFE_P:
                    fin.append(m)
        if fin:
            return sorted(fin, key=lambda m: (-C.total(m.cards),
                                              burn_cost(m), -m.max_rank))

        if self.guaranteed_lead:
            sure = [m for m in pool
                    if burn_cost(m) == 0.0
                    and keep_lead_prob(state, state.current, m, self.cfg) >= 1.0]
            if sure:
                return sorted(sure, key=lambda m: (-C.total(m.cards),
                                                   m.max_rank))

        if state.last_play is None:
            cheap = [m for m in pool if burn_cost(m) == 0.0]
            if cheap:
                pool = cheap
        pool.sort(key=lambda m: (m.max_rank, -C.total(m.cards)))
        return pool

    def rank_moves(self, state: GameState,
                   moves: Optional[List[Move]] = None) -> List[Move]:
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        if len(ms) <= 1:
            return ms

        hand = state.hand()
        out = [m for m in ms if not m.is_pass and m.cards == hand]
        if out:
            return out
        pool = [m for m in ms if not m.is_pass]
        if not pool:
            return ms

        if not self.objective.score_primary:
            return self._rank_win_primary(state, pool, hand)

        # 净分模式：先用原胜率排序得到 win-safe 集合，再只在集合内按
        # 基础分/春天/炸弹净分重排。禁止为了分项收益跳出胜率安全集。
        safe_moves = self._rank_win_primary(state, pool, hand)
        info = []
        for m in safe_moves:
            score, prog, steps_after, finish, keep = self._score_metrics(
                state, m, hand)
            info.append({
                "move": m, "score": score, "progress": prog,
                "steps_after": steps_after, "finish": finish,
                "keep": keep, "bomb": m.type == MoveType.BOMB,
            })
        info.sort(key=lambda x: (-x["score"], -x["progress"],
                                 x["steps_after"], -C.total(x["move"].cards),
                                 x["move"].max_rank))
        return [x["move"] for x in info]

    def act(self, state: GameState,
            moves: Optional[List[Move]] = None) -> Move:
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        eg = self._endgame_move(state, ms)
        if eg is not None:
            return eg
        return self.rank_moves(state, ms)[0]

    # ------------------------------------------------------------------

    def explain(self, state: GameState,
                moves: Optional[List[Move]] = None) -> str:
        hand = state.hand()
        lines = [f"P{state.current} 手牌[{C.total(hand)}] "
                 f"剩 {min_steps(hand, self.cfg)} 手"]
        for m in self.rank_moves(state, moves)[:5]:
            if m.is_pass:
                lines.append("  PASS")
                continue
            after = hand - m.cards
            lines.append(
                f"  {str(m):<38} 进度={self._progress(state, m):+d} "
                f"之后剩 {min_steps(after, self.cfg)} 手 "
                f"保权 {keep_lead_prob(state, state.current, m, self.cfg):.2f}")
        return "\n".join(lines)
