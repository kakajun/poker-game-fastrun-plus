"""M1 策略（规则版，全部结论来自竞技场实测，不是拍脑袋）。

决策顺序:
  1. 只有一个合法动作 / 只能过 -> 执行；
  2. 能一手走完 -> 走完；
  3. 在"不拆自己牌型"的候选里（progress 最大）挑；
  4. 首出时若存在不消耗 2/A/K 的候选，先排除消耗大牌的那些；
  5. 剩下的按 点数最小 -> 出牌张数多 排序。

实测（同牌三座位，600 局，`tools/arena_m1.py`）:
    greedy_min   vs random 78.3%   vs greedy 33.3%
    prog_rank    vs random 82.3%   vs greedy 43.3%
    prog_hold    vs random 83.7%   vs greedy 43.3%   <- 本实现
  线性加权打分（手数 + 控制概率 + 大牌代价）实测反而只有 66%/40%，
  原因是它会把 2/A 当成"高控制概率"打出去，属于典型的短视，已废弃。
"""

from typing import List, Optional

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from .control import burn_cost, keep_lead_prob
from .decompose import min_steps

# 残局兑现的保权阈值（实测 0.7~0.8 最好，0.9 开始过于保守）
SAFE_P = 0.8


class M1Policy:
    def __init__(self, cfg: RuleConfig = DEFAULT_RULES):
        self.cfg = cfg

    # ------------------------------------------------------------------

    def _progress(self, state: GameState, move: Move) -> int:
        hand = state.hand()
        return (min_steps(hand, self.cfg)
                - min_steps(hand - move.cards, self.cfg))

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

        progs = [(m, self._progress(state, m)) for m in pool]
        top = max(p for _, p in progs)
        pool = [m for m, p in progs if p == top]

        # 残局兑现: 打完只剩 1 手且对手确定管不起 => 直接锁定胜局（+6~8pp 实测）
        fin = []
        for m in pool:
            if min_steps(hand - m.cards, self.cfg) <= 1:
                if keep_lead_prob(state, state.current, m, self.cfg) >= SAFE_P:
                    fin.append(m)
        if fin:
            return sorted(fin, key=lambda m: (-C.total(m.cards),
                                              burn_cost(m), -m.max_rank))

        if state.last_play is None:
            cheap = [m for m in pool if burn_cost(m) == 0.0]
            if cheap:
                pool = cheap

        pool.sort(key=lambda m: (m.max_rank, -C.total(m.cards)))
        return pool

    def act(self, state: GameState,
            moves: Optional[List[Move]] = None) -> Move:
        return self.rank_moves(state, moves)[0]

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
