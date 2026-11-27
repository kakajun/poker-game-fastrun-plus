"""最少手数拆牌（min-tricks decomposition）。

把一手牌拆成最少的"手"（每手是一组合法出牌）。这是"手牌离走完还差几步"的
核心评估量，也是 M1 策略的决策依据。

实现:
  * 小牌（<= EXACT_CARDS）用记忆化 DFS 求精确最优，带节点预算兜底；
  * 大牌用多路贪心（按张数/点数两种偏好 + 少量扰动）取最好；
  * 结果进 LRU 缓存：同一手牌在一局里会反复出现（过牌后再轮到你）。

注意: 拆牌是在"无人干扰"下求最小手数，不等于胜率；控制权由 control.py 补。
"""

from functools import lru_cache
from typing import List, Tuple

from ..fastrun import cards as C
from ..fastrun import moves as M
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move

EXACT_CARDS = 10
NODE_BUDGET = 6000
INF = 99


def _groups_with(hand: int, idx: int, cfg: RuleConfig) -> List[Move]:
    """所有“用到了点数 idx”的合法手（保证每一步都在推进）。"""
    return [m for m in M.gen_lead(hand, cfg) if C.n_at(m.cards, idx) > 0]


def _exact_steps(hand: int, cfg: RuleConfig, memo: dict, counter: list) -> int:
    if hand == 0:
        return 0
    hit = memo.get(hand)
    if hit is not None:
        return hit

    idx = C.min_idx(hand)
    groups = _groups_with(hand, idx, cfg)
    groups.sort(key=lambda m: -C.total(m.cards))

    best = INF
    for mv in groups:
        counter[0] += 1
        if counter[0] > NODE_BUDGET:
            break
        v = 1 + _exact_steps(hand - mv.cards, cfg, memo, counter)
        if v < best:
            best = v
            if best == 1:
                break
    memo[hand] = best
    return best


def _greedy_steps(hand: int, cfg: RuleConfig, prefer: str) -> int:
    steps = 0
    h = hand
    while h:
        cands = M.gen_lead(h, cfg)
        if prefer == "cards":
            cands.sort(key=lambda m: (-C.total(m.cards), m.max_rank))
        elif prefer == "rank":
            cands.sort(key=lambda m: (m.max_rank, -C.total(m.cards)))
        else:  # 结构优先: 顺子/连对/飞机/三张 先走，单张留后
            order = {
                M.MoveType.AIRPLANE_WITH_WINGS: 0,
                M.MoveType.AIRPLANE: 1,
                M.MoveType.DOUBLE_SEQ: 2,
                M.MoveType.STRAIGHT: 3,
                M.MoveType.TRIPLE_WITH_TWO: 4,
                M.MoveType.TRIPLE_WITH_SINGLE: 5,
                M.MoveType.TRIPLE: 6,
                M.MoveType.PAIR: 7,
                M.MoveType.SINGLE: 8,
                M.MoveType.BOMB: 9,
            }
            cands.sort(key=lambda m: (order.get(m.type, 9), -C.total(m.cards),
                                      m.max_rank))
        h -= cands[0].cards
        steps += 1
    return steps


@lru_cache(maxsize=200_000)
def min_steps(hand: int, cfg: RuleConfig = DEFAULT_RULES) -> int:
    """这手牌还需要多少手才能走完（越小越好）。"""
    if hand == 0:
        return 0
    if C.total(hand) <= EXACT_CARDS:
        memo: dict = {}
        steps = _exact_steps(hand, cfg, memo, [0])
        if steps < INF:
            return steps
    return min(_greedy_steps(hand, cfg, "cards"),
               _greedy_steps(hand, cfg, "rank"),
               _greedy_steps(hand, cfg, "struct"))


@lru_cache(maxsize=50_000)
def decompose(hand: int, cfg: RuleConfig = DEFAULT_RULES) -> Tuple[Move, ...]:
    """给出一套（接近）最少手数的拆法，用于解释层与调试。"""
    if hand == 0:
        return tuple()
    steps = min_steps(hand, cfg)
    h = hand
    out: List[Move] = []
    remaining = steps
    while h:
        cands = M.gen_lead(h, cfg)
        cands.sort(key=lambda m: (min_steps(h - m.cards, cfg), -C.total(m.cards),
                                  m.max_rank))
        pick = cands[0]
        out.append(pick)
        h -= pick.cards
        remaining -= 1
        if remaining < 0:
            break
    return tuple(out)


def hand_quality(hand: int, cfg: RuleConfig = DEFAULT_RULES) -> float:
    """手牌结构质量: 手数越少、控制牌越多越好（用于开局评估/解释）。"""
    steps = min_steps(hand, cfg)
    control = (C.n_at(hand, 12) * 3.0        # 2
               + C.n_at(hand, 11) * 2.0      # A
               + C.n_at(hand, 10) * 1.5      # K
               + C.n_at(hand, 9) * 0.8)      # Q
    return control - 1.2 * steps
