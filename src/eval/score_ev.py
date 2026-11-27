"""统一终局计分拆解。

规则、M1/M2、数据标签和竞技场都必须使用这里的口径，避免各自重算：

    total = base + spring + bomb

`base` 为对手剩余牌造成的基础分；剩 1 张时按 `protect_one_card` 记 0。
`spring` 为每个零出牌对手的春天分，按玩家视角带符号。
`bomb` 为炸弹累计分。
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig


@dataclass(frozen=True)
class ScoreBreakdown:
    base: int
    spring: int
    bomb: int
    total: int
    spring_opponents: int
    full_spring: bool


def score_breakdowns(hands: Sequence[int], tricks: Sequence[int],
                     bombs: Sequence[int], winner: int,
                     cfg: RuleConfig = DEFAULT_RULES
                     ) -> Tuple[ScoreBreakdown, ...]:
    """返回三名玩家各自的计分拆解。要求终局 hands/tricks/bombs 长度为 3。"""
    if len(hands) != 3 or len(tricks) != 3 or len(bombs) != 3:
        raise ValueError("hands/tricks/bombs must have length 3")
    if winner not in (0, 1, 2):
        raise ValueError(f"invalid winner: {winner}")

    base = [0, 0, 0]
    spring = [0, 0, 0]

    for p in range(3):
        if p == winner:
            continue
        remain = C.total(hands[p])
        base_loss = 0 if (cfg.protect_one_card and remain == 1) else remain
        base[p] -= base_loss
        base[winner] += base_loss
        if tricks[p] == 0:
            spring[p] -= cfg.spring_score
            spring[winner] += cfg.spring_score

    out: List[ScoreBreakdown] = []
    for p in range(3):
        spring_opponents = sum(
            1 for q in range(3) if q != p and tricks[q] == 0)
        out.append(ScoreBreakdown(
            base=base[p],
            spring=spring[p],
            bomb=int(bombs[p]),
            total=base[p] + spring[p] + int(bombs[p]),
            spring_opponents=spring_opponents,
            full_spring=(p == winner and spring_opponents == 2),
        ))
    return tuple(out)


def terminal_scores(hands: Sequence[int], tricks: Sequence[int],
                    bombs: Sequence[int], winner: int,
                    cfg: RuleConfig = DEFAULT_RULES) -> List[int]:
    """兼容旧接口：把统一拆解投影为三家总分。"""
    return [b.total for b in score_breakdowns(
        hands, tricks, bombs, winner, cfg)]


def zero_play_opponents(tricks: Sequence[int], player: int) -> int:
    """该玩家视角下，尚未出过任何牌的对手数量。"""
    return sum(1 for q in range(3) if q != player and tricks[q] == 0)
