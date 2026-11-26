"""快速 rollout（greedy 策略），用于给 Q(s,a) 生成蒙特卡洛回报。

刻意不用 M1（太慢，每步要算拆牌）；greedy 的偏差在"同一状态的不同候选之间"
是共模的，用来做相对排序足够。
"""

from typing import List

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState


def greedy_move(ms: List[Move]) -> Move:
    pool = [m for m in ms if not m.is_pass]
    if not pool:
        return ms[0]
    normal = [m for m in pool if m.type != MoveType.BOMB]
    return min(normal or pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


def rollout(state: GameState, cfg: RuleConfig = DEFAULT_RULES,
            max_steps: int = 200) -> GameState:
    st = state
    steps = 0
    while not st.over:
        steps += 1
        if steps > max_steps:
            break
        st = st.apply(greedy_move(st.legal_moves(cfg)), cfg)
    return st
