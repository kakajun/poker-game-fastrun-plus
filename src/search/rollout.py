"""快速 rollout。

确定性 rollout 只允许执行一次，不得把重复调用称为 MC；需要随机继续策略时
显式传入 seed/epsilon，并由数据集记录该 continuation_id。
"""

import random
from typing import List, Optional

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
            max_steps: int = 200, seed: Optional[int] = None,
            epsilon: float = 0.0) -> GameState:
    """epsilon=0 时完全确定；epsilon>0 时每个决策按 seed 独立采样。"""
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    rng = random.Random(seed)
    st = state
    steps = 0
    while not st.over:
        steps += 1
        if steps > max_steps:
            break
        ms = st.legal_moves(cfg)
        if epsilon > 0.0 and rng.random() < epsilon:
            mv = rng.choice(ms)
        else:
            mv = greedy_move(ms)
        st = st.apply(mv, cfg)
    return st
