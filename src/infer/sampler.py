"""对手手牌采样（determinization）。

第一版只保证"物理一致"：未见牌按对手剩余张数随机分配。
M2 要在此基础上加入约束（本轮 Pass 过 ⇒ 该玩家手里没有任何能管住当前牌的
组合；这在本规则里是硬证据，价值很高）。
"""

import random
from dataclasses import replace
from typing import List, Tuple

from ..fastrun import cards as C
from ..fastrun.state import GameState


def sample_world(state: GameState, player: int,
                 rng: random.Random) -> GameState:
    """随机生成一个与 player 视角一致的完整世界（只替换对手手牌）。"""
    un = C.FULL_DECK - state.played - state.hands[player]
    pool: List[int] = []
    for i in range(C.NUM_RANKS):
        pool.extend([i] * C.n_at(un, i))
    rng.shuffle(pool)

    sizes = {q: C.total(state.hands[q]) for q in range(3) if q != player}
    hands = list(state.hands)
    ptr = 0
    for q in sorted(sizes):
        n = sizes[q]
        cnt = [0] * C.NUM_RANKS
        for idx in pool[ptr:ptr + n]:
            cnt[idx] += 1
        hands[q] = C.make(cnt)
        ptr += n
    return replace(state, hands=tuple(hands))


def sample_worlds(state: GameState, player: int, k: int,
                  seed: int = 0) -> List[GameState]:
    rng = random.Random(seed)
    return [sample_world(state, player, rng) for _ in range(k)]
