"""对手手牌采样（determinization）+ 过牌证据约束。

必压规则给出的硬信息:
    某玩家在某手牌 T 上"过" ⇒ 他当时手里没有任何能管住 T 的组合。
    手牌只减不增 ⇒ 这条证据对"当前手牌"依然成立。
所以采样时必须拒绝"当前手牌能管住 T"的世界。这一条在实战里极强：
例如对手在单张 A 上过牌，那他手里就绝不会有 A/2/炸弹。
"""

import random
from dataclasses import replace
from typing import List, Sequence, Tuple

from ..fastrun import cards as C
from ..fastrun import moves as M
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState


def _uniform_world(state: GameState, player: int,
                   rng: random.Random) -> GameState:
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


def consistent(world: GameState, state: GameState, player: int,
               cfg: RuleConfig = DEFAULT_RULES) -> bool:
    """采样的世界是否与所有过牌证据相容。"""
    for q, target in state.pass_log:
        if q == player:
            continue
        if M.gen_beats(world.hands[q], target, cfg):
            return False
    return True


def sample_world(state: GameState, player: int, rng: random.Random,
                 cfg: RuleConfig = DEFAULT_RULES,
                 max_tries: int = 40) -> GameState:
    """在证据约束下采一个世界；约束过紧时退化为无约束采样。"""
    world = _uniform_world(state, player, rng)
    for _ in range(max_tries - 1):
        if consistent(world, state, player, cfg):
            return world
        world = _uniform_world(state, player, rng)
    return world


def sample_worlds(state: GameState, player: int, k: int,
                  cfg: RuleConfig = DEFAULT_RULES,
                  seed: int = 0) -> List[GameState]:
    rng = random.Random(seed)
    return [sample_world(state, player, rng, cfg) for _ in range(k)]


def evidence_summary(state: GameState, player: int) -> List[Tuple[int, Move]]:
    """给解释层用: 我掌握的对手过牌证据。"""
    return [(q, t) for q, t in state.pass_log if q != player]
