"""对手手牌采样（determinization）+ 过牌证据约束。

必压规则给出的硬信息:
    某玩家在某手牌 T 上"过" ⇒ 他当时手里没有任何能管住 T 的组合。
    手牌只减不增 ⇒ 这条证据对"当前手牌"依然成立。
所以采样时必须拒绝"当前手牌能管住 T"的世界。这一条在实战里极强：
例如对手在单张 A 上过牌，那他手里就绝不会有 A/2/炸弹。
"""

import random
from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

from ..fastrun import cards as C
from ..fastrun import moves as M
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState


class SamplingError(RuntimeError):
    """约束过紧，无法在一定尝试内采到一致的隐藏世界。"""


@dataclass
class WorldSampleReport:
    worlds: List[GameState]
    failures: int
    max_failures: int

    @property
    def total(self) -> int:
        return len(self.worlds)


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
                 max_tries: int = 40, strict: bool = True) -> GameState:
    """在证据约束下采一个世界。

    strict=True 时失败显式抛错；只有明确需要退避的调用方才允许 strict=False。
    """
    fallback = None
    for _ in range(max(1, max_tries)):
        world = _uniform_world(state, player, rng)
        fallback = world
        if consistent(world, state, player, cfg):
            return world
    if strict:
        raise SamplingError(
            f"no consistent world for player {player} after {max_tries} tries")
    return fallback


def sample_worlds(state: GameState, player: int, k: int,
                  cfg: RuleConfig = DEFAULT_RULES,
                  seed: int = 0, max_tries: int = 40,
                  strict: bool = True) -> List[GameState]:
    rng = random.Random(seed)
    return [sample_world(state, player, rng, cfg, max_tries, strict)
            for _ in range(k)]


def sample_worlds_report(state: GameState, player: int, k: int,
                         cfg: RuleConfig = DEFAULT_RULES,
                         seed: int = 0, max_tries: int = 40
                         ) -> WorldSampleReport:
    """数据生成专用：逐世界尝试，失败世界按无约束退避并显式计数。"""
    rng = random.Random(seed)
    worlds: List[GameState] = []
    failures = 0
    for _ in range(k):
        try:
            world = sample_world(state, player, rng, cfg, max_tries,
                                 strict=True)
        except SamplingError:
            failures += 1
            world = sample_world(state, player, rng, cfg, 1, strict=False)
        worlds.append(world)
    return WorldSampleReport(worlds=worlds, failures=failures,
                             max_failures=max_tries)


def evidence_summary(state: GameState, player: int) -> List[Tuple[int, Move]]:
    """给解释层用: 我掌握的对手过牌证据。"""
    return [(q, t) for q, t in state.pass_log if q != player]
