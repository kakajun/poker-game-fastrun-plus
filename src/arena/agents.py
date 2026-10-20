"""基线 agent（固定标尺，不随版本变化）。"""

import random
from typing import List, Optional

from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType
from ..fastrun.state import GameState
from ..eval.policy_m1 import M1Policy


class Agent:
    name = "agent"

    def act(self, state: GameState, moves: List[Move]) -> Move:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{self.name}>"


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, state, moves):
        return self.rng.choice(moves)


class GreedyMinAgent(Agent):
    """旧 HeuristicAgent 的等价行为：普通牌型里挑点数最小的，炸弹留到最后。"""

    name = "greedy_min"

    def act(self, state, moves):
        ms = [m for m in moves if not m.is_pass]
        if not ms:
            return moves[0]
        normal = [m for m in ms if m.type != MoveType.BOMB]
        pool = normal if normal else ms
        return min(pool, key=lambda m: (m.max_rank, -C.total(m.cards)))


class M1Agent(Agent):
    name = "m1"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES):
        self.policy = M1Policy(cfg)

    def act(self, state, moves):
        return self.policy.act(state, moves)


class SearchAgent(Agent):
    """PIMC 雏形（M2/M3 的先行版）：K 个确定性化 + 只搜一轮。"""

    name = "search"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES, k: int = 8,
                 limit: int = 8, seed: int = 0):
        self.cfg = cfg
        self.k = k
        self.limit = limit
        self.seed = seed
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def act(self, state, moves):
        from ..search.pimc import choose
        self.calls += 1
        return choose(state, self.cfg, self.k, self.limit,
                      self.seed * 100003 + self.calls)


class ScriptedAgent(Agent):
    """测试用: 按固定动作序列走，走不通就退回第一个合法动作。"""

    name = "scripted"

    def __init__(self, specs: List[str]):
        self.queue = list(specs)

    def act(self, state, moves):
        from ..fastrun import cards as C
        while self.queue:
            spec = self.queue.pop(0)
            want = C.parse(spec)
            for m in moves:
                if m.cards == want:
                    return m
        return moves[0]
