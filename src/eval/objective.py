"""净分优先的统一决策目标。

默认 `score_primary`：在胜率不低于最高胜率 `win_guard` 的动作中，选净分最高者。
`win_primary` 仅用于消融/回滚：先胜率，再净分。
"""

from dataclasses import dataclass
from typing import Sequence

OBJECTIVE_VERSION = "v1"


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "score_primary"
    win_guard: float = 0.01
    spring_weight: float = 1.0
    bomb_weight: float = 1.0
    max_extra_steps: int = 1

    def __post_init__(self):
        if self.mode not in ("score_primary", "win_primary"):
            raise ValueError(f"unknown objective mode: {self.mode}")
        if not 0.0 <= self.win_guard <= 1.0:
            raise ValueError("win_guard must be in [0, 1]")
        if self.spring_weight < 0 or self.bomb_weight < 0:
            raise ValueError("score weights must be non-negative")
        if self.max_extra_steps < 0:
            raise ValueError("max_extra_steps must be >= 0")

    @property
    def score_primary(self) -> bool:
        return self.mode == "score_primary"

    def select_index(self, win_probs: Sequence[float],
                     scores: Sequence[float]) -> int:
        """返回被选动作下标；两个数组必须等长且非空。"""
        if len(win_probs) != len(scores):
            raise ValueError("win_probs and scores must have equal length")
        if not win_probs:
            raise ValueError("cannot select from empty actions")

        if not self.score_primary:
            return max(range(len(win_probs)),
                       key=lambda i: (win_probs[i], scores[i]))

        best_win = max(float(w) for w in win_probs)
        cutoff = best_win - self.win_guard
        eligible = [i for i, w in enumerate(win_probs)
                    if float(w) >= cutoff]
        return max(eligible, key=lambda i: (scores[i], win_probs[i],
                                            -i))
