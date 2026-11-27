"""无 torch 的策略网 Agent：numpy 前向 + 与 PolicyAgent 相同的决策流程。

行为对齐 `PolicyAgent`：
  * 候选 = 全部合法动作（不过 M1 topK）；
  * checkpoint meta 里的 win_guard / endgame_cards / endgame_k 直接生效；
  * 残局先用精确解，否则按 objective（净分优先 + 胜率守卫）选动作。

运行镜像不需要 torch，权重来自 `models/policy_v4.npz`（tools/export_npz.py 导出）。
"""

import os
from typing import List, Optional

import numpy as np

from ..eval.objective import ObjectiveConfig
from ..eval.policy_m1 import M1Policy
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from ..net.encode_v3 import (ACTION_VERSION, EVENT_DIM, FEATURE_VERSION,
                             HISTORY_STEPS, MOVE_DIM, STATE_DIM,
                             encode_history, encode_move_set, encode_state)
from ..net.numpy_model import NumpyPolicyNet
from ..net.versions import ENGINE_VERSION
from .agents import Agent


class NumpyPolicyAgent(Agent):
    name = "policy_np"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 npz_path: str = "models/policy_v4.npz",
                 objective: Optional[ObjectiveConfig] = None,
                 win_guard: Optional[float] = None,
                 endgame_cards: Optional[int] = None,
                 endgame_k: Optional[int] = None,
                 seed: int = 0):
        self.cfg = cfg
        self.npz_path = npz_path
        self.seed = seed
        self._win_guard_override = win_guard
        self._endgame_override = endgame_cards
        self._endgame_k_override = endgame_k
        self.objective = objective or ObjectiveConfig(
            win_guard=0.01 if win_guard is None else win_guard)
        self.win_guard = self.objective.win_guard
        self.endgame_cards = int(endgame_cards or 0)
        self.endgame_k = int(endgame_k or 6)
        self._net: Optional[NumpyPolicyNet] = None
        self._endgame: Optional[M1Policy] = None
        self._refresh_endgame()

    # ---------------------------------------------------------------- 载入

    def _refresh_endgame(self) -> None:
        self._endgame = (M1Policy(self.cfg,
                                  endgame_cards=self.endgame_cards,
                                  endgame_k=self.endgame_k,
                                  seed=self.seed,
                                  objective=self.objective)
                         if self.endgame_cards else None)

    def _load(self) -> None:
        if self._net is not None:
            return
        expected = {
            "feature_version": FEATURE_VERSION,
            "action_version": ACTION_VERSION,
            "engine_version": ENGINE_VERSION,
            "state_dim": STATE_DIM,
            "move_dim": MOVE_DIM,
            "history_steps": HISTORY_STEPS,
            "event_dim": EVENT_DIM,
        }
        net = NumpyPolicyNet.load(self.npz_path, expected=expected)
        meta = net.meta

        if self._win_guard_override is None and "win_guard" in meta:
            self.objective = ObjectiveConfig(
                mode=self.objective.mode,
                win_guard=float(meta["win_guard"]),
                spring_weight=self.objective.spring_weight,
                bomb_weight=self.objective.bomb_weight,
                max_extra_steps=self.objective.max_extra_steps)
            self.win_guard = self.objective.win_guard

        changed = False
        if self._endgame_override is None and "endgame_cards" in meta:
            self.endgame_cards = int(meta["endgame_cards"])
            changed = True
        if self._endgame_k_override is None and "endgame_k" in meta:
            self.endgame_k = int(meta["endgame_k"])
            changed = True
        if changed:
            self._refresh_endgame()

        self._net = net

    def reset(self) -> None:
        if self._endgame is not None:
            self._endgame.reset()

    # ---------------------------------------------------------------- 推理

    def evaluate(self, state: GameState, moves: Optional[List[Move]] = None):
        """返回 (moves, 胜率概率, 净分预测)，与 PolicyAgent.evaluate 同签名。"""
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        if not ms:
            return [], np.asarray([]), np.asarray([])
        if len(ms) == 1:
            return ms, np.asarray([1.0]), np.asarray([0.0])

        self._load()
        me = state.current
        sf = encode_state(state, me, self.cfg)
        hf = encode_history(state, me)
        mf = encode_move_set(state, ms, self.cfg)

        win, score = self._net.forward(sf, hf, mf)
        prob = 1.0 / (1.0 + np.exp(-win))
        return ms, prob, score

    def act(self, state: GameState,
            moves: Optional[List[Move]] = None) -> Move:
        ms, prob, score = self.evaluate(state, moves)
        if len(ms) <= 1:
            return ms[0]
        if self._endgame is not None:
            eg = self._endgame._endgame_move(state, ms)
            if eg is not None:
                return eg
        best = self.objective.select_index(prob.tolist(), score.tolist())
        return ms[best]

    def rank_moves(self, state: GameState,
                   moves: Optional[List[Move]] = None) -> List[Move]:
        ms, prob, score = self.evaluate(state, moves)
        if self.objective.score_primary:
            def key(i):
                return (-float(score[i]), -float(prob[i]))
        else:
            def key(i):
                return (-float(prob[i]), -float(score[i]))
        return [ms[i] for i in sorted(range(len(ms)), key=key)]
