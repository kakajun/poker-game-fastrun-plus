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

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state, moves):
        return self.policy.act(state, moves)


class M2Agent(Agent):
    """M1 + 残局精确解（约束采样 + max-n 搜索）。"""

    name = "m2"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 endgame_cards: int = 8, endgame_k: int = 8, seed: int = 0,
                 guaranteed_lead: bool = True):
        self.policy = M1Policy(cfg, endgame_cards=endgame_cards,
                               endgame_k=endgame_k, seed=seed,
                               guaranteed_lead=guaranteed_lead)

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state, moves):
        return self.policy.act(state, moves)


class ValueAgent(Agent):
    """M1 候选 + 价值网 1-ply 评估（M3-A）。

    torch 延迟导入：WSL 里没有 torch 时其它模块仍可正常使用。
    """

    name = "value"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 model_path: str = "models/value_v1.pt",
                 top_k: int = 8, win_weight: float = 1.0,
                 prior: float = 0.05, device: str = "cuda"):
        self.cfg = cfg
        self.policy = M1Policy(cfg)
        self.top_k = top_k
        self.win_weight = win_weight
        self.prior = prior
        self.model_path = model_path
        self.device = device
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from ..net.model import ENGINE_VERSION, load
        from ..net.encode import DIM
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        expected = {"model_type": "value_legacy", "dim": DIM,
                    "engine_version": ENGINE_VERSION,
                    "rule_hash": self.cfg.hash()}
        model, _ = load(self.model_path, map_location=self.device,
                        expected=expected, require_meta=True)
        model.to(self.device)
        self._torch = torch
        self._model = model
        self._dim = DIM

    def reset(self) -> None:
        self.policy.reset()

    def act(self, state, moves):
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        if len(ms) <= 1:
            return ms[0]
        ranked = self.policy.rank_moves(state, ms)
        if ranked[0].is_pass:
            return ranked[0]
        cands = ranked[:self.top_k]
        if len(cands) == 1:
            return cands[0]

        self._load()
        from ..net.encode import encode
        import numpy as np
        me = state.current
        feats = np.stack([encode(state.apply(m, self.cfg), me, self.cfg)
                          for m in cands])
        with self._torch.no_grad():
            xb = self._torch.from_numpy(feats).to(self.device)
            s, logit = self._model(xb)
            vals = (s + self.win_weight * self._torch.sigmoid(logit)).cpu().numpy()
        best = int(np.argmax(vals + self.prior * np.linspace(0, -1, len(cands))))
        return cands[best]


class SearchValueAgent(Agent):
    """M3-B: 确定性化 + 本轮 rollout + 价值网叶子估值。"""

    name = "svalue"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 model_path: str = "models/value_v1.pt",
                 k: int = 6, top_k: int = 8, win_weight: float = 1.0,
                 device: str = "cuda", seed: int = 0):
        self.cfg = cfg
        self.policy = M1Policy(cfg, guaranteed_lead=True)
        self.model_path = model_path
        self.k = k
        self.top_k = top_k
        self.win_weight = win_weight
        self.device = device
        self.seed = seed
        self.calls = 0
        self._model = None

    def reset(self) -> None:
        self.calls = 0
        self.policy.reset()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from ..net.model import ENGINE_VERSION, load
        from ..net.encode import DIM
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        expected = {"model_type": "value_legacy", "dim": DIM,
                    "engine_version": ENGINE_VERSION,
                    "rule_hash": self.cfg.hash()}
        model, _ = load(self.model_path, map_location=self.device,
                        expected=expected, require_meta=True)
        model.to(self.device)
        self._torch = torch
        self._model = model

    def act(self, state, moves):
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        if len(ms) <= 1:
            return ms[0]
        ranked = self.policy.rank_moves(state, ms)
        if ranked[0].is_pass:
            return ranked[0]
        cands = ranked[:self.top_k]
        if len(cands) == 1:
            return cands[0]

        from ..infer.sampler import SamplingError, sample_worlds
        from ..search.pimc import _rollout_trick
        from ..net.encode import encode
        import numpy as np

        self._load()
        me = state.current
        self.calls += 1
        try:
            worlds = sample_worlds(state, me, self.k, self.cfg,
                                   seed=self.seed * 104729 + self.calls)
        except SamplingError:
            return cands[0]

        feats = []
        for m in cands:
            for w in worlds:
                st = _rollout_trick(w.apply(m, self.cfg), self.cfg)
                feats.append(encode(st, me, self.cfg))

        with self._torch.no_grad():
            xb = self._torch.from_numpy(np.stack(feats)).to(self.device)
            s, logit = self._model(xb)
            v = (s + self.win_weight * self._torch.sigmoid(logit)).cpu().numpy()
        v = v.reshape(len(cands), len(worlds)).mean(axis=1)
        return cands[int(np.argmax(v))]


class QAgent(Agent):
    """Q(s,a): 在 M1 候选上直接打分取最优。"""

    name = "q"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 model_path: str = "models/q_v1.pt",
                 top_k: int = 6, prior: float = 0.0,
                 device: str = "cuda", guaranteed_lead: bool = False):
        self.cfg = cfg
        self.policy = M1Policy(cfg, guaranteed_lead=guaranteed_lead)
        self.model_path = model_path
        self.top_k = top_k
        self.prior = prior
        self.device = device
        self._model = None

    def reset(self) -> None:
        self.policy.reset()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from ..net.model import ENGINE_VERSION, load
        from ..net.encode import DIM, MOVE_DIM
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        expected = {"model_type": "q_legacy", "dim": DIM,
                    "move_dim": MOVE_DIM, "engine_version": ENGINE_VERSION,
                    "rule_hash": self.cfg.hash()}
        model, _ = load(self.model_path, map_location=self.device,
                        expected=expected, require_meta=True)
        model.to(self.device)
        self._torch = torch
        self._model = model

    def act(self, state, moves):
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        if len(ms) <= 1:
            return ms[0]
        ranked = self.policy.rank_moves(state, ms)
        if ranked[0].is_pass:
            return ranked[0]
        cands = ranked[:self.top_k]
        if len(cands) == 1:
            return cands[0]

        self._load()
        import numpy as np
        from ..net.encode import encode, encode_move
        me = state.current
        sf = encode(state, me, self.cfg)
        mf = np.stack([encode_move(state, m, self.cfg) for m in cands])
        with self._torch.no_grad():
            s = self._torch.from_numpy(sf[None, :]).to(self.device)
            m = self._torch.from_numpy(mf[None, :, :]).to(self.device)
            q = self._model(s, m)[0].cpu().numpy()
        if self.prior:
            q = q + self.prior * np.linspace(0.0, -1.0, len(cands))
        return cands[int(np.argmax(q))]


class PolicyAgent(Agent):
    """v3 全动作策略网：对全部合法动作打分，不经过 M1 topK。"""

    name = "policy_v3"

    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 model_path: str = "models/policy_v3.pt",
                 device: str = "cuda", win_tie_epsilon: float = 1e-6):
        self.cfg = cfg
        self.model_path = model_path
        self.device = device
        self.win_tie_epsilon = win_tie_epsilon
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from ..net.encode_v3 import (ACTION_VERSION, EVENT_DIM,
                                     FEATURE_VERSION, HISTORY_STEPS, MOVE_DIM,
                                     STATE_DIM)
        from ..net.model import ENGINE_VERSION, load
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        expected = {
            "feature_version": FEATURE_VERSION,
            "action_version": ACTION_VERSION,
            "engine_version": ENGINE_VERSION,
            "state_dim": STATE_DIM,
            "move_dim": MOVE_DIM,
            "history_steps": HISTORY_STEPS,
            "event_dim": EVENT_DIM,
        }
        model, _ = load(self.model_path, map_location=self.device,
                        expected=expected, require_meta=True)
        model.to(self.device)
        self._torch = torch
        self._model = model

    def evaluate(self, state, moves=None):
        """返回 (moves, 胜率概率, 净分预测)，供策略与解释层共用。"""
        import numpy as np
        ms = list(state.legal_moves(self.cfg) if moves is None else moves)
        if not ms:
            return [], np.asarray([]), np.asarray([])
        if len(ms) == 1:
            return ms, np.asarray([1.0]), np.asarray([0.0])

        self._load()
        from ..net.encode_v3 import (encode_history, encode_move_set,
                                     encode_state)
        me = state.current
        sf = encode_state(state, me, self.cfg)
        hf = encode_history(state, me)
        mf = encode_move_set(state, ms, self.cfg)
        with self._torch.no_grad():
            s = self._torch.from_numpy(sf[None, :]).to(self.device)
            h = self._torch.from_numpy(hf[None, :, :]).to(self.device)
            m = self._torch.from_numpy(mf[None, :, :]).to(self.device)
            win, score = self._model(s, h, m)
            win = win[0].cpu().numpy()
            score = score[0].cpu().numpy()
        return ms, 1.0 / (1.0 + np.exp(-win)), score

    def act(self, state, moves=None):
        import numpy as np
        ms, prob, score = self.evaluate(state, moves)
        if len(ms) <= 1:
            return ms[0]
        best_win = float(prob.max())
        eligible = np.flatnonzero(prob >= best_win - self.win_tie_epsilon)
        best = int(eligible[np.argmax(score[eligible])])
        return ms[best]

    def rank_moves(self, state, moves=None):
        ms, prob, score = self.evaluate(state, moves)
        order = sorted(range(len(ms)),
                       key=lambda i: (-float(prob[i]), -float(score[i])))
        return [ms[i] for i in order]


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
