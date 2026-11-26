"""价值网与全动作策略网。

`PolicyNetV3` 直接给全部合法动作打分，不接受 M1 topK 剪枝；
checkpoint 可绑定规则/特征/动作版本，加载时严格校验。
"""

from typing import Optional

import torch
import torch.nn as nn

from .encode import DIM
from .encode_v3 import (ACTION_VERSION, EVENT_DIM, FEATURE_VERSION,
                        HISTORY_STEPS, MOVE_DIM, STATE_DIM)
from .versions import ENGINE_VERSION


class CheckpointMismatch(RuntimeError):
    """checkpoint 与当前规则/特征/动作版本不兼容。"""


class ValueNet(nn.Module):
    def __init__(self, dim: int = DIM, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.score = nn.Linear(hidden // 2, 1)   # 预测 分数/30
        self.win = nn.Linear(hidden // 2, 1)     # 预测 是否获胜

    def forward(self, x):
        h = self.trunk(x)
        return self.score(h).squeeze(-1), self.win(h).squeeze(-1)


class QNet(nn.Module):
    """Q(s,a): 状态特征 + 动作特征 -> 预计净分。"""

    def __init__(self, sdim: int = DIM, mdim: int = 19, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(sdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.move_enc = nn.Sequential(
            nn.Linear(mdim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden // 2 + 64, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s, m):
        h = self.trunk(s).unsqueeze(1)
        me = self.move_enc(m)
        x = torch.cat([h.expand(-1, me.size(1), -1), me], dim=-1)
        return self.head(x).squeeze(-1)


class PolicyNetV3(nn.Module):
    """全动作策略网：状态 + 有序历史 + 每个合法动作 -> 胜率/净分优势。"""

    model_type = "policy_v3"

    def __init__(self, state_dim: int = STATE_DIM, move_dim: int = MOVE_DIM,
                 history_steps: int = HISTORY_STEPS,
                 event_dim: int = EVENT_DIM, hidden: int = 256,
                 history_mode: str = "flatten"):
        super().__init__()
        if history_mode not in ("flatten", "gru"):
            raise ValueError(f"unknown history_mode: {history_mode}")
        self.state_dim = state_dim
        self.move_dim = move_dim
        self.history_steps = history_steps
        self.event_dim = event_dim
        self.hidden = hidden
        self.history_mode = history_mode

        if history_mode == "flatten":
            hist_out = 128
            self.hist_encoder = nn.Sequential(
                nn.Linear(history_steps * event_dim, hist_out), nn.ReLU(),
            )
        else:
            hist_out = 64
            self.hist_encoder = nn.GRU(event_dim, hist_out, batch_first=True)

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + hist_out, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.move_enc = nn.Sequential(
            nn.Linear(move_dim, 96), nn.ReLU(),
            nn.Linear(96, 96), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden // 2 + 96, 128), nn.ReLU(),
        )
        self.win_head = nn.Linear(128, 1)
        self.score_head = nn.Linear(128, 1)

    def forward(self, state, history, moves):
        if self.history_mode == "flatten":
            h = self.hist_encoder(history.flatten(1))
        else:
            _, h = self.hist_encoder(history)
            h = h[-1]
        s = self.trunk(torch.cat([state, h], dim=-1)).unsqueeze(1)
        m = self.move_enc(moves)
        x = self.head(torch.cat([s.expand(-1, m.size(1), -1), m], dim=-1))
        return self.win_head(x).squeeze(-1), self.score_head(x).squeeze(-1)


def save(path: str, model, meta: dict) -> None:
    payload = dict(meta)
    if isinstance(model, PolicyNetV3):
        payload.setdefault("model_type", model.model_type)
        payload.setdefault("feature_version", FEATURE_VERSION)
        payload.setdefault("action_version", ACTION_VERSION)
        payload.setdefault("engine_version", ENGINE_VERSION)
        payload.setdefault("state_dim", model.state_dim)
        payload.setdefault("move_dim", model.move_dim)
        payload.setdefault("history_steps", model.history_steps)
        payload.setdefault("event_dim", model.event_dim)
        payload.setdefault("history_mode", model.history_mode)
    torch.save({"state_dict": model.state_dict(), "meta": payload}, path)


def _check_meta(meta: dict, expected: Optional[dict]) -> None:
    if not expected:
        return
    mismatch = {k: (meta.get(k), v) for k, v in expected.items()
                if meta.get(k) != v}
    if mismatch:
        detail = ", ".join(f"{k}: ckpt={a!r} expected={b!r}"
                           for k, (a, b) in mismatch.items())
        raise CheckpointMismatch(detail)


def load(path: str, map_location="cpu", expected: Optional[dict] = None,
         require_meta: bool = False):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt.get("meta", {})
    if require_meta and not meta:
        raise CheckpointMismatch("checkpoint has no metadata")
    _check_meta(meta, expected)

    if meta.get("model_type") == PolicyNetV3.model_type:
        model = PolicyNetV3(
            state_dim=meta.get("state_dim", STATE_DIM),
            move_dim=meta.get("move_dim", MOVE_DIM),
            history_steps=meta.get("history_steps", HISTORY_STEPS),
            event_dim=meta.get("event_dim", EVENT_DIM),
            hidden=meta.get("hidden", 256),
            history_mode=meta.get("history_mode", "flatten"),
        )
    elif "move_dim" in meta:
        model = QNet(sdim=meta.get("dim", DIM), mdim=meta["move_dim"],
                     hidden=meta.get("hidden", 256))
    else:
        model = ValueNet(dim=meta.get("dim", DIM),
                         hidden=meta.get("hidden", 256))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta
