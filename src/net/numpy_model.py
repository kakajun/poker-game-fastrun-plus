"""纯 numpy 版 PolicyNetV3 前向（history_mode="flatten" ⇒ MLP）。

目的：运行镜像不装 torch。数值与 `src.net.model.PolicyNetV3.forward` 等价，
权重由 `tools/export_npz.py` 从 .pt 导出，接口刻意保持与 torch 版一致：

    win, score = net(state[97], history[128,8], moves[K,28])

`win` 是未过 sigmoid 的 logit，`score` 是净分预测（单位与训练时一致）。
"""

import json
import os
from typing import Dict, Optional

import numpy as np

from .versions import ENGINE_VERSION


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class NumpyPolicyNet:
    """只支持 flatten 历史；GRU 版本请继续用 torch。"""

    model_type = "policy_v3"

    def __init__(self, arrays: Dict[str, np.ndarray], meta: dict):
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        self.meta = dict(meta)

        self.state_dim = int(meta.get("state_dim", 97))
        self.move_dim = int(meta.get("move_dim", 28))
        self.history_steps = int(meta.get("history_steps", 128))
        self.event_dim = int(meta.get("event_dim", 8))
        self.hidden = int(meta.get("hidden", 256))
        if meta.get("history_mode", "flatten") != "flatten":
            raise ValueError("NumpyPolicyNet only supports history_mode='flatten'")

        required = [
            "hist_encoder.0", "trunk.0", "trunk.2", "trunk.4",
            "move_enc.0", "move_enc.2", "head.0", "win_head", "score_head",
        ]
        missing = [n for n in required
                   if f"{n}.weight" not in self.p or f"{n}.bias" not in self.p]
        if missing:
            raise ValueError(f"npz missing layers: {missing}")

    # ---------------------------------------------------------------- 载入

    @classmethod
    def load(cls, npz_path: str, expected: Optional[dict] = None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        meta_path = npz_path + ".json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}

        net = cls(arrays, meta)
        if expected:
            mismatch = {k: (meta.get(k), v) for k, v in expected.items()
                        if meta.get(k) != v}
            if mismatch:
                detail = ", ".join(f"{k}: ckpt={a!r} expected={b!r}"
                                   for k, (a, b) in mismatch.items())
                raise ValueError(f"npz metadata mismatch: {detail}")
        return net

    def expected_meta(self) -> dict:
        return {
            "feature_version": self.meta.get("feature_version"),
            "action_version": self.meta.get("action_version"),
            "engine_version": self.meta.get("engine_version", ENGINE_VERSION),
            "state_dim": self.state_dim,
            "move_dim": self.move_dim,
            "history_steps": self.history_steps,
            "event_dim": self.event_dim,
        }

    # ---------------------------------------------------------------- 前向

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return x @ self.p[f"{name}.weight"].T + self.p[f"{name}.bias"]

    def _mlp(self, x: np.ndarray, names) -> np.ndarray:
        for name in names:
            x = _relu(self._linear(x, name))
        return x

    def forward(self, state: np.ndarray, history: np.ndarray,
                moves: np.ndarray):
        """state (97,), history (128,8), moves (K,28) -> (win K, score K)."""
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        history = np.asarray(history, dtype=np.float32)
        moves = np.asarray(moves, dtype=np.float32)

        # hist_encoder: Linear(128*8 -> 128) + ReLU
        h = self._mlp(history.reshape(1, -1), ["hist_encoder.0"])

        # trunk: Linear(97+128 -> hidden) → hidden → hidden/2
        s = self._mlp(np.concatenate([state, h], axis=1),
                      ["trunk.0", "trunk.2", "trunk.4"])

        # move_enc: Linear(28 -> 96) → 96
        m = self._mlp(moves, ["move_enc.0", "move_enc.2"])

        k = m.shape[0]
        s = np.repeat(s[:, None, :], k, axis=1)       # (1, K, hidden/2)
        x = np.concatenate([s, m[None, :, :]], axis=-1)  # (1, K, h/2+96)

        # head: Linear(→128) + ReLU，然后两个线性头
        x = self._mlp(x, ["head.0"])
        win = self._linear(x, "win_head").reshape(-1)
        score = self._linear(x, "score_head").reshape(-1)
        return win.astype(np.float64), score.astype(np.float64)
