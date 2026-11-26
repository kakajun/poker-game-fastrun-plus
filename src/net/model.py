"""价值网: 共享主干 + 两个头（净收益 / 胜率）。"""

import torch
import torch.nn as nn

from .encode import DIM


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


def save(path: str, model: ValueNet, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt.get("meta", {})
    if "move_dim" in meta:
        model = QNet(sdim=meta.get("dim", DIM), mdim=meta["move_dim"],
                     hidden=meta.get("hidden", 256))
    else:
        model = ValueNet(dim=meta.get("dim", DIM),
                         hidden=meta.get("hidden", 256))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta
