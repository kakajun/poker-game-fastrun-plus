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


def save(path: str, model: ValueNet, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = ValueNet(dim=ckpt.get("meta", {}).get("dim", DIM),
                     hidden=ckpt.get("meta", {}).get("hidden", 256))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})
