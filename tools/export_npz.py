"""把 PolicyNetV3 的 .pt 权重导出成纯 numpy 可推理的 .npz。

只在开发机（有 torch）跑一次；运行镜像里不需要 torch。
仅支持 history_mode="flatten"（纯 MLP）。

用法: PYTHONPATH=. python tools/export_npz.py models/policy_v4.pt
"""

import json
import os
import sys

import numpy as np
import torch  # noqa: F401  仅导出时使用

from src.net.encode_v3 import EVENT_DIM, HISTORY_STEPS, MOVE_DIM, STATE_DIM
from src.net.model import load
from src.net.versions import ENGINE_VERSION


def main(path: str) -> int:
    model, meta = load(path, map_location="cpu", require_meta=True)
    if getattr(model, "history_mode", None) != "flatten":
        print(f"unsupported history_mode={model.history_mode!r}, only flatten")
        return 2
    if meta.get("state_dim") != STATE_DIM or meta.get("move_dim") != MOVE_DIM:
        print(f"dimension mismatch: {meta.get('state_dim')}/{meta.get('move_dim')}"
              f" != {STATE_DIM}/{MOVE_DIM}")
        return 2

    out = os.path.splitext(path)[0] + ".npz"
    weights = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in model.state_dict().items()}
    np.savez_compressed(out, **weights)

    payload = dict(meta)
    payload.update({
        "history_steps": HISTORY_STEPS,
        "event_dim": EVENT_DIM,
        "state_dim": STATE_DIM,
        "move_dim": MOVE_DIM,
        "history_mode": "flatten",
        "engine_version": meta.get("engine_version", ENGINE_VERSION),
        "npz": os.path.basename(out),
    })
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    size = os.path.getsize(out) / 1024 / 1024
    print(f"wrote {out} ({size:.2f} MB, {len(weights)} tensors)")
    print(f"  hidden={meta.get('hidden')} val_regret="
          f"{meta.get('val', {}).get('regret_score')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "models/policy_v4.pt"))
