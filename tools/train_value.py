"""训练价值网（监督学习，目标 = 最终净分 + 是否获胜）。

运行:
  python tools/train_value.py --data data/value_train.npz --epochs 12 --out models/value_v1.pt
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun.config import DEFAULT_RULES
from src.net.model import ENGINE_VERSION, ValueNet, save


def load_parts(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no data: {pattern}")
    xs, ys, ws, deals = [], [], [], []
    for f in files:
        d = np.load(f)
        xs.append(d["x"]); ys.append(d["y"]); ws.append(d["w"])
        if "deal_id" in d:
            deals.append(d["deal_id"])
        else:
            print(f"  WARNING: {f} has no deal_id; using sample ids")
            deals.append(np.arange(len(d["y"]), dtype=np.int64) + len(deals) * 10**9)
        print(f"  loaded {f}: {len(d['y'])}")
    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(ys), np.concatenate(ws),
            np.concatenate(deals).astype(np.int64))


def split_by_deal(deal_ids: np.ndarray, val_frac: float, seed: int):
    """按 deal 分组切分，同一局的状态只能进入一侧。"""
    uniq = np.unique(deal_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac)) if len(uniq) > 1 else 0
    val_deals = set(int(x) for x in uniq[:n_val])
    is_val = np.fromiter((int(d) in val_deals for d in deal_ids),
                         dtype=bool, count=len(deal_ids))
    vi = np.flatnonzero(is_val)
    ti = np.flatnonzero(~is_val)
    if len(ti) == 0:
        raise SystemExit("deal split produced no training samples")
    return ti, vi, sorted(val_deals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/value_train.w*.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/value_v1.pt")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    x, y, w, deal_ids = load_parts(args.data)
    n = len(y)
    print(f"total {n} samples, dim={x.shape[1]}")
    tr_idx, val_idx, val_deals = split_by_deal(
        deal_ids, args.val, args.split_seed)
    print(f"split by deal: train={len(tr_idx)} val={len(val_idx)} "
          f"({len(val_deals)} val deals)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueNet(dim=x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(tr_idx) // args.batch))

    xt = torch.from_numpy(x[tr_idx]); yt = torch.from_numpy(y[tr_idx])
    wt = torch.from_numpy(w[tr_idx])
    xv = torch.from_numpy(x[val_idx]).to(device)
    yv = torch.from_numpy(y[val_idx]).to(device)
    wv = torch.from_numpy(w[val_idx]).to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        steps = 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            xb = xt[b].to(device, non_blocking=True)
            yb = yt[b].to(device, non_blocking=True)
            wb = wt[b].to(device, non_blocking=True)
            s, logit = model(xb)
            loss = mse(s, yb) + 0.5 * bce(logit, wb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            steps += 1

        model.eval()
        with torch.no_grad():
            sv, lv = model(xv)
            mae = (sv - yv).abs().mean().item() * 30.0
            acc = ((lv > 0) == (wv > 0.5)).float().mean().item()
        print(f"epoch {ep:2d} loss={tot/max(1,steps):.4f} "
              f"val_MAE={mae:.2f}分 val_win_acc={acc:.2%}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save(args.out, model.cpu(), {
        "model_type": "value_legacy", "dim": x.shape[1], "hidden": 256,
        "samples": n, "features": "legacy_v2_no_leak",
        "rule_hash": DEFAULT_RULES.hash(), "engine_version": ENGINE_VERSION,
        "split": "deal", "val_deals": val_deals, "split_seed": args.split_seed,
    })
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
