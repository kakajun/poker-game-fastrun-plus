"""训练 Q(s,a) 网络：掩码 MSE + 候选内 softmax（排序）损失。

运行:
  python tools/train_q.py --data "data/q_train.w*.npz" --epochs 10 \
      --out models/q_v1.pt
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun.config import DEFAULT_RULES
from src.net.model import ENGINE_VERSION, QNet, save


def load_parts(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no data: {pattern}")
    xs, xm, mk, y, deals = [], [], [], [], []
    for f in files:
        d = np.load(f)
        xs.append(d["xs"]); xm.append(d["xm"])
        mk.append(d["mask"]); y.append(d["y"])
        if "deal_id" in d:
            deals.append(d["deal_id"])
        else:
            print(f"  WARNING: {f} has no deal_id; using sample ids")
            deals.append(np.arange(len(d["y"]), dtype=np.int64)
                         + len(deals) * 10**9)
        print(f"  {f}: {len(d['y'])}")
    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(xm).astype(np.float32),
            np.concatenate(mk).astype(np.float32),
            np.concatenate(y).astype(np.float32),
            np.concatenate(deals).astype(np.int64))


def split_by_deal(deal_ids, val_frac, seed):
    uniq = np.unique(deal_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac)) if len(uniq) > 1 else 0
    val_deals = set(int(x) for x in uniq[:n_val])
    is_val = np.fromiter((int(d) in val_deals for d in deal_ids),
                         dtype=bool, count=len(deal_ids))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val), sorted(val_deals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/q_train.w*.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="models/q_v1.pt")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    xs, xm, mask, y, deal_ids = load_parts(args.data)
    n, k, mdim = xm.shape
    print(f"total {n} states, K={k}, state_dim={xs.shape[1]}, move_dim={mdim}")

    ti, vi, val_deals = split_by_deal(deal_ids, args.val, args.split_seed)
    if len(ti) == 0:
        raise SystemExit("deal split produced no training samples")
    print(f"split by deal: train={len(ti)} val={len(vi)} "
          f"({len(val_deals)} val deals)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QNet(xs.shape[1], mdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(ti) // args.batch))

    xt = torch.from_numpy(xs[ti]); mt = torch.from_numpy(xm[ti])
    kt = torch.from_numpy(mask[ti]); yt = torch.from_numpy(y[ti])
    xv = torch.from_numpy(xs[vi]).to(device); mv = torch.from_numpy(xm[vi]).to(device)
    kv = torch.from_numpy(mask[vi]).to(device); yv = torch.from_numpy(y[vi]).to(device)

    def masked_ce(q, y, mask, temp):
        neg = mask < 0.5
        logits = q.masked_fill(neg, -1e9)
        tgt = torch.softmax(y.masked_fill(neg, -1e9) / temp, dim=-1)
        return -(tgt * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        tot_mse = tot_ce = 0.0
        steps = 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            sb = xt[b].to(device, non_blocking=True)
            mb = mt[b].to(device, non_blocking=True)
            kb = kt[b].to(device, non_blocking=True)
            yb = yt[b].to(device, non_blocking=True)
            q = model(sb, mb)
            m = kb > 0.5
            mse = ((q - yb) ** 2 * kb).sum() / kb.sum().clamp(min=1)
            ce = masked_ce(q, yb, kb, args.temp)
            loss = mse + 0.5 * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_mse += float(mse); tot_ce += float(ce); steps += 1

        model.eval()
        with torch.no_grad():
            qv = model(xv, mv)
            m = kv > 0.5
            mae = (((qv - yv).abs() * kv).sum() / kv.sum()).item() * 30.0
            qm = qv.masked_fill(~m, -1e8).argmax(-1)
            ym = yv.masked_fill(~m, -1e8).argmax(-1)
            agree = (qm == ym).float().mean().item()
        print(f"epoch {ep:2d} mse={tot_mse/steps:.4f} ce={tot_ce/steps:.4f} "
              f"val_MAE={mae:.2f}分 top1={agree:.2%}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save(args.out, model.cpu(), {
        "model_type": "q_legacy", "dim": xs.shape[1], "move_dim": mdim,
        "hidden": 256, "samples": n, "features": "legacy_v2_no_leak",
        "rule_hash": DEFAULT_RULES.hash(), "engine_version": ENGINE_VERSION,
        "split": "deal", "val_deals": val_deals, "split_seed": args.split_seed,
    })
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
