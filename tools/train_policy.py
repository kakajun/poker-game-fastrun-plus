"""训练 v3 全动作策略网。

数据来自 `tools/gen_counterfactual_data.py`；按 dataset 内的 split/deal 隔离。
主目标是相对优势排序（listwise 或 pairwise），绝对胜率/净分只作辅助监督。

运行:
  python tools/train_policy.py --data "data/cf_v3.w*.npz" \
      --out models/policy_v3.pt --seeds 0,1,2
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun.config import DEFAULT_RULES
from src.net.encode_v3 import (ACTION_VERSION, EVENT_DIM, FEATURE_VERSION,
                               HISTORY_STEPS, MOVE_DIM, STATE_DIM)
from src.net.model import ENGINE_VERSION, PolicyNetV3, save


class Dataset:
    def __init__(self, states, history, actions, offsets, win, score,
                 win_se, score_se, split, deal_id, indices):
        self.states = states
        self.history = history
        self.actions = actions
        self.offsets = offsets
        self.win = win
        self.score = score
        self.win_se = win_se
        self.score_se = score_se
        self.split = split
        self.deal_id = deal_id
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def batch(self, idx):
        rows = self.indices[idx]
        ks = [self.offsets[i + 1] - self.offsets[i] for i in rows]
        kmax = max(ks)
        n = len(rows)
        actions = np.zeros((n, kmax, self.actions.shape[1]), dtype=np.float32)
        win = np.zeros((n, kmax), dtype=np.float32)
        score = np.zeros((n, kmax), dtype=np.float32)
        wse = np.zeros((n, kmax), dtype=np.float32)
        sse = np.zeros((n, kmax), dtype=np.float32)
        mask = np.zeros((n, kmax), dtype=np.float32)
        for j, (row, k) in enumerate(zip(rows, ks)):
            a, b = self.offsets[row], self.offsets[row + 1]
            actions[j, :k] = self.actions[a:b]
            win[j, :k] = self.win[a:b]
            score[j, :k] = self.score[a:b]
            wse[j, :k] = self.win_se[a:b]
            sse[j, :k] = self.score_se[a:b]
            mask[j, :k] = 1.0
        return {
            "state": torch.from_numpy(self.states[rows].astype(np.float32)),
            "history": torch.from_numpy(self.history[rows].astype(np.float32)),
            "actions": torch.from_numpy(actions),
            "win": torch.from_numpy(win),
            "score": torch.from_numpy(score),
            "win_se": torch.from_numpy(wse),
            "score_se": torch.from_numpy(sse),
            "mask": torch.from_numpy(mask),
        }


def load_dataset(patterns):
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    files = sorted({f for p in pats for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no data: {patterns}")
    states, histories, action_parts = [], [], []
    win, score, win_se, score_se = [], [], [], []
    split, deal_id = [], []
    base = 0
    offsets = [0]
    for f in files:
        d = np.load(f)
        required = ("state", "history", "actions", "action_offsets",
                    "win", "score", "win_se", "score_se", "split", "deal_id")
        missing = [k for k in required if k not in d]
        if missing:
            raise SystemExit(f"{f} missing fields: {missing}")
        expected = {
            "feature_version": FEATURE_VERSION,
            "action_version": ACTION_VERSION,
            "engine_version": ENGINE_VERSION,
            "rule_hash": DEFAULT_RULES.hash(),
            "state_dim": STATE_DIM,
            "move_dim": MOVE_DIM,
            "history_steps": HISTORY_STEPS,
        }
        for key, want in expected.items():
            if key not in d:
                raise SystemExit(f"{f} missing metadata: {key}")
            got = d[key].item() if hasattr(d[key], "item") else d[key]
            if got != want:
                raise SystemExit(
                    f"{f} metadata mismatch {key}: {got!r} != {want!r}")
        if ".test." in os.path.basename(f) or np.any(d["split"] == 2):
            raise SystemExit(f"refusing to load test split into training: {f}")
        n = len(d["state"])
        local_off = d["action_offsets"]
        if len(local_off) != n + 1:
            raise SystemExit(f"{f}: bad action_offsets length")
        if n == 0:
            continue
        states.append(d["state"])
        histories.append(d["history"])
        action_parts.append(d["actions"])
        win.append(d["win"])
        score.append(d["score"])
        win_se.append(d["win_se"])
        score_se.append(d["score_se"])
        split.append(d["split"])
        deal_id.append(d["deal_id"])
        for x in local_off[1:]:
            offsets.append(base + int(x))
        base += len(d["actions"])
        print(f"  loaded {f}: {n} states")
    if not states:
        raise SystemExit("dataset is empty")
    return {
        "states": np.concatenate(states),
        "history": np.concatenate(histories),
        "actions": np.concatenate(action_parts),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "win": np.concatenate(win),
        "score": np.concatenate(score),
        "win_se": np.concatenate(win_se),
        "score_se": np.concatenate(score_se),
        "split": np.concatenate(split).astype(np.int8),
        "deal_id": np.concatenate(deal_id).astype(np.int64),
    }


def _subset(data, indices):
    idx = np.asarray(indices, dtype=np.int64)
    offsets = [0]
    action_parts = []
    label_fields = ("win", "score", "win_se", "score_se")
    labels = {k: [] for k in label_fields}
    for i in idx:
        a, b = int(data["offsets"][i]), int(data["offsets"][i + 1])
        action_parts.append(data["actions"][a:b])
        for k in label_fields:
            labels[k].append(data[k][a:b])
        offsets.append(offsets[-1] + b - a)
    out = {
        "states": data["states"][idx],
        "history": data["history"][idx],
        "actions": np.concatenate(action_parts),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "split": data["split"][idx],
        "deal_id": data["deal_id"][idx],
    }
    for k in label_fields:
        out[k] = np.concatenate(labels[k])
    return out


def make_dataset(data, mask):
    return Dataset(data["states"], data["history"], data["actions"],
                   data["offsets"], data["win"], data["score"], data["win_se"],
                   data["score_se"], data["split"], data["deal_id"], mask)


def masked_listwise_loss(logits, target, mask, temp):
    logits = logits.masked_fill(mask < 0.5, -1e9)
    tgt = target.masked_fill(mask < 0.5, -1e9)
    tgt = torch.softmax(tgt / temp, dim=-1)
    return -(tgt * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def pairwise_loss(logits, target, mask, pairs, rng):
    """随机采样候选对；目标只使用胜率/净分差符号。"""
    b, k = logits.shape
    if k < 2:
        return logits.new_zeros(())
    i = torch.randint(0, k, (b, pairs), device=logits.device)
    j = torch.randint(0, k, (b, pairs), device=logits.device)
    valid = (mask.gather(1, i) > 0.5) & (mask.gather(1, j) > 0.5)
    diff = target.gather(1, i) - target.gather(1, j)
    pair_valid = valid & (diff.abs() > 1e-4)
    if pair_valid.sum() == 0:
        return logits.new_zeros(())
    logit_diff = logits.gather(1, i) - logits.gather(1, j)
    labels = (diff > 0).float()
    loss = F.binary_cross_entropy_with_logits(
        logit_diff[pair_valid], labels[pair_valid], reduction="none")
    return loss.mean()


def evaluate(model, ds, args, device):
    model.eval()
    total = 0.0
    count = 0
    regret_w = regret_s = 0.0
    pair_hits = pair_total = 0
    eval_batch = min(args.batch, 64)
    with torch.no_grad():
        for start in range(0, len(ds), eval_batch):
            b = ds.batch(np.arange(start, min(start + eval_batch, len(ds))))
            b = {k: v.to(device) for k, v in b.items()}
            win, score = model(b["state"], b["history"], b["actions"])
            mask = b["mask"]
            loss = masked_listwise_loss(win, b["win"], mask, args.win_temp)
            loss += 0.25 * masked_listwise_loss(
                score, b["score"], mask, args.score_temp)
            total += float(loss) * len(mask)
            count += len(mask)

            masked_win = win.masked_fill(mask < 0.5, -1e9)
            pick = masked_win.argmax(-1, keepdim=True)
            sw = b["win"].gather(1, pick).squeeze(1)
            ss = b["score"].gather(1, pick).squeeze(1)
            best_w = b["win"].masked_fill(mask < 0.5, -1e9).max(-1).values
            best_s = b["score"].masked_fill(mask < 0.5, -1e9).max(-1).values
            regret_w += float((best_w - sw).sum())
            regret_s += float((best_s - ss).sum())

            # 仅统计教师确实有差别的候选对；向量化避免 K^2 的 Python 循环。
            pred_diff = win.unsqueeze(2) - win.unsqueeze(1)
            true_diff = b["win"].unsqueeze(2) - b["win"].unsqueeze(1)
            valid_pair = (mask.unsqueeze(2) > 0.5) & (
                mask.unsqueeze(1) > 0.5)
            ar = torch.arange(win.size(1), device=win.device)
            tri = ar.view(1, -1, 1) < ar.view(1, 1, -1)
            valid_pair &= tri & (true_diff.abs() > 1e-4)
            pair_hits += int(((pred_diff > 0) == (true_diff > 0))[
                valid_pair].sum())
            pair_total += int(valid_pair.sum())
    return {
        "loss": total / max(1, count),
        "regret_win": regret_w / max(1, count),
        "regret_score": regret_s / max(1, count),
        "pairwise_acc": pair_hits / max(1, pair_total),
        "n": count,
    }


def train_one(data, train_idx, val_idx, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    train_ds = make_dataset(data, train_idx)
    val_ds = make_dataset(data, val_idx)
    model = PolicyNetV3(hidden=args.hidden,
                        history_mode=args.history_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    steps_per_epoch = max(1, (len(train_ds) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)

    best = float("inf")
    best_state = None
    best_val = None
    bad_epochs = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_ds))
        tot = 0.0
        steps = 0
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            b = train_ds.batch(idx)
            b = {k: v.to(device) for k, v in b.items()}
            win, score = model(b["state"], b["history"], b["actions"])
            mask = b["mask"]
            if args.loss == "pairwise":
                main_w = pairwise_loss(win, b["win"], mask, args.pairs, rng)
                main_s = pairwise_loss(score, b["score"], mask, args.pairs, rng)
            else:
                main_w = masked_listwise_loss(win, b["win"], mask, args.win_temp)
                main_s = masked_listwise_loss(
                    score, b["score"], mask, args.score_temp)

            m = mask > 0.5
            # 绝对标签只作弱辅助，且按标签标准误降权。
            sample_w = 1.0 / (1.0 + 2.0 * (b["win_se"] + b["score_se"]) / 2.0)
            bce = F.binary_cross_entropy_with_logits(
                win[m], b["win"][m], reduction="none")
            mse = F.mse_loss(score[m], b["score"][m], reduction="none")
            aux = ((bce + 0.25 * mse) * sample_w[m]).mean()
            loss = main_w + 0.25 * main_s + args.aux_weight * aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            steps += 1

        val = evaluate(model, val_ds, args, device)
        print(f"seed {seed} epoch {ep:2d} loss={tot/max(1,steps):.4f} "
              f"val_regret_win={val['regret_win']:.4f} "
              f"val_regret_score={val['regret_score']:.4f} "
              f"pairwise={val['pairwise_acc']:.2%}")
        if val["regret_win"] + 0.25 * val["regret_score"] < best - 1e-5:
            best = val["regret_win"] + 0.25 * val["regret_score"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_val = val
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"seed {seed} early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, (best_val if best_val is not None else val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default=("data/cf_v3.train.w*.npz,"
                             "data/cf_v3.val.w*.npz"),
                    help="只允许 train/val 分片；test 分片会被拒绝")
    ap.add_argument("--out", type=str, default="models/policy_v3.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--history-mode", choices=("flatten", "gru"),
                    default="flatten")
    ap.add_argument("--loss", choices=("listwise", "pairwise"),
                    default="listwise")
    ap.add_argument("--win-temp", type=float, default=0.2)
    ap.add_argument("--score-temp", type=float, default=0.15)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    data = load_dataset(args.data)
    train_idx = np.flatnonzero(data["split"] == 0)
    val_idx = np.flatnonzero(data["split"] == 1)
    if args.max_samples > 0:
        train_idx = train_idx[:args.max_samples]
        val_idx = val_idx[:max(1, args.max_samples // 10)]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise SystemExit("need train and val samples (check generator split)")
    # 不允许同一 deal 跨 train/val。
    overlap = set(int(x) for x in data["deal_id"][train_idx]) & set(
        int(x) for x in data["deal_id"][val_idx])
    if overlap:
        raise SystemExit(f"deal leakage between train/val: {sorted(overlap)[:5]}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"device={device} train={len(train_idx)} val={len(val_idx)} "
          f"actions={len(data['actions'])}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stem, ext = os.path.splitext(args.out)
    best_score = float("inf")
    best_path = None
    best_meta = None
    for seed in seeds:
        model, score, val = train_one(data, train_idx, val_idx, args, seed,
                                      device)
        seed_path = f"{stem}.seed{seed}{ext or '.pt'}"
        meta = {
            "model_type": PolicyNetV3.model_type,
            "feature_version": FEATURE_VERSION,
            "action_version": ACTION_VERSION,
            "engine_version": ENGINE_VERSION,
            "state_dim": STATE_DIM,
            "move_dim": MOVE_DIM,
            "history_steps": HISTORY_STEPS,
            "event_dim": EVENT_DIM,
            "history_mode": args.history_mode,
            "hidden": args.hidden,
            "rule_hash": DEFAULT_RULES.hash(),
            "samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "train_deals": int(len(np.unique(data["deal_id"][train_idx]))),
            "val_deals": int(len(np.unique(data["deal_id"][val_idx]))),
            "seed": seed,
            "seeds": seeds,
            "loss": args.loss,
            "win_temp": args.win_temp,
            "score_temp": args.score_temp,
            "aux_weight": args.aux_weight,
            "val": val,
        }
        save(seed_path, model.cpu(), meta)
        print(f"saved {seed_path} (val utility regret={score:.4f})")
        if score < best_score:
            best_score = score
            best_path = seed_path
            best_meta = meta

    if best_path and os.path.abspath(best_path) != os.path.abspath(args.out):
        shutil.copyfile(best_path, args.out)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(best_meta, f, ensure_ascii=False, indent=2)
    print(f"best: {best_path} -> {args.out}")


if __name__ == "__main__":
    main()
