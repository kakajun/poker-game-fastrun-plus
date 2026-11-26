"""自对弈生成价值网训练数据（多进程）。

样本 = (编码状态, 该玩家最终净分/30, 是否获胜)
策略 = M1 + ε 随机（少量探索，避免只覆盖 M1 的窄分布）

运行:
  python tools/gen_value_data.py --games 20000 --workers 10 --eps 0.08 \
      --out data/value_train.npz
"""

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import M1Agent
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState
from src.net.encode import encode

CFG = DEFAULT_RULES


def _gen_range(task):
    wid, start, n_games, eps, out_prefix, chunk, seed_base = task
    policy = M1Agent(CFG)
    rng = random.Random(seed_base * 131 + wid)
    xs, ys, ws, deal_ids, game_ids = [], [], [], [], []
    samples = 0

    def flush(part):
        if not xs:
            return
        path = f"{out_prefix}.w{wid}.part{part}.npz"
        np.savez_compressed(path,
                            x=np.stack(xs).astype(np.float16),
                            y=np.asarray(ys, dtype=np.float32),
                            w=np.asarray(ws, dtype=np.float32),
                            deal_id=np.asarray(deal_ids, dtype=np.int64),
                            game_id=np.asarray(game_ids, dtype=np.int64))
        xs.clear(); ys.clear(); ws.clear(); deal_ids.clear(); game_ids.clear()

    part = 0
    for i in range(n_games):
        deal_id = seed_base + (start + i) * 7919
        st = GameState.new(deal_id)
        traj = []
        while not st.over:
            p = st.current
            moves = st.legal_moves(CFG)
            feats = encode(st, p, CFG)
            if rng.random() < eps:
                mv = rng.choice(moves)
            else:
                mv = policy.act(st, moves)
            traj.append((feats, p))
            st = st.apply(mv, CFG)
        for feats, p in traj:
            xs.append(feats)
            ys.append(min(max(st.scores[p], -90), 90) / 30.0)
            ws.append(1.0 if st.winner == p else 0.0)
            deal_ids.append(deal_id)
            game_ids.append(i)
        samples += len(traj)
        if (i + 1) % chunk == 0:
            flush(part)
            part += 1
    flush(part)
    return wid, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--out", type=str, default="data/value_train.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()

    prefix = args.out[:-4] if args.out.endswith(".npz") else args.out
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    per = args.games // args.workers
    tasks = []
    for w in range(args.workers):
        n = per if w < args.workers - 1 else args.games - per * (args.workers - 1)
        tasks.append((w, w * per, n, args.eps, prefix, args.chunk, args.seed))

    t0 = time.perf_counter()
    total = 0
    if args.workers <= 1:
        total = _gen_range(tasks[0])[1]
    else:
        with Pool(args.workers) as pool:
            for wid, samples in pool.imap_unordered(_gen_range, tasks):
                total += samples
                print(f"worker {wid} done: {samples} samples "
                      f"({(time.perf_counter()-t0):.0f}s)")
    dt = time.perf_counter() - t0
    print(f"done: {total} samples from {args.games} games in {dt:.0f}s "
          f"({args.games/dt:.1f} games/s)")


if __name__ == "__main__":
    main()
