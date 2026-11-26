"""生成 Q(s,a) 训练数据：同一状态下多个候选动作 + 蒙特卡洛回报。

对每个决策状态:
  * 候选 = M1 排序的前 K 个动作（与实际推理时的候选集一致）；
  * 每个候选用 R 次 greedy rollout 求平均净分（去掉单次运气）；
  * 存 (状态特征, 动作特征, mask, 目标)。

运行:
  python tools/gen_q_data.py --games 12000 --workers 10 --k 6 --rollouts 2 \
      --out data/q_train.npz
"""

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.policy_m1 import M1Policy
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState
from src.net.encode import MOVE_DIM, encode, encode_move
from src.search.rollout import rollout

CFG = DEFAULT_RULES


def _gen_range(task):
    wid, start, n_games, eps, k, rollouts, out_prefix, chunk, seed_base = task
    policy = M1Policy(CFG)
    rng = random.Random(seed_base * 977 + wid)
    xs, xm, ms_, ys = [], [], [], []
    samples = 0

    def flush(part):
        if not xs:
            return
        path = f"{out_prefix}.w{wid}.part{part}.npz"
        np.savez_compressed(path,
                            xs=np.stack(xs).astype(np.float16),
                            xm=np.stack(xm).astype(np.float16),
                            mask=np.stack(ms_).astype(np.float16),
                            y=np.stack(ys).astype(np.float16))
        xs.clear(); xm.clear(); ms_.clear(); ys.clear()

    part = 0
    for i in range(n_games):
        st = GameState.new(seed_base + (start + i) * 6197)
        while not st.over:
            actor = st.current
            ranked = policy.rank_moves(st)
            if ranked[0].is_pass:
                mv = ranked[0]
                st = st.apply(mv, CFG)
                continue

            cands = ranked[:k]
            sf = encode(st, actor, CFG)
            mf = np.zeros((k, MOVE_DIM), dtype=np.float32)
            mask = np.zeros(k, dtype=np.float32)
            y = np.zeros(k, dtype=np.float32)
            for j, m in enumerate(cands):
                mf[j] = encode_move(st, m, CFG)
                mask[j] = 1.0
                child = st.apply(m, CFG)
                if child.over:
                    y[j] = child.scores[actor]
                else:
                    tot = 0.0
                    for _ in range(rollouts):
                        tot += rollout(child, CFG).scores[actor]
                    y[j] = tot / rollouts
            xs.append(sf); xm.append(mf); ms_.append(mask)
            ys.append(np.clip(y, -90, 90) / 30.0)
            samples += 1

            if rng.random() < eps:
                mv = rng.choice(cands)
            else:
                mv = cands[0]
            st = st.apply(mv, CFG)

        if (i + 1) % chunk == 0:
            flush(part)
            part += 1
    flush(part)
    return wid, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--out", type=str, default="data/q_train.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    prefix = args.out[:-4] if args.out.endswith(".npz") else args.out
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    per = args.games // args.workers
    tasks = []
    for w in range(args.workers):
        n = per if w < args.workers - 1 else args.games - per * (args.workers - 1)
        tasks.append((w, w * per, n, args.eps, args.k, args.rollouts,
                      prefix, args.chunk, args.seed))

    t0 = time.perf_counter()
    total = 0
    if args.workers <= 1:
        total = _gen_range(tasks[0])[1]
    else:
        with Pool(args.workers) as pool:
            for wid, samples in pool.imap_unordered(_gen_range, tasks):
                total += samples
                print(f"worker {wid}: {samples} samples ({time.perf_counter()-t0:.0f}s)")
    dt = time.perf_counter() - t0
    print(f"done: {total} samples / {args.games} games in {dt:.0f}s")


if __name__ == "__main__":
    main()
