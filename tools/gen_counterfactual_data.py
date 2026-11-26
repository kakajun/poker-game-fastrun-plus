"""生成 v4 反事实教师数据。

每个样本:
  * 只编码当前玩家的合法信息集；
  * 对全部合法动作，在相同隐藏世界、相同 continuation 种子下跑完整对局；
  * 保存每动作的胜率/净分均值及标准误，供相对优势排序训练。

动作以 ragged 格式存储:
  actions(action_offsets[i]:action_offsets[i+1]) 是第 i 个状态的动作特征。

运行示例:
  python tools/gen_counterfactual_data.py --games 200 --workers 4 --worlds 2 \
      --max-states-per-game 16 --opponents greedy,m1,m2 \
      --out data/cf_v3.npz
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import (GreedyMinAgent, M1Agent, M2Agent,
                              PolicyAgent)
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState
from src.eval.score_ev import score_breakdowns
from src.infer.sampler import sample_worlds_report
from src.net.encode_v3 import (ACTION_VERSION, FEATURE_VERSION,
                               HISTORY_STEPS, MOVE_DIM, STATE_DIM,
                               encode_history, encode_move_set, encode_state)
from src.net.versions import ENGINE_VERSION

CFG = DEFAULT_RULES
OPPONENTS = ("greedy", "m1", "m2", "net")
CONTINUATIONS = ("m1", "m2", "net")
POLICY_VERSION = "counterfactual-v2"
_NET_CACHE = {}


def _agent(name: str, seed: int, model_path: str):
    if name == "greedy":
        return GreedyMinAgent()
    if name == "m1":
        return M1Agent(CFG)
    if name == "m2":
        return M2Agent(CFG, endgame_cards=8, endgame_k=4, seed=seed)
    if name == "net":
        if model_path not in _NET_CACHE:
            _NET_CACHE[model_path] = PolicyAgent(CFG, model_path=model_path)
        return _NET_CACHE[model_path]
    raise ValueError(f"unknown agent: {name}")


def _playout(state: GameState, me: int, continuation: str, villain: str,
             seed: int, model_path: str):
    """从 state 打到终局；返回 (win, score, ScoreBreakdown) 或 None。"""
    hero = _agent(continuation, seed, model_path)
    v1 = _agent(villain, seed + 1, model_path)
    v2 = _agent(villain, seed + 2, model_path)
    agents = [None, None, None]
    agents[me] = hero
    vi = 0
    for q in range(3):
        if q != me:
            agents[q] = v1 if vi == 0 else v2
            vi += 1

    st = state
    for _ in range(500):
        if st.over:
            bd = score_breakdowns(
                st.hands, st.tricks, st.bombs, st.winner, CFG)[me]
            return ((1.0 if st.winner == me else 0.0),
                    float(st.scores[me]), bd)
        p = st.current
        mv = agents[p].act(st, st.legal_moves(CFG))
        st = st.apply(mv, CFG)
    return None


def _split_for(deal_id: int, split_seed: int, val_frac: float,
               test_frac: float) -> int:
    """0=train, 1=val, 2=test；按 deal 确定，重复生成保持一致。"""
    h = (deal_id * 2654435761 + split_seed * 2246822519) & 0xFFFFFFFF
    p = h / 4294967296.0
    if p < test_frac:
        return 2
    if p < test_frac + val_frac:
        return 1
    return 0


def _mean_se(x: np.ndarray):
    mean = float(x.mean())
    if len(x) <= 1:
        return mean, 0.0
    return mean, float(x.std(ddof=1) / np.sqrt(len(x)))


def _gen_range(task):
    (wid, start, n_games, seed_base, worlds, max_states, label_every,
     eps, behavior, continuation, opponents, model_path, out_prefix, chunk,
     split_seed, val_frac, test_frac) = task

    rng = random.Random(seed_base * 131 + wid)
    behavior_agent = _agent(behavior, seed_base + wid, model_path)
    keys = ("state", "history", "actions", "win", "score", "win_se",
            "score_se", "base_score", "spring_score", "bomb_score",
            "spring_opponents", "full_spring",
            "baseline_idx", "deal_id", "game_id", "opponent_idx",
            "continuation_idx", "behavior_idx", "split", "world_seed")
    buffers = {s: {k: [] for k in keys} for s in (0, 1, 2)}
    offsets = {s: [0] for s in (0, 1, 2)}
    split_names = {0: "train", 1: "val", 2: "test"}
    samples = 0
    failures = 0
    part = 0

    def flush():
        nonlocal part
        wrote = False
        for split, b in buffers.items():
            if not b["state"]:
                continue
            path = (f"{out_prefix}.{split_names[split]}."
                    f"w{wid}.part{part}.npz")
            np.savez_compressed(
                path,
                state=np.stack(b["state"]).astype(np.float16),
                history=np.stack(b["history"]).astype(np.float16),
                actions=np.concatenate(b["actions"],
                                       axis=0).astype(np.float16),
                action_offsets=np.asarray(offsets[split], dtype=np.int64),
                win=np.concatenate(b["win"]).astype(np.float32),
                score=np.concatenate(b["score"]).astype(np.float32),
                win_se=np.concatenate(b["win_se"]).astype(np.float32),
                score_se=np.concatenate(b["score_se"]).astype(np.float32),
                base_score=np.concatenate(b["base_score"]).astype(np.float32),
                spring_score=np.concatenate(
                    b["spring_score"]).astype(np.float32),
                bomb_score=np.concatenate(b["bomb_score"]).astype(np.float32),
                spring_opponents=np.concatenate(
                    b["spring_opponents"]).astype(np.float32),
                full_spring=np.concatenate(
                    b["full_spring"]).astype(np.float32),
                label_scale=np.asarray(30.0, dtype=np.float32),
                baseline_idx=np.asarray(b["baseline_idx"], dtype=np.int32),
                deal_id=np.asarray(b["deal_id"], dtype=np.int64),
                game_id=np.asarray(b["game_id"], dtype=np.int64),
                opponent_idx=np.asarray(b["opponent_idx"], dtype=np.int8),
                continuation_idx=np.asarray(b["continuation_idx"],
                                            dtype=np.int8),
                behavior_idx=np.asarray(b["behavior_idx"], dtype=np.int8),
                split=np.asarray(b["split"], dtype=np.int8),
                world_seed=np.asarray(b["world_seed"], dtype=np.int64),
                feature_version=np.asarray(FEATURE_VERSION),
                action_version=np.asarray(ACTION_VERSION),
                engine_version=np.asarray(ENGINE_VERSION),
                rule_hash=np.asarray(CFG.hash()),
                state_dim=np.asarray(STATE_DIM, dtype=np.int32),
                move_dim=np.asarray(MOVE_DIM, dtype=np.int32),
                history_steps=np.asarray(HISTORY_STEPS, dtype=np.int32),
            )
            for v in b.values():
                v.clear()
            offsets[split][:] = [0]
            wrote = True
        if wrote:
            part += 1

    for i in range(n_games):
        deal_id = seed_base + (start + i) * 6197
        split = _split_for(deal_id, split_seed, val_frac, test_frac)
        opponent = opponents[(start + i) % len(opponents)]
        opp_idx = OPPONENTS.index(opponent)
        cont_idx = CONTINUATIONS.index(continuation)
        beh_idx = CONTINUATIONS.index(behavior)
        st = GameState.new(deal_id)
        frame = 0
        labeled = 0
        while not st.over:
            ms = st.legal_moves(CFG)
            me = st.current
            if rng.random() < eps:
                mv = rng.choice(ms)
            else:
                mv = behavior_agent.act(st, ms)
            baseline_idx = ms.index(mv)

            should_label = (
                len(ms) > 1
                and frame % label_every == 0
                and (max_states <= 0 or labeled < max_states)
            )
            if should_label:
                world_seed = (deal_id * 1000003 + frame * 7919 + wid) & 0x7FFFFFFF
                report = sample_worlds_report(
                    st, me, worlds, CFG, seed=world_seed)
                failures += report.failures
                win_rows = [[] for _ in ms]
                score_rows = [[] for _ in ms]
                base_rows = [[] for _ in ms]
                spring_rows = [[] for _ in ms]
                bomb_rows = [[] for _ in ms]
                spring_opp_rows = [[] for _ in ms]
                full_rows = [[] for _ in ms]
                # 任一世界不满足 Pass 证据时不生成伪精确教师标签。
                bad = report.failures > 0
                for w_i, world in enumerate(report.worlds):
                    for a_i, action in enumerate(ms):
                        child = world.apply(action, CFG)
                        out = _playout(child, me, continuation, opponent,
                                       seed=world_seed * 31 + w_i,
                                       model_path=model_path)
                        if out is None:
                            bad = True
                            break
                        win_rows[a_i].append(out[0])
                        score_rows[a_i].append(
                            np.clip(out[1], -90, 90) / 30.0)
                        bd = out[2]
                        base_rows[a_i].append(
                            np.clip(bd.base, -90, 90) / 30.0)
                        spring_rows[a_i].append(bd.spring / 30.0)
                        bomb_rows[a_i].append(bd.bomb / 30.0)
                        spring_opp_rows[a_i].append(
                            bd.spring_opponents / 2.0)
                        full_rows[a_i].append(1.0 if bd.full_spring else 0.0)
                    if bad:
                        break
                if not bad:
                    wins = np.asarray(win_rows, dtype=np.float32)
                    scores = np.asarray(score_rows, dtype=np.float32)
                    wm, wse = zip(*[_mean_se(row) for row in wins])
                    sm, sse = zip(*[_mean_se(row) for row in scores])
                    bm = np.asarray([np.mean(row)
                                     for row in base_rows], dtype=np.float32)
                    spm = np.asarray([np.mean(row)
                                      for row in spring_rows], dtype=np.float32)
                    bom = np.asarray([np.mean(row)
                                      for row in bomb_rows], dtype=np.float32)
                    som = np.asarray([np.mean(row)
                                      for row in spring_opp_rows],
                                     dtype=np.float32)
                    fsm = np.asarray([np.mean(row)
                                      for row in full_rows], dtype=np.float32)
                    b = buffers[split]
                    b["state"].append(encode_state(st, me, CFG))
                    b["history"].append(encode_history(st, me))
                    b["actions"].append(encode_move_set(st, ms, CFG))
                    offsets[split].append(offsets[split][-1] + len(ms))
                    b["win"].append(np.asarray(wm, dtype=np.float32))
                    b["score"].append(np.asarray(sm, dtype=np.float32))
                    b["win_se"].append(np.asarray(wse, dtype=np.float32))
                    b["score_se"].append(np.asarray(sse, dtype=np.float32))
                    b["base_score"].append(bm)
                    b["spring_score"].append(spm)
                    b["bomb_score"].append(bom)
                    b["spring_opponents"].append(som)
                    b["full_spring"].append(fsm)
                    b["baseline_idx"].append(baseline_idx)
                    b["deal_id"].append(deal_id)
                    b["game_id"].append(start + i)
                    b["opponent_idx"].append(opp_idx)
                    b["continuation_idx"].append(cont_idx)
                    b["behavior_idx"].append(beh_idx)
                    b["split"].append(split)
                    b["world_seed"].append(world_seed)
                    samples += 1
                    labeled += 1

            st = st.apply(mv, CFG)
            frame += 1

        if (i + 1) % chunk == 0:
            flush()
    flush()
    return wid, samples, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--max-states-per-game", type=int, default=16)
    ap.add_argument("--label-every", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--behavior", choices=CONTINUATIONS, default="m2")
    ap.add_argument("--continuation", choices=CONTINUATIONS, default="m2")
    ap.add_argument("--opponents", type=str, default="greedy,m1,m2")
    ap.add_argument("--model", type=str, default="models/policy_v4.pt")
    ap.add_argument("--out", type=str, default="data/cf_v4.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--split-seed", type=int, default=20261126)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    if not opponents or any(x not in OPPONENTS for x in opponents):
        raise SystemExit(f"invalid opponents: {args.opponents}")
    if args.workers < 1 or args.worlds < 1 or args.label_every < 1:
        raise SystemExit("workers/worlds/label_every must be >= 1")
    if args.workers > args.games:
        raise SystemExit("workers cannot exceed games")
    if not (0.0 <= args.val_frac < 1.0 and 0.0 <= args.test_frac < 1.0
            and args.val_frac + args.test_frac < 1.0):
        raise SystemExit("val/test fractions must be in [0,1) and sum < 1")

    prefix = args.out[:-4] if args.out.endswith(".npz") else args.out
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    per = args.games // args.workers
    tasks = []
    for w in range(args.workers):
        n = per if w < args.workers - 1 else args.games - per * (args.workers - 1)
        tasks.append((w, w * per, n, args.seed, args.worlds,
                      args.max_states_per_game, args.label_every, args.eps,
                      args.behavior, args.continuation, opponents,
                      args.model, prefix, args.chunk, args.split_seed,
                      args.val_frac, args.test_frac))

    t0 = time.perf_counter()
    total = failures = 0
    if args.workers == 1:
        _, total, failures = _gen_range(tasks[0])
    else:
        with Pool(args.workers) as pool:
            for wid, n, f in pool.imap_unordered(_gen_range, tasks):
                total += n
                failures += f
                print(f"worker {wid}: {n} states, {f} sampling failures")

    meta = {
        "policy_version": POLICY_VERSION,
        "feature_version": FEATURE_VERSION,
        "action_version": ACTION_VERSION,
        "engine_version": ENGINE_VERSION,
        "rule_hash": CFG.hash(),
        "state_dim": STATE_DIM,
        "move_dim": MOVE_DIM,
        "history_steps": HISTORY_STEPS,
        "games": args.games,
        "worlds": args.worlds,
        "max_states_per_game": args.max_states_per_game,
        "label_every": args.label_every,
        "behavior": args.behavior,
        "behavior_idx": CONTINUATIONS.index(args.behavior),
        "continuation": args.continuation,
        "opponents": opponents,
        "sampling_failures": failures,
        "samples": total,
        "split": {"train": 0, "val": 1, "test": 2,
                  "split_seed": args.split_seed,
                  "val_frac": args.val_frac, "test_frac": args.test_frac},
        "action_layout": "ragged: actions[action_offsets[i]:action_offsets[i+1]]",
        "labels": ["win", "score", "base_score", "spring_score",
                   "bomb_score", "spring_opponents", "full_spring"],
        "label_scale": 30.0,
        "file_layout": "{prefix}.{train|val|test}.w{worker}.part{n}.npz",
    }
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    dt = time.perf_counter() - t0
    print(f"done: {total} states / {args.games} games, "
          f"{failures} sampling failures, {dt:.1f}s")
    print(f"manifest: {prefix}.json")


if __name__ == "__main__":
    main()
