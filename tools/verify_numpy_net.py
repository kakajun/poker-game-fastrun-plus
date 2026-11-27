"""校验 numpy 前向与 torch 前向数值一致（开发机专用，需要 torch）。

用法: PYTHONPATH=. python tools/verify_numpy_net.py models/policy_v4.pt
"""

import sys

import numpy as np
import torch

from src.fastrun.config import DEFAULT_RULES
from src.fastrun.state import GameState
from src.net.encode_v3 import encode_history, encode_move_set, encode_state
from src.net.model import load
from src.net.numpy_model import NumpyPolicyNet


def main(pt_path: str, n_states: int = 60, tol: float = 1e-4) -> int:
    model, meta = load(pt_path, map_location="cpu", require_meta=True)
    model.eval()
    npz_path = pt_path.rsplit(".", 1)[0] + ".npz"
    net = NumpyPolicyNet.load(npz_path)

    cfg = DEFAULT_RULES
    max_win = 0.0
    max_score = 0.0
    checked = 0
    argmax_mismatch = 0
    mismatch_gap = 0.0      # 翻转时两个动作的真实分差（判断是否只是平局噪声）
    tie_tol = 1e-4

    for seed in range(n_states):
        st = GameState.new(seed)
        # 走几步，制造有历史、有 last_play 的中局
        steps = seed % 12
        for _ in range(steps):
            if st.over:
                break
            ms = st.legal_moves(cfg)
            if not ms:
                break
            st = st.apply(ms[seed % len(ms)], cfg)
        if st.over:
            continue

        me = st.current
        ms = st.legal_moves(cfg)
        if not ms:
            continue

        sf = encode_state(st, me, cfg)
        hf = encode_history(st, me)
        mf = encode_move_set(st, ms, cfg)

        with torch.no_grad():
            t_win, t_score = model(
                torch.from_numpy(sf[None, :]),
                torch.from_numpy(hf[None, :, :]),
                torch.from_numpy(mf[None, :, :]))
        t_win = t_win[0].cpu().numpy()
        t_score = t_score[0].cpu().numpy()

        n_win, n_score = net.forward(sf, hf, mf)

        max_win = max(max_win, float(np.abs(t_win - n_win).max()))
        max_score = max(max_score, float(np.abs(t_score - n_score).max()))
        if int(np.argmax(t_score)) != int(np.argmax(n_score)):
            argmax_mismatch += 1
            mismatch_gap = max(mismatch_gap,
                               float(np.sort(t_score)[-1] - np.sort(t_score)[-2]))
        checked += 1

    print(f"states={checked}  max|Δwin|={max_win:.3e}  "
          f"max|Δscore|={max_score:.3e}  score_argmax_mismatch="
          f"{argmax_mismatch}/{checked} (max gap at flip={mismatch_gap:.2e})")
    # 只在"翻转发生在非平局（分差 > tie_tol）"时才判失败
    ok = max_win < tol and max_score < tol and mismatch_gap < tie_tol
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "models/policy_v4.pt"))
