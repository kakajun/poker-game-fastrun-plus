"""v3 特征、动作历史、残局状态和数据 schema 回归测试。"""

import glob
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.rules import PASS, classify
from src.fastrun.state import GameState
from src.net.encode import DIM as LEGACY_DIM, encode as encode_legacy
from src.net.encode_v3 import (EVENT_DIM, HISTORY_STEPS, MOVE_DIM, STATE_DIM,
                               encode_history, encode_state)
from src.search.endgame import solve_values, solve_with_status
from src.search.rollout import rollout

H = C.parse


class TestActionHistory(unittest.TestCase):
    def test_passes_and_plays_are_ordered(self):
        st = GameState.from_hands(["14*1 3*1", "5*1 6*1", "7*1 8*1"])
        st = st.apply(classify(H("14*1")), validate=True)
        st = st.apply(PASS, validate=True)
        st = st.apply(PASS, validate=True)
        st = st.apply(classify(H("3*1")), validate=True)
        self.assertEqual(len(st.action_log), 4)
        self.assertEqual([mv.type for _, mv in st.action_log],
                         [classify(H("14*1")).type, PASS.type, PASS.type,
                          classify(H("3*1")).type])
        self.assertEqual([q for q, _ in st.action_log], [0, 1, 2, 0])


class TestInformationSet(unittest.TestCase):
    def _state(self, swap=False):
        a, b = "5*1 7*2", "9*1 11*2"
        if swap:
            a, b = b, a
        return GameState.from_hands(
            ["3*2 4*1 8*3", a, b], current=0,
            last_play=classify(H("6*2")), last_player=2,
            action_log=[(2, classify(H("6*2")))])

    def test_hidden_hands_do_not_change_encoding(self):
        s1 = self._state(False)
        s2 = self._state(True)
        # 两个对手都剩 3 张，公开信息完全一致，只有隐藏构成不同。
        self.assertEqual(s1.remaining_cards(), s2.remaining_cards())
        np.testing.assert_array_equal(encode_state(s1, 0), encode_state(s2, 0))
        np.testing.assert_array_equal(encode_legacy(s1, 0),
                                      encode_legacy(s2, 0))

    def test_feature_dimensions_and_history(self):
        st = self._state(False)
        self.assertEqual(encode_state(st, 0).shape, (STATE_DIM,))
        self.assertEqual(encode_legacy(st, 0).shape, (LEGACY_DIM,))
        self.assertNotEqual(LEGACY_DIM, 80)
        h = encode_history(st, 0)
        self.assertEqual(h.shape, (HISTORY_STEPS, EVENT_DIM))
        self.assertEqual(float(h[-1, 2]), 1.0)  # 最后事件由上家 P2 发出
        self.assertAlmostEqual(float(h[-1, 6]), 6.0 / 15.0)


class TestHonestSearch(unittest.TestCase):
    def test_endgame_reports_exactness_and_all_root_actions(self):
        st = GameState.from_hands(["4*1 5*1", "9*2", "10*2"],
                                  current=0, last_play=None)
        values, exact, nodes = solve_values(st)
        legal = st.legal_moves(DEFAULT_RULES)
        self.assertEqual(set(values), set(legal))
        self.assertGreater(nodes, 0)
        self.assertIsInstance(exact, bool)
        result = solve_with_status(st)
        self.assertIn(result.move, legal)
        self.assertEqual(result.exact, exact)

    def test_deterministic_rollout_is_reproducible(self):
        st = GameState.new(20261126)
        a = rollout(st, seed=7)
        b = rollout(st, seed=7)
        self.assertTrue(a.over)
        self.assertEqual(a.winner, b.winner)
        self.assertEqual(a.scores, b.scores)


class TestPairedArena(unittest.TestCase):
    def test_six_layouts_and_paired_bootstrap(self):
        from src.arena.agents import GreedyMinAgent, M1Agent
        from src.arena.match import (paired_bootstrap_diff,
                                     run_duplicate_permutations)

        a = run_duplicate_permutations(
            M1Agent(), [GreedyMinAgent(), GreedyMinAgent()],
            n_deals=2, seed0=11)
        b = run_duplicate_permutations(
            M1Agent(), [GreedyMinAgent(), GreedyMinAgent()],
            n_deals=2, seed0=11)
        self.assertEqual(a.n, 12)
        self.assertEqual(len(a.records), 12)
        self.assertEqual(sorted({r.deal_id for r in a.records}), [11, 12])
        c = run_duplicate_permutations(
            M1Agent(), [GreedyMinAgent(), GreedyMinAgent()],
            n_deals=0, seed0=99, deal_ids=[101, 102])
        self.assertEqual(sorted({r.deal_id for r in c.records}), [101, 102])
        diff = paired_bootstrap_diff(a, b, n_boot=200, seed=3)
        self.assertEqual(diff.n_deals, 2)
        self.assertAlmostEqual(diff.win_diff, 0.0)
        self.assertAlmostEqual(diff.score_diff, 0.0)


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPolicyCheckpoint(unittest.TestCase):
    def test_forward_and_metadata_validation(self):
        import torch
        from src.net.model import (CheckpointMismatch, PolicyNetV3, load,
                                   save)

        model = PolicyNetV3(hidden=32)
        s = torch.zeros(2, STATE_DIM)
        h = torch.zeros(2, HISTORY_STEPS, EVENT_DIM)
        m = torch.zeros(2, 5, MOVE_DIM)
        win, score = model(s, h, m)
        self.assertEqual(tuple(win.shape), (2, 5))
        self.assertEqual(tuple(score.shape), (2, 5))

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p.pt")
            save(path, model, {"hidden": 32})
            _, meta = load(path, expected={
                "feature_version": "v4", "objective_version": "v1"},
                require_meta=True)
            self.assertEqual(meta["state_dim"], STATE_DIM)
            with self.assertRaises(CheckpointMismatch):
                load(path, expected={"feature_version": "v2"},
                     require_meta=True)


class TestDataSchema(unittest.TestCase):
    def test_ragged_counterfactual_schema(self):
        from tools.gen_counterfactual_data import _gen_range

        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, "cf")
            task = (0, 0, 3, 424242, 1, 1, 1, 0.0,
                    "m1", "m1", ["greedy"], os.path.join(td, "unused.pt"),
                    prefix, 100, 20261126, 0.1, 0.1)
            _, n, failures = _gen_range(task)
            self.assertEqual(failures, 0)
            self.assertGreater(n, 0)
            files = glob.glob(prefix + ".*.w0.part*.npz")
            self.assertTrue(files)
            with np.load(files[0]) as d:
                states = d["state"]
                off = d["action_offsets"]
                self.assertEqual(len(off), len(states) + 1)
                self.assertEqual(int(off[-1]), len(d["actions"]))
                self.assertEqual(d["history"].shape[1:],
                                 (HISTORY_STEPS, EVENT_DIM))
                self.assertEqual(len(d["win"]), len(d["actions"]))
                self.assertEqual(len(d["split"]), len(states))
                for key in ("base_score", "spring_score", "bomb_score",
                            "spring_opponents", "full_spring"):
                    self.assertEqual(len(d[key]), len(d["actions"]))
                self.assertEqual(str(d["feature_version"]), "v4")
                self.assertEqual(int(d["state_dim"]), STATE_DIM)
                self.assertEqual(int(d["move_dim"]), MOVE_DIM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
