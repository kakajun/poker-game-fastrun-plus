"""统一计分拆解与净分优先目标测试。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.objective import ObjectiveConfig
from src.eval.score_ev import (score_breakdowns, terminal_scores,
                               zero_play_opponents)
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.rules import classify
from src.fastrun.state import GameState

H = C.parse


class TestScoreBreakdown(unittest.TestCase):
    def test_base_and_protect_one_card(self):
        hands = [0, H("4*5"), H("9*1")]
        out = score_breakdowns(hands, (3, 2, 1), (0, 0, 0), 0)
        self.assertEqual([x.base for x in out], [5, -5, 0])
        self.assertEqual([x.spring for x in out], [0, 0, 0])
        self.assertEqual([x.total for x in out], [5, -5, 0])

    def test_partial_spring_is_thirty(self):
        hands = [0, H("4*5"), H("9*1")]
        out = score_breakdowns(hands, (3, 0, 1), (0, 0, 0), 0)
        self.assertEqual(out[0].base, 5)
        self.assertEqual(out[0].spring, 30)
        self.assertEqual(out[0].total, 35)
        self.assertEqual(out[0].spring_opponents, 1)
        self.assertFalse(out[0].full_spring)
        self.assertEqual(out[1].spring, -30)

    def test_full_spring_is_sixty(self):
        hands = [0, H("4*5"), H("9*1")]
        out = score_breakdowns(hands, (3, 0, 0), (0, 0, 0), 0)
        self.assertEqual(out[0].spring, 60)
        self.assertEqual(out[0].total, 65)
        self.assertEqual(out[0].spring_opponents, 2)
        self.assertTrue(out[0].full_spring)

    def test_bomb_component_and_zero_sum(self):
        hands = [0, H("4*5"), H("9*1")]
        out = score_breakdowns(hands, (3, 2, 1), (20, -10, -10), 0)
        self.assertEqual([x.bomb for x in out], [20, -10, -10])
        self.assertEqual([x.total for x in out], [25, -15, -10])
        self.assertEqual(sum(x.total for x in out), 0)

    def test_bomb_refund_when_beaten(self):
        st = GameState.from_hands(
            [H("9*4 3*1"), H("10*4 4*1"), H("8*4 5*1")],
            current=2, last_play=None)
        st = st.apply(classify(H("8*4")), validate=True)
        self.assertEqual(st.bombs, (-10, -10, 20))
        st = st.apply(classify(H("9*4")), validate=True)
        self.assertEqual(st.bombs, (20, -10, -10))
        st = st.apply(classify(H("10*4")), validate=True)
        self.assertEqual(st.bombs, (-10, 20, -10))
        self.assertEqual(sum(st.bombs), 0)

    def test_game_state_uses_unified_terminal_scores(self):
        st = GameState.from_hands(
            [H("9*1"), H("4*5"), H("8*1")], current=0,
            tricks=(2, 0, 1))
        self.assertTrue(st.over is False)
        st = st.apply(classify(H("9*1")), validate=True)
        self.assertTrue(st.over)
        expected = terminal_scores(st.hands, st.tricks, st.bombs,
                                   st.winner, DEFAULT_RULES)
        self.assertEqual(list(st.scores), expected)
        self.assertEqual(sum(st.scores), 0)

    def test_zero_play_opponents_view(self):
        self.assertEqual(zero_play_opponents((2, 0, 0), 1), 1)
        self.assertEqual(zero_play_opponents((2, 0, 0), 0), 2)


class TestObjective(unittest.TestCase):
    def test_score_primary_with_win_guard(self):
        cfg = ObjectiveConfig(mode="score_primary", win_guard=0.01)
        # 1 号动作只低 0.005 胜率，但净分高 10，应该选 1。
        self.assertEqual(cfg.select_index([0.50, 0.495, 0.40],
                                          [0.0, 10.0, 100.0]), 1)
        # guard 放大后，100 分动作进入可接受集合，选 2。
        cfg = ObjectiveConfig(mode="score_primary", win_guard=0.02)
        self.assertEqual(cfg.select_index([0.50, 0.495, 0.48],
                                          [0.0, 10.0, 100.0]), 2)

    def test_win_primary_mode(self):
        cfg = ObjectiveConfig(mode="win_primary", win_guard=0.10)
        self.assertEqual(cfg.select_index([0.50, 0.49], [0.0, 100.0]), 0)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            ObjectiveConfig(mode="unknown")
        with self.assertRaises(ValueError):
            ObjectiveConfig(win_guard=-0.1)


if __name__ == "__main__":
    unittest.main()
