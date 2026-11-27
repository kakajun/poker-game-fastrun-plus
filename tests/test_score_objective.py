"""M1/M2/v3 净分优先与胜率守卫测试。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arena.agents import GreedyMinAgent, M1Agent, M2Agent, PolicyAgent
from src.arena.match import paired_bootstrap_diff, run_duplicate_permutations
from src.eval.objective import ObjectiveConfig
from src.eval.policy_m1 import M1Policy
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.rules import PASS, classify
from src.fastrun.state import GameState
from src.search.endgame import solve_with_status

H = C.parse


class TestRulePolicyObjective(unittest.TestCase):
    def test_rank_keeps_legal_and_finishes(self):
        cfg = DEFAULT_RULES
        st = GameState.from_hands(["9*1 5*1", "6*1 7*1", "8*1 10*1"],
                                  current=0, last_play=None)
        p = M1Policy(cfg, objective=ObjectiveConfig())
        ranked = p.rank_moves(st)
        legal = st.legal_moves(cfg)
        self.assertTrue(ranked)
        self.assertTrue(all(m in legal for m in ranked))

    def test_bomb_candidate_is_considered(self):
        cfg = DEFAULT_RULES
        # 上家出了 8 炸，当前玩家只有 9 炸可压；候选必须包含炸弹。
        st = GameState.from_hands(
            ["9*4 3*1", "10*4 4*1", "8*4 5*1"], current=2,
            last_play=None)
        st = st.apply(classify(H("8*4")), validate=True)
        self.assertEqual(st.current, 0)
        p = M1Policy(cfg, objective=ObjectiveConfig())
        ranked = p.rank_moves(st)
        self.assertIn(classify(H("9*4")), ranked)

    def test_endgame_returns_win_and_score(self):
        st = GameState.from_hands(["4*1 5*1", "9*2", "10*2"],
                                  current=0, last_play=None)
        r = solve_with_status(st, objective=ObjectiveConfig())
        win, score = r.values
        self.assertGreaterEqual(win, 0.0)
        self.assertLessEqual(win, 1.0)
        self.assertIsInstance(score, float)


class TestPolicySelection(unittest.TestCase):
    def test_score_primary_beats_win_tie_break(self):
        st = GameState.from_hands(["3*1 4*1 5*1", "6*1 7*1", "8*1 9*1"],
                                  current=0, last_play=None)
        agent = PolicyAgent(objective=ObjectiveConfig(win_guard=0.01))
        ms = st.legal_moves(DEFAULT_RULES)[:3]
        # 第二个动作胜率只低 0.005，但净分高 10，应选第二个。
        agent.evaluate = lambda state, moves=None: (
            ms, __import__("numpy").asarray([0.50, 0.495, 0.40]),
            __import__("numpy").asarray([0.0, 10.0, 100.0]))
        self.assertEqual(agent.act(st), ms[1])

    def test_win_primary_mode(self):
        st = GameState.from_hands(["3*1 4*1 5*1", "6*1 7*1", "8*1 9*1"],
                                  current=0, last_play=None)
        agent = PolicyAgent(objective=ObjectiveConfig(mode="win_primary"))
        ms = st.legal_moves(DEFAULT_RULES)[:3]
        agent.evaluate = lambda state, moves=None: (
            ms, __import__("numpy").asarray([0.50, 0.49, 0.40]),
            __import__("numpy").asarray([0.0, 100.0, 200.0]))
        self.assertEqual(agent.act(st), ms[0])


class TestArenaComponents(unittest.TestCase):
    def test_component_metrics_and_paired_ci(self):
        a = run_duplicate_permutations(
            M2Agent(endgame_k=2), [GreedyMinAgent(), GreedyMinAgent()],
            n_deals=2, seed0=31)
        b = run_duplicate_permutations(
            M1Agent(), [GreedyMinAgent(), GreedyMinAgent()],
            n_deals=2, seed0=31)
        self.assertTrue(a.records)
        r = a.records[0]
        self.assertAlmostEqual(r.base + r.spring + r.bomb, r.score)
        diff = paired_bootstrap_diff(a, b, n_boot=200, seed=5)
        self.assertAlmostEqual(diff.base_diff + diff.spring_diff
                               + diff.bomb_diff, diff.score_diff, places=6)


if __name__ == "__main__":
    unittest.main()
