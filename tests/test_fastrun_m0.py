"""M0 引擎验收：规则口径 + 状态机不变量。

运行: python tests/test_fastrun_m0.py
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.moves import gen_lead, gen_beats, legal_moves
from src.fastrun.rules import (Move, MoveType, PASS, can_beat, classify)
from src.fastrun.state import GameState


H = C.parse


class TestCards(unittest.TestCase):
    def test_deck(self):
        self.assertEqual(C.DECK_SIZE, 45)
        self.assertEqual(C.total(C.FULL_DECK), 45)
        self.assertEqual(C.RANK_COUNTS, (4,) * 10 + (3, 1, 1))

    def test_roundtrip(self):
        ids = C.card_ids(C.FULL_DECK)
        self.assertEqual(len(ids), 45)
        self.assertEqual(C.hand_from_card_ids(ids), C.FULL_DECK)
        self.assertEqual(C.hand_from_card_ids([44]), H("14*1"))
        self.assertEqual(C.hand_from_card_ids([48]), H("15*1"))


class TestRuleSpec(unittest.TestCase):
    def test_r1_triple_can_lead(self):
        hand = H("5*3 7*1 9*1")
        types = {m.type for m in gen_lead(hand)}
        self.assertIn(MoveType.TRIPLE, types)
        self.assertIsNotNone(classify(H("5*3")))

    def test_r2_pair_seq_needs_three(self):
        self.assertIsNone(classify(H("3*2 4*2")))
        self.assertEqual(classify(H("3*2 4*2 5*2")).type, MoveType.DOUBLE_SEQ)
        self.assertEqual(classify(H("3*2 4*2 5*2")).length, 3)
        moves = gen_lead(H("3*2 4*2 5*2 7*1"))
        self.assertFalse([m for m in moves
                          if m.type == MoveType.DOUBLE_SEQ and m.length < 3])

    def test_r3_airplane_pair_wings(self):
        m = classify(H("3*3 4*3 5*2 6*2"))
        self.assertIsNotNone(m)
        self.assertEqual(m.type, MoveType.AIRPLANE_WITH_WINGS)
        self.assertEqual(m.length, 2)
        self.assertEqual(m.max_rank, 4)

    def test_r3_airplane_single_wings_dup(self):
        m = classify(H("3*3 4*3 5*2"))
        self.assertIsNotNone(m)
        self.assertEqual(m.type, MoveType.AIRPLANE_WITH_WINGS)
        self.assertEqual(m.length, 2)

    def test_r4_must_bomb_when_only_option(self):
        hand = H("9*4 3*1")
        target = classify(H("14*1"))
        ms = legal_moves(hand, target, next_player_cards=5)
        self.assertEqual([m.type for m in ms], [MoveType.BOMB])

    def test_r4_no_pass_when_beat_exists(self):
        hand = H("5*1 7*1 9*2")
        ms = legal_moves(hand, classify(H("4*1")), next_player_cards=5)
        self.assertNotIn(MoveType.PASS, [m.type for m in ms])
        self.assertTrue(ms)

    def test_pass_only_when_cannot_beat(self):
        ms = legal_moves(H("3*1 4*1"), classify(H("14*1")),
                         next_player_cards=5)
        self.assertEqual(ms, [PASS])

    def test_r5_top_big_single(self):
        hand = H("5*1 14*1 7*2")
        ms = legal_moves(hand, None, next_player_cards=1)
        singles = [m for m in ms if m.type == MoveType.SINGLE]
        self.assertEqual([m.max_rank for m in singles], [14])
        self.assertTrue([m for m in ms if m.type == MoveType.PAIR])

    def test_r5_top_big_when_beating(self):
        hand = H("8*1 14*1")
        ms = legal_moves(hand, classify(H("7*1")), next_player_cards=1)
        self.assertEqual([m.max_rank for m in ms], [14])


class TestStateMachine(unittest.TestCase):
    def test_cannot_pass_when_leading(self):
        st = GameState.from_hands(["3*1", "5*1", "6*1"])
        with self.assertRaises(ValueError):
            st.apply(PASS)

    def test_two_passes_end_trick(self):
        st = GameState.from_hands(["14*1 3*1", "5*1 6*1", "7*1 8*1"])
        st = st.apply(classify(H("14*1")), validate=True)
        self.assertEqual(st.current, 1)
        st = st.apply(PASS, validate=True)
        st = st.apply(PASS, validate=True)
        self.assertEqual(st.current, 0)
        self.assertIsNone(st.last_play)

    def test_basic_scoring_with_spring(self):
        st = GameState.from_hands(["3*1", "5*2", "6*4"])
        st = st.apply(classify(H("3*1")), validate=True)
        self.assertTrue(st.over)
        self.assertEqual(st.winner, 0)
        # B: 2 + 30(春天) = 32; C: 4 + 30 = 34; A = +66
        self.assertEqual(st.scores, (66, -32, -34))
        self.assertEqual(sum(st.scores), 0)

    def test_protect_one_card(self):
        # 两家都出过牌，避免春天罚分；B 剩 1 张 -> 保本 0 分
        st = GameState.from_hands(["3*1", "5*1", "6*2"], tricks=(0, 1, 1))
        st = st.apply(classify(H("3*1")), validate=True)
        self.assertEqual(st.scores, (2, 0, -2))

    def test_spring_with_one_opponent_played(self):
        st = GameState.from_hands(["3*1", "5*2", "6*4"], tricks=(0, 0, 1))
        st = st.apply(classify(H("3*1")), validate=True)
        self.assertEqual(st.scores, (36, -32, -4))

    def test_bomb_scores_and_revert(self):
        st = GameState.from_hands(["5*4 3*1", "9*4 3*1", "7*1 8*1"])
        st = st.apply(classify(H("5*4")), validate=True)
        self.assertEqual(st.bombs, (20, -10, -10))
        st = st.apply(classify(H("9*4")), validate=True)
        self.assertEqual(st.bombs, (-10, 20, -10))
        self.assertEqual(sum(st.bombs), 0)
        # 剩下的人没有更大的炸弹 -> 只能过
        self.assertEqual(st.legal_moves(), [PASS])
        st = st.apply(PASS, validate=True)
        st = st.apply(PASS, validate=True)
        self.assertIsNone(st.last_play)
        self.assertEqual(st.current, 1)

    def test_generated_moves_are_legal(self):
        hand = H("3*3 4*2 5*1 9*4 14*1")
        for m in gen_lead(hand):
            self.assertTrue(C.subset(m.cards, hand), m)
            parsed = classify(m.cards)
            self.assertIsNotNone(parsed, m)
            self.assertEqual((parsed.type, parsed.length, parsed.max_rank),
                             (m.type, m.length, m.max_rank), m)

    def test_no_duplicate_moves(self):
        hand = H("3*4 4*4 5*4 6*4 7*1 14*1")
        ms = gen_lead(hand)
        self.assertEqual(len(ms), len(set(ms)))


class TestRandomGames(unittest.TestCase):
    def test_invariants(self):
        rng = random.Random(20261020)
        for seed in range(120):
            st = GameState.new(seed)
            steps = 0
            while not st.over:
                steps += 1
                self.assertLess(steps, 400, "game did not terminate")
                ms = st.legal_moves()
                self.assertTrue(ms)
                mv = rng.choice(ms)
                if st.last_play is not None and mv.type != MoveType.PASS:
                    self.assertTrue(can_beat(mv, st.last_play))
                if st.last_play is None:
                    self.assertNotEqual(mv.type, MoveType.PASS)
                if mv.type != MoveType.PASS:
                    parsed = classify(mv.cards)
                    self.assertIsNotNone(parsed)
                    self.assertEqual((parsed.type, parsed.length, parsed.max_rank),
                                     (mv.type, mv.length, mv.max_rank))
                st = st.apply(mv)
                self.assertEqual(sum(st.remaining_cards()) + C.total(st.played),
                                 C.DECK_SIZE)
            self.assertIn(st.winner, (0, 1, 2))
            self.assertEqual(sum(st.scores), 0)
            self.assertGreaterEqual(min(st.scores), -45 - 10 - 60 * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
