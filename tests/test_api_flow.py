"""M5 集成回归：API 适配层 + 会话 + 人类/AI 交替出牌。

运行: python tests/test_api_flow.py
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.ai_service import AIService
from src.api.fast_service import (SessionManager, card_ids_for_move,
                                  packed_from_card_ids)
from src.api.models import ActionRequest, GameStartRequest
from src.fastrun import cards as C
from src.fastrun.config import DEFAULT_RULES
from src.fastrun.rules import MoveType
from src.fastrun.state import GameState

CFG = DEFAULT_RULES


class TestApiFlow(unittest.TestCase):
    def test_full_games(self):
        rng = random.Random(2026)
        ai = AIService()
        for seed in range(120):
            sm = SessionManager()
            sid = sm.create_session()
            s = sm.get_session(sid)
            s.state = GameState.new(seed)
            steps = 0

            while not s.state.over:
                steps += 1
                self.assertLess(steps, 400)
                if s.state.current == s.human_player_idx:
                    legal = s.legal_moves()
                    mv = rng.choice(legal)
                    ids = card_ids_for_move(s.state.hand(), mv)
                    got = s.apply_human_cards(ids)
                    self.assertEqual(got.cards, mv.cards)
                else:
                    mv, thinking = ai.predict(s.state, return_thinking=True)
                    self.assertIsNotNone(thinking)
                    self.assertEqual(thinking.player_idx, s.state.current)
                    s.apply_ai_move(mv)

                dto = s.to_dto()
                for p in range(3):
                    self.assertEqual(len(dto.hands[p]), C.total(s.state.hands[p]))
                    for c in dto.hands[p]:
                        self.assertEqual(c.rank, c.id // 4 + 3)
                if not s.state.over:
                    if s.state.current == s.human_player_idx:
                        self.assertGreater(len(dto.legal_actions), 0)
                    else:
                        self.assertEqual(dto.legal_actions, [])

            self.assertIn(s.state.winner, (0, 1, 2))
            self.assertEqual(sum(s.state.scores), 0)
            dto = s.to_dto()
            self.assertEqual(dto.winner, s.state.winner)
            self.assertTrue(dto.is_over)

    def test_human_illegal_move(self):
        sm = SessionManager()
        s = sm.get_session(sm.create_session())
        s.state = GameState.from_hands(["3*1 4*1", "5*1", "6*1"], current=0)
        with self.assertRaises(ValueError):
            s.apply_human_cards([])                      # 首出不能过
        with self.assertRaises(ValueError):
            s.apply_human_cards([0, 4])                  # 3 + 4 不是牌型

    def test_top_big_exposed_through_api(self):
        sm = SessionManager()
        s = sm.get_session(sm.create_session())
        # 下家（P1）只剩 1 张：P0 的合法单张只能是最大单张
        s.state = GameState.from_hands(["5*1 14*1 7*2", "9*1", "8*2"], current=0)
        dto = s.to_dto()
        singles = [p for p in dto.legal_actions if p.type == "SINGLE"]
        self.assertEqual([p.max_rank for p in singles], [14])
        self.assertTrue([p for p in dto.legal_actions if p.type == "PAIR"])

    def test_must_beat_via_api(self):
        sm = SessionManager()
        s = sm.get_session(sm.create_session())
        s.state = GameState.from_hands(["14*1 3*1", "5*2", "6*2"], current=0)
        s.apply_human_cards([44])                        # 出 A
        # P1 管不起 A -> 只能过
        dto = s.to_dto()
        self.assertEqual(dto.legal_actions, [])          # 不是人类回合，不返回
        s.state = s.state.apply(s.state.legal_moves()[0])
        self.assertEqual(s.state.current, 2)

    def test_dto_roundtrip_card_ids(self):
        sm = SessionManager()
        s = sm.get_session(sm.create_session())
        hand = s.state.hands[0]
        dto = s.to_dto()
        ids = [c.id for c in dto.hands[0]]
        self.assertEqual(packed_from_card_ids(ids), hand)
        self.assertEqual(len(ids), 15)

    def test_app_imports(self):
        from src.api import app as api_app
        self.assertTrue(hasattr(api_app, "player_action"))
        self.assertTrue(api_app.ai_service.list_models())
        dto = api_app.start_game(GameStartRequest(model_name="m2-search"))
        self.assertEqual(dto.current_player, 0)
        self.assertTrue(dto.legal_actions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
