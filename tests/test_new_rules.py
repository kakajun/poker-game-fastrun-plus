import unittest
from src.core.game import Game
from src.core.card import Card, Rank, Suit
from src.core.evaluator import HandEvaluator
from src.core.hand_type import HandType

class TestNewRules(unittest.TestCase):
    def test_triple_with_two(self):
        """测试三带二识别"""
        cards = [
            Card(Rank.SIX, Suit.SPADE),
            Card(Rank.SIX, Suit.HEART),
            Card(Rank.SIX, Suit.CLUB),
            Card(Rank.SEVEN, Suit.DIAMOND),
            Card(Rank.EIGHT, Suit.DIAMOND)
        ]
        play = HandEvaluator.evaluate(cards)
        self.assertIsNotNone(play)
        self.assertEqual(play.type, HandType.TRIPLE_WITH_TWO)
        self.assertEqual(play.max_rank, Rank.SIX)

    def test_break_even_scoring(self):
        """测试保本规则计分"""
        game = Game()
        # 模拟游戏结束状态
        game.winner = 0
        # Player 1 剩 1 张 (保本)
        # Player 2 剩 5 张
        game.hands[0] = []
        game.hands[1] = [Card(Rank.ACE, Suit.SPADE)]
        game.hands[2] = [Card(Rank.THREE, Suit.SPADE)] * 5
        
        # 模拟出牌次数，防止触发春天
        game.cards_played_count = [10, 1, 1]
        
        game._calculate_final_scores()
        
        # Player 1 (保本): 0 分
        # Player 2: -5 分
        # Player 0 (赢家): 5 分
        self.assertEqual(game.scores[1], 0)
        self.assertEqual(game.scores[2], -5)
        self.assertEqual(game.scores[0], 5)

if __name__ == '__main__':
    unittest.main()
