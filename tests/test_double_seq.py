import unittest
from src.core.card import Card, Rank, Suit
from src.core.evaluator import HandEvaluator
from src.core.hand_type import HandType

class TestDoubleSeq(unittest.TestCase):
    def test_2_pair_double_seq(self):
        # 7,7,8,8
        cards = [
            Card(Rank.SEVEN, Suit.HEART),
            Card(Rank.SEVEN, Suit.SPADE),
            Card(Rank.EIGHT, Suit.DIAMOND),
            Card(Rank.EIGHT, Suit.HEART)
        ]
        play = HandEvaluator.evaluate(cards)
        self.assertIsNotNone(play)
        self.assertEqual(play.type, HandType.DOUBLE_SEQUENCE)
        self.assertEqual(play.length, 2)
        self.assertEqual(play.max_rank, Rank.EIGHT)

    def test_3_pair_double_seq(self):
        # 7,7,8,8,9,9
        cards = [
            Card(Rank.SEVEN, Suit.HEART),
            Card(Rank.SEVEN, Suit.SPADE),
            Card(Rank.EIGHT, Suit.DIAMOND),
            Card(Rank.EIGHT, Suit.HEART),
            Card(Rank.NINE, Suit.CLUB),
            Card(Rank.NINE, Suit.DIAMOND)
        ]
        play = HandEvaluator.evaluate(cards)
        self.assertIsNotNone(play)
        self.assertEqual(play.type, HandType.DOUBLE_SEQUENCE)
        self.assertEqual(play.length, 3)
        self.assertEqual(play.max_rank, Rank.NINE)

if __name__ == '__main__':
    unittest.main()
