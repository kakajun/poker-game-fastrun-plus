from src.core.evaluator import HandEvaluator
from src.core.action_generator import ActionGenerator
from src.core.hand_type import HandType
from src.core.card import Card, Rank, Suit
import sys
import os
# Force add root directory
sys.path.insert(0, r"f:\git\poker-game-fastrun")


def test_action_gen_basic():
    print("Testing ActionGenerator Basic...")
    # Hand: 3, 3, 4, 5, 6, 7
    c = [
        Card(Rank.THREE, Suit.DIAMOND),
        Card(Rank.THREE, Suit.CLUB),
        Card(Rank.FOUR, Suit.HEART),
        Card(Rank.FIVE, Suit.SPADE),
        Card(Rank.SIX, Suit.DIAMOND),
        Card(Rank.SEVEN, Suit.CLUB)
    ]

    actions = ActionGenerator.get_all_actions(c)
    print(f"Total actions: {len(actions)}")

    # Check Singles: 3(2), 4, 5, 6, 7. Total 2+1+1+1+1 = 6 singles.
    singles = [a for a in actions if a.type == HandType.SINGLE]
    assert len(singles) == 6

    # Check Pairs: 33. Total 1 pair.
    pairs = [a for a in actions if a.type == HandType.PAIR]
    assert len(pairs) == 1
    assert pairs[0].max_rank == Rank.THREE

    # Check Straight: 3,4,5,6,7.
    # Note: 3D, 4H, 5S, 6D, 7C. One combination.
    # But 3C, 4H, 5S, 6D, 7C. Another combination.
    # Total 2 straights.
    straights = [a for a in actions if a.type == HandType.STRAIGHT]
    assert len(straights) == 2
    assert straights[0].length == 5

    print("Basic test passed!")


def test_action_gen_bomb():
    print("Testing ActionGenerator Bomb...")
    # Hand: 3333
    c = [
        Card(Rank.THREE, Suit.DIAMOND),
        Card(Rank.THREE, Suit.CLUB),
        Card(Rank.THREE, Suit.HEART),
        Card(Rank.THREE, Suit.SPADE)
    ]

    actions = ActionGenerator.get_all_actions(c)

    # Bomb: 1
    bombs = [a for a in actions if a.type == HandType.BOMB]
    assert len(bombs) == 1

    # Triple + Single: 333+3.
    # 333(DCH)+S, 333(DCS)+H... 4 triples * 1 single?
    # No, remaining single is the 4th 3.
    # 4 combinations of triples.
    # Each triple has 1 remaining card.
    # So 4 TripleWithSingle actions.
    tws = [a for a in actions if a.type == HandType.TRIPLE_WITH_SINGLE]
    assert len(tws) == 4

    print("Bomb test passed!")


def test_legal_actions():
    print("Testing Legal Actions...")
    c = [
        Card(Rank.THREE, Suit.DIAMOND),
        Card(Rank.FOUR, Suit.CLUB),
        Card(Rank.FIVE, Suit.HEART),
        Card(Rank.SIX, Suit.SPADE),
        Card(Rank.SEVEN, Suit.DIAMOND),  # Straight 3-7
        Card(Rank.ACE, Suit.SPADE)      # Single A
    ]

    # Target: Single 10
    target = ActionGenerator.get_all_actions(
        [Card(Rank.TEN, Suit.DIAMOND)])[0]  # Single 10

    legal = ActionGenerator.get_legal_actions(c, target)
    # Should contain Single A.
    # Should NOT contain Single 3,4,5,6,7.
    # Should NOT contain Straight.

    assert len(legal) == 1
    assert legal[0].type == HandType.SINGLE
    assert legal[0].max_rank == Rank.ACE

    print("Legal actions test passed!")


if __name__ == "__main__":
    test_action_gen_basic()
    test_action_gen_bomb()
    test_legal_actions()
