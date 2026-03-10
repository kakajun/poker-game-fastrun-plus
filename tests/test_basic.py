import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.card import Card, Rank, Suit
from src.core.deck import Deck
from src.core.hand_type import HandType
from src.core.evaluator import HandEvaluator

def test_deck():
    print("Testing Deck...")
    deck = Deck()
    print(f"Deck size: {len(deck.cards)}")
    assert len(deck.cards) == 45
    
    # Check specific cards
    # Should have Heart 2
    h2 = Card(Rank.TWO, Suit.HEART)
    assert h2 in deck.cards
    
    # Should NOT have Diamond 2
    d2 = Card(Rank.TWO, Suit.DIAMOND)
    assert d2 not in deck.cards
    
    # Should have Spade A
    sa = Card(Rank.ACE, Suit.SPADE)
    assert sa in deck.cards
    
    # Should NOT have Diamond K
    dk = Card(Rank.KING, Suit.DIAMOND)
    assert dk not in deck.cards
    
    print("Deck test passed!")

def test_evaluator():
    print("Testing Evaluator...")
    
    # Single
    c = [Card(Rank.THREE, Suit.DIAMOND)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.SINGLE
    
    # Pair
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.PAIR
    
    # Triple
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB), Card(Rank.THREE, Suit.HEART)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.TRIPLE
    
    # Bomb
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB), Card(Rank.THREE, Suit.HEART), Card(Rank.THREE, Suit.SPADE)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.BOMB
    
    # Triple with Single
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB), Card(Rank.THREE, Suit.HEART), Card(Rank.FOUR, Suit.SPADE)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.TRIPLE_WITH_SINGLE
    
    # Straight
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.FOUR, Suit.CLUB), Card(Rank.FIVE, Suit.HEART), Card(Rank.SIX, Suit.SPADE), Card(Rank.SEVEN, Suit.DIAMOND)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.STRAIGHT
    
    # Straight with 2 (Invalid)
    c = [Card(Rank.JACK, Suit.DIAMOND), Card(Rank.QUEEN, Suit.CLUB), Card(Rank.KING, Suit.HEART), Card(Rank.ACE, Suit.SPADE), Card(Rank.TWO, Suit.HEART)]
    p = HandEvaluator.evaluate(c)
    assert p is None
    
    # Airplane
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB), Card(Rank.THREE, Suit.HEART),
         Card(Rank.FOUR, Suit.DIAMOND), Card(Rank.FOUR, Suit.CLUB), Card(Rank.FOUR, Suit.HEART)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.AIRPLANE
    
    # Airplane with Wings (Singles)
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB), Card(Rank.THREE, Suit.HEART),
         Card(Rank.FOUR, Suit.DIAMOND), Card(Rank.FOUR, Suit.CLUB), Card(Rank.FOUR, Suit.HEART),
         Card(Rank.FIVE, Suit.DIAMOND), Card(Rank.SIX, Suit.CLUB)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.AIRPLANE_WITH_WINGS
    
    # Airplane with Wings (Pairs)
    c = [Card(Rank.THREE, Suit.DIAMOND), Card(Rank.THREE, Suit.CLUB), Card(Rank.THREE, Suit.HEART),
         Card(Rank.FOUR, Suit.DIAMOND), Card(Rank.FOUR, Suit.CLUB), Card(Rank.FOUR, Suit.HEART),
         Card(Rank.FIVE, Suit.DIAMOND), Card(Rank.FIVE, Suit.CLUB),
         Card(Rank.SIX, Suit.DIAMOND), Card(Rank.SIX, Suit.CLUB)]
    p = HandEvaluator.evaluate(c)
    assert p.type == HandType.AIRPLANE_WITH_WINGS
    
    print("Evaluator test passed!")

if __name__ == "__main__":
    test_deck()
    test_evaluator()
