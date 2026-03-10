from src.core.hand_type import HandType, Play
from src.core.card import Card, Rank, Suit
from src.core.game import Game
import sys
import os
# Force add root directory
sys.path.insert(0, r"f:\git\poker-game-fastrun")


def test_game_basic():
    print("Testing Game Basic...")
    game = Game(seed=42)

    # Check hands
    assert len(game.hands) == 3
    assert len(game.hands[0]) == 15
    assert len(game.hands[1]) == 15
    assert len(game.hands[2]) == 15

    # Check Heart 3
    h3 = Card(Rank.THREE, Suit.HEART)
    start_player = game.current_player
    assert h3 in game.hands[start_player]

    print(f"Start Player: {start_player}")

    # Check Legal Actions
    actions = game.get_legal_actions()
    # Must contain H3
    for a in actions:
        assert h3 in a.cards

    print("Game Basic Passed!")


def test_game_flow():
    print("Testing Game Flow...")
    game = Game(seed=42)
    start_player = game.current_player

    # Player 0 plays H3 (Single)
    h3 = Card(Rank.THREE, Suit.HEART)
    play = Play(HandType.SINGLE, [h3], length=1, max_rank=Rank.THREE)

    is_over, events = game.step(play)
    assert not is_over
    assert game.current_player == (start_player + 1) % 3
    assert game.last_play == play
    assert h3 not in game.hands[start_player]

    # Player 1 actions
    # Should beat H3
    actions = game.get_legal_actions()
    if actions[0].type == HandType.PASS:
        print("Player 1 passed")
        game.step(actions[0])
    else:
        print(f"Player 1 plays: {actions[0]}")
        game.step(actions[0])

    print("Game Flow Passed!")


def test_bomb_score():
    print("Testing Bomb Score...")
    game = Game(seed=42)
    # Manually set hands for testing bomb
    # P0: 3333, 4
    # P1: 5
    # P2: 6
    game.hands[0] = [
        Card(Rank.THREE, Suit.DIAMOND),
        Card(Rank.THREE, Suit.CLUB),
        Card(Rank.THREE, Suit.HEART),
        Card(Rank.THREE, Suit.SPADE),
        Card(Rank.FOUR, Suit.DIAMOND)
    ]
    game.hands[1] = [Card(Rank.FIVE, Suit.DIAMOND)]
    game.hands[2] = [Card(Rank.SIX, Suit.DIAMOND)]

    game.current_player = 0
    game.last_play = None
    game.last_play_player = 0

    # P0 plays Bomb 3333
    bomb = Play(HandType.BOMB, game.hands[0][:4],
                length=1, max_rank=Rank.THREE, is_bomb=True)
    game.step(bomb)

    # Check scores: P0 +20, P1 -10, P2 -10
    assert game.bomb_scores[0] == 20
    assert game.bomb_scores[1] == -10
    assert game.bomb_scores[2] == -10

    print("Bomb Score Passed!")


if __name__ == "__main__":
    test_game_basic()
    test_game_flow()
    test_bomb_score()
