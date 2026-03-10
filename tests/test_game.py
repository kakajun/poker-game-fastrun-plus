from src.core.hand_type import HandType, Play
from src.core.card import Card, Rank
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

    # Check Random Start Player
    start_player = game.current_player
    assert 0 <= start_player <= 2

    print(f"Start Player: {start_player}")

    # Check Legal Actions
    actions = game.get_legal_actions()
    assert len(actions) > 0

    print("Game Basic Passed!")


def test_game_flow():
    print("Testing Game Flow...")
    game = Game(seed=42)
    start_player = game.current_player

    # Player plays a Single
    hand = game.hands[start_player]
    card = hand[0]
    play = Play(HandType.SINGLE, [card], length=1, max_rank=card.rank)

    is_over, events = game.step(play)
    assert not is_over
    assert game.current_player == (start_player + 1) % 3
    assert game.last_play == play
    assert card not in game.hands[start_player]

    print("Game Flow Passed!")


def test_bomb_score():
    print("Testing Bomb Score...")
    game = Game(seed=42)
    # Manually set hands for testing bomb
    # P0: 3333, 4
    # P1: 5
    # P2: 6
    game.hands[0] = [
        Card(Rank.THREE, 0),
        Card(Rank.THREE, 1),
        Card(Rank.THREE, 2),
        Card(Rank.THREE, 3),
        Card(Rank.FOUR, 0)
    ]
    game.hands[1] = [Card(Rank.FIVE, 0)]
    game.hands[2] = [Card(Rank.SIX, 0)]

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
