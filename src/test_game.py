import sys
import os

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.game import Game
from src.core.action_generator import ActionGenerator

def test_game_step():
    print("Initializing game...")
    game = Game(seed=42)
    print(f"Game initialized. Players: {len(game.hands)}")
    
    print("Getting legal actions...")
    actions = game.get_legal_actions()
    print(f"Legal actions: {len(actions)}")
    
    if actions:
        print(f"Executing action: {actions[0]}")
        game.step(actions[0])
        print("Step executed.")
    
    print("Cloning game...")
    cloned = game.clone()
    print("Game cloned.")
    
    print("Shuffling other hands...")
    cloned.shuffle_other_hands(game.current_player)
    print("Hands shuffled.")

if __name__ == "__main__":
    test_game_step()
