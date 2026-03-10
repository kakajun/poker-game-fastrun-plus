import sys
import os
import time

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.game import Game

def test_legal_actions_speed():
    game = Game(seed=42)
    start = time.time()
    for _ in range(10000):
        a = game.get_legal_actions()
    end = time.time()
    print(f"10,000 legal actions took {end - start:.4f}s")
    print(f"Average time: {(end - start) / 10000 * 1000:.4f}ms")

if __name__ == "__main__":
    test_legal_actions_speed()
