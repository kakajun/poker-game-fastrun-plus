import sys
import os
import time

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.game import Game
from src.core.action_generator import ActionGenerator

def test_clone_speed():
    game = Game(seed=42)
    start = time.time()
    for _ in range(10000):
        c = game.clone()
    end = time.time()
    print(f"10,000 clones took {end - start:.4f}s")
    print(f"Average clone time: {(end - start) / 10000 * 1000:.4f}ms")

if __name__ == "__main__":
    test_clone_speed()
