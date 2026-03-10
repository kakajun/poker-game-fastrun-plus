import sys
import os
import statistics

# Add project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.game import Game

def test_branching_factor():
    game = Game(seed=42)
    
    depths = []
    branching_factors = []
    
    stack = [(game, 0)]
    # DFS sample
    max_nodes = 100
    nodes_visited = 0
    
    import random
    
    while stack and nodes_visited < max_nodes:
        g, depth = stack.pop()
        nodes_visited += 1
        
        actions = g.get_legal_actions()
        bf = len(actions)
        branching_factors.append(bf)
        depths.append(depth)
        
        if bf > 0 and not g.is_over:
            # Sample one child
            a = random.choice(actions)
            next_g = g.clone()
            next_g.step(a)
            stack.append((next_g, depth + 1))
            
    print(f"Average BF: {statistics.mean(branching_factors):.2f}")
    print(f"Max BF: {max(branching_factors)}")
    print(f"Depths: {depths}")
    print(f"BFs: {branching_factors}")

if __name__ == "__main__":
    test_branching_factor()
