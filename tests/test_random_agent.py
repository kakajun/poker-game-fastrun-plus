import numpy as np
import sys
import os
# Force add root directory
sys.path.insert(0, r"f:\git\poker-game-fastrun")

from src.env.poker_env import PokerEnv

class RandomAgent:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        
    def act(self, obs, mask):
        # 随机选择一个合法的动作
        legal_actions = np.where(mask == 1)[0]
        if len(legal_actions) == 0:
            # 理论上不应发生，除非 mask 全 0
            # 这里返回 0 (Pass)
            return 0
        return np.random.choice(legal_actions)

def test_random_agent():
    print("Initializing PokerEnv...")
    env = PokerEnv()
    print(f"Action Space Size: {env.action_space.n}")
    print(f"Observation Shape: {env.observation_space.shape}")
    
    agents = [RandomAgent(env.action_space.n) for _ in range(3)]
    
    n_episodes = 10
    total_steps = 0
    wins = [0, 0, 0]
    
    print(f"Running {n_episodes} episodes with Random Agents...")
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            current_player = info["player_id"]
            mask = info["action_mask"]
            
            # Agent act
            action = agents[current_player].act(obs, mask)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Check win
            if terminated:
                winner = current_player
                wins[winner] += 1
            elif truncated:
                print(f"Episode {ep} truncated at step {steps}. Reason: {info.get('error')}")
                # Debug: Print last action and mask
                # print(f"Action: {action}, Mask sum: {np.sum(mask)}")
                
        total_steps += steps
        
    print("-" * 20)
    print(f"Average steps per episode: {total_steps / n_episodes:.2f}")
    print(f"Win distribution: {wins}")
    print("Random Agent Test Passed!")

if __name__ == "__main__":
    test_random_agent()
