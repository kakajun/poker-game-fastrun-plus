import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from src.env.poker_env import PokerEnv
from src.env.single_agent_wrapper import SingleAgentWrapper
from src.agent.model_agent import ModelAgent
from src.core.hand_type import HandType

# 2. Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ModelEvaluator:
    """
    Model Evaluator for VAD-CFR.
    """

    def __init__(self, n_episodes: int = 200):
        self.n_episodes = n_episodes
        self.results = []

    def evaluate_model(self, model_path: str, model_name: str, opponent_path: str):
        """
        Evaluate VAD-CFR vs Opponent (also VAD-CFR).
        """
        opp_name = os.path.basename(opponent_path)
        print(f"\n--- Evaluating: {model_name} VS {opp_name} ---")

        # 1. Init Env
        base_env = PokerEnv()
        # Opponents must be VAD-CFR now
        opponents = [ModelAgent(opponent_path), ModelAgent(opponent_path)]
        env = SingleAgentWrapper(base_env, opponents=opponents)

        # 2. Init Hero Agent
        try:
            agent = ModelAgent(model_path, deterministic=True)
        except Exception as e:
            print(f"Failed to init Agent {model_name}: {e}")
            return None

        # 3. Evaluation Loop
        stats = {
            "wins": 0,
            "total_reward": 0,
            "scores": [],
            "steps_to_win": [],
            "remain_cards_on_loss": [],
            "bombs_played": 0,
            "springs": 0,
            "top_single_violations": 0,
            "top_single_opportunities": 0,
            "initial_hand_strength": [],
            "total_episodes": 0
        }

        for i in range(self.n_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            steps = 0

            # Record initial hand strength
            hero_hand = env.unwrapped.game.hands[0]
            strength = sum([c.rank.value for c in hero_hand])
            stats["initial_hand_strength"].append(strength)

            while not (terminated or truncated):
                mask = info.get("action_mask")

                # Top-Card Rule Check
                next_player = (env.unwrapped.game.current_player + 1) % 3
                is_reported_single = (
                    len(env.unwrapped.game.hands[next_player]) == 1)

                # Agent Act
                # Pass 'game' object for VAD-CFR
                action = agent.act(obs, mask, env.unwrapped.game)

                # Rule Check Logic
                if is_reported_single:
                    play = env.unwrapped.action_space_manager.get_action(
                        action)
                    if play and play.type == HandType.SINGLE:
                        stats["top_single_opportunities"] += 1
                        hand = env.unwrapped.game.hands[0]
                        max_rank = hand[-1].rank.value
                        if play.max_rank < max_rank:
                            stats["top_single_violations"] += 1

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                game = env.unwrapped.game
                last_play = game.last_play
                if last_play and last_play.type == HandType.BOMB and game.last_play_player == 0:
                    stats["bombs_played"] += 1

            game = env.unwrapped.game
            stats["total_episodes"] += 1

            if terminated:
                final_score = game.scores[0]
                if game.winner == 0:
                    stats["wins"] += 1
                    stats["steps_to_win"].append(steps)
                    if any(count == 0 for count in game.cards_played_count[1:]):
                        stats["springs"] += 1
                else:
                    stats["remain_cards_on_loss"].append(len(game.hands[0]))
            else:
                final_score = -100 - len(game.hands[0])
                stats["remain_cards_on_loss"].append(len(game.hands[0]))

            stats["scores"].append(final_score)

        summary = {
            "Model": model_name,
            "Opponent": opp_name,
            "Win Rate": stats["wins"] / stats["total_episodes"],
            "Avg Score": np.mean(stats["scores"]),
            "Avg Steps (Win)": np.mean(stats["steps_to_win"]) if stats["steps_to_win"] else 0,
            "Avg Loss Cards": np.mean(stats["remain_cards_on_loss"]) if stats["remain_cards_on_loss"] else 0,
            "Spring Rate": stats["springs"] / max(1, stats["wins"]),
            "Bomb Freq": stats["bombs_played"] / stats["total_episodes"],
            "Rule Violation": stats["top_single_violations"] / max(1, stats["top_single_opportunities"]),
        }

        self.results.append(summary)
        print(
            f"Done: Win Rate {summary['Win Rate']:.2%}, Avg Score {summary['Avg Score']:.2f}")
        return summary

    def save_report(self, output_dir: str):
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "eval_report.csv"), index=False)
        print(f"\nReport saved to: {output_dir}")


if __name__ == "__main__":
    evaluator = ModelEvaluator(n_episodes=50)

    model_dir = "models"
    all_files = os.listdir(model_dir)

    # Benchmark: Find a .pth model (VAD-CFR)
    benchmark_file = None
    pths = [f for f in all_files if f.endswith(".pth")]
    if pths:
        benchmark_file = pths[0] # Use first available as benchmark
    
    if not benchmark_file:
        print("Error: No VAD-CFR models (.pth) found for benchmark.")
        # Create a dummy or exit?
        # sys.exit(1)
    else:
        benchmark_path = os.path.join(model_dir, benchmark_file)
        
        # Test models: All other .pth files
        test_models = [f for f in pths if f != benchmark_file]

        print(f"=== VAD-CFR Evaluation ===")
        print(f"Benchmark: {benchmark_file}")
        print(f"Test Models: {test_models}")

        for m in test_models:
            m_path = os.path.join(model_dir, m)
            evaluator.evaluate_model(m_path, m, benchmark_path)
            
        # Self-play latest
        if len(pths) >= 2:
            pths.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            print("\n--- Latest Models Self-Play ---")
            evaluator.evaluate_model(
                os.path.join(model_dir, pths[0]), 
                f"{pths[0]} (Latest)", 
                os.path.join(model_dir, pths[1])
            )

        evaluator.save_report("src/evaluate/reports")
