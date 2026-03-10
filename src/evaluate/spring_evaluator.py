import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from src.env.poker_env import PokerEnv
from src.env.single_agent_wrapper import SingleAgentWrapper
from src.agent.random_agent import RandomAgent
from src.agent.heuristic_agent import HeuristicAgent
from src.core.hand_type import HandType
from src.core.card import Card, Rank, Suit

class SpringChallengeEvaluator:
    """
    专门针对“打春天”能力的模型评估工具。
    1. 移除无意义的顶大违规率。
    2. 增加预设的“必打春天”手牌测试。
    3. 统计模型在好牌情况下的春天达成率。
    """
    
    def __init__(self, n_episodes: int = 100):
        self.n_episodes = n_episodes
        self.results = []
        
    def _create_spring_hand(self) -> List[Card]:
        """
        构造一组极其强力、理论上必打春天的手牌。
        例如：红桃3开始的长顺子 + 大对子 + 炸弹。
        """
        # 必须包含红桃3 (用于首出)
        hand = [Card(Rank.THREE, Suit.HEART)]
        # 顺子 4-5-6-7-8
        for r in range(4, 9):
            hand.append(Card(Rank(r), Suit.SPADE))
        # 连对 99-1010-JJ
        for r in range(9, 12):
            hand.append(Card(Rank(r), Suit.SPADE))
            hand.append(Card(Rank(r), Suit.HEART))
        # 炸弹 QQQQ
        for s in Suit:
            hand.append(Card(Rank.QUEEN, s))
        # 大单张 2
        hand.append(Card(Rank.TWO, Suit.HEART))
        
        # 确保正好15张
        while len(hand) < 15:
            # 补点大牌
            hand.append(Card(Rank.ACE, Suit.SPADE))
            
        return sorted(hand, key=lambda c: (c.rank.value, -c.suit.value), reverse=True)

    def evaluate_model(self, model_path: str, model_name: str, opponent_type: str = "heuristic"):
        """
        评估模型，包含常规测试和春天挑战测试。
        """
        print(f"\n--- 正在启动【春天挑战】评估: {model_name} 对战 {opponent_type} ---")
        
        base_env = PokerEnv()
        if opponent_type == "random":
            opponents = [RandomAgent(base_env.action_space.n), RandomAgent(base_env.action_space.n)]
        else:
            opponents = [HeuristicAgent(), HeuristicAgent()]
            
        env = SingleAgentWrapper(base_env, opponents=opponents)
        
        try:
            model = MaskablePPO.load(model_path, env=env)
        except Exception as e:
            print(f"无法加载模型 {model_path}: {e}")
            return

        stats = {
            "wins": 0,
            "springs": 0,
            "challenge_attempts": 0,
            "challenge_springs": 0,
            "total_reward": 0,
            "avg_steps_to_win": [],
            "total_episodes": 0
        }

        for i in range(self.n_episodes):
            # 每 5 局插入一次“春天挑战” (固定好牌)
            is_challenge = (i % 5 == 0)
            
            obs, info = env.reset()
            
            if is_challenge:
                # 注入必胜手牌给 Player 0 (Hero)
                spring_hand = self._create_spring_hand()
                env.unwrapped.game.hands[0] = spring_hand
                # 重新编码 obs，因为手牌变了
                obs = env.unwrapped._get_obs()
                stats["challenge_attempts"] += 1

            terminated = False
            truncated = False
            steps = 0
            
            while not (terminated or truncated):
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                
                if isinstance(action, np.ndarray):
                    action = action.item()
                
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
            
            game = env.unwrapped.game
            stats["total_episodes"] += 1
            
            if terminated and game.winner == 0:
                stats["wins"] += 1
                stats["avg_steps_to_win"].append(steps)
                # 检查是否达成春天 (对手一张牌没出)
                is_spring = any(count == 0 for count in game.cards_played_count[1:])
                if is_spring:
                    stats["springs"] += 1
                    if is_challenge:
                        stats["challenge_springs"] += 1

        summary = {
            "模型": model_name,
            "对手": opponent_type,
            "总胜率": stats["wins"] / stats["total_episodes"],
            "总体春天率": stats["springs"] / max(1, stats["wins"]),
            "春天挑战达成率": stats["challenge_springs"] / max(1, stats["challenge_attempts"]),
            "获胜平均步数": np.mean(stats["avg_steps_to_win"]) if stats["avg_steps_to_win"] else 0,
            "对局数": stats["total_episodes"]
        }
        
        self.results.append(summary)
        print(f"评估结果: 胜率 {summary['总胜率']:.2%}, 春天挑战成功率: {summary['春天挑战达成率']:.2%}")
        return summary

    def save_report(self, output_dir: str):
        if not self.results: return
        df = pd.DataFrame(self.results)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 CSV
        df.to_csv(os.path.join(output_dir, "spring_eval_report.csv"), index=False, encoding='utf-8-sig')
        
        # 绘图
        plt.figure(figsize=(12, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        x = np.arange(len(df))
        plt.bar(x - 0.2, df["总体春天率"] * 100, width=0.4, label="自然对局春天率")
        plt.bar(x + 0.2, df["春天挑战达成率"] * 100, width=0.4, label="好牌挑战春天率")
        
        plt.xticks(x, df["模型"])
        plt.ylabel("达成率 (%)")
        plt.title("模型【打春天】能力专项评估")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spring_challenge_results.png"))
        print(f"\n专项报告已生成至: {output_dir}")

if __name__ == "__main__":
    evaluator = SpringChallengeEvaluator(n_episodes=50)
    model_dir = "models"
    models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    
    for m in models:
        evaluator.evaluate_model(os.path.join(model_dir, m), m, opponent_type="heuristic")
        
    evaluator.save_report("src/evaluate/reports")
