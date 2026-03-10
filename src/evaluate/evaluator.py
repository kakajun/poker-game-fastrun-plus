import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from src.env.poker_env import PokerEnv
from src.env.single_agent_wrapper import SingleAgentWrapper
from src.agent.random_agent import RandomAgent
from src.agent.heuristic_agent import HeuristicAgent
from src.core.hand_type import HandType

class ModelEvaluator:
    """
    模型多维度评估工具。
    评估指标包括：
    1. 竞技维度 (Win Rate, Score, Distribution)
    2. 策略质量 (Top Single, Bomb Usage, Spring)
    3. 效率维度 (Avg Steps, Card Clearance)
    4. 鲁棒性 (Weak Hand Performance)
    """

    def __init__(self, n_episodes: int = 200):
        self.n_episodes = n_episodes
        self.results = []

    def evaluate_model(self, model_path: str, model_name: str, opponent_type: str = "random"):
        """
        评估单个模型。
        opponent_type: "random" 或 "heuristic"
        """
        print(f"\n--- 正在评估模型: {model_name} 对战 {opponent_type} ---")

        # 1. 初始化环境与对手
        base_env = PokerEnv()
        if opponent_type == "random":
            opponents = [RandomAgent(base_env.action_space.n), RandomAgent(base_env.action_space.n)]
        else:
            opponents = [HeuristicAgent(), HeuristicAgent()]

        env = SingleAgentWrapper(base_env, opponents=opponents)

        # 2. 加载模型
        try:
            # 使用 custom_objects 忽略版本不匹配导致的警告（可选）
            model = MaskablePPO.load(model_path, env=env)
        except Exception as e:
            print(f"无法加载模型 {model_path}: {e}")
            return

        # 3. 运行评估循环
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

            # 记录初始手牌强度 (Rank 值之和)
            hero_hand = env.unwrapped.game.hands[0]
            strength = sum([c.rank.value for c in hero_hand])
            stats["initial_hand_strength"].append(strength)

            while not (terminated or truncated):
                # 获取 mask 并预测
                mask = env.action_masks()

                # 检查顶大规则机会 (下家报单且轮到我出牌)
                next_player = (env.unwrapped.game.current_player + 1) % 3
                is_reported_single = (len(env.unwrapped.game.hands[next_player]) == 1)

                action, _ = model.predict(obs, action_masks=mask, deterministic=True)

                if isinstance(action, np.ndarray):
                    action = action.item()

                # 评估策略质量: 顶大规则执行情况
                if is_reported_single:
                    play = env.unwrapped.action_space_manager.get_action(action)
                    if play and play.type == HandType.SINGLE:
                        stats["top_single_opportunities"] += 1
                        # 检查是否是手中最大的单张
                        hand = env.unwrapped.game.hands[0]
                        max_rank = hand[-1].rank.value
                        if play.max_rank < max_rank:
                            stats["top_single_violations"] += 1

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                # 记录炸弹使用
                game = env.unwrapped.game
                last_play = game.last_play
                if last_play and last_play.type == HandType.BOMB and game.last_play_player == 0:
                    stats["bombs_played"] += 1

            # 游戏结束后的统计
            game = env.unwrapped.game
            stats["total_episodes"] += 1

            if terminated:
                final_score = game.scores[0]
                if game.winner == 0:
                    stats["wins"] += 1
                    stats["steps_to_win"].append(steps)
                    # 检查是否春天
                    if any(count == 0 for count in game.cards_played_count[1:]):
                        stats["springs"] += 1
                else:
                    stats["remain_cards_on_loss"].append(len(game.hands[0]))
            else:
                # Truncated (通常是因为非法动作)
                final_score = -100 - len(game.hands[0])
                stats["remain_cards_on_loss"].append(len(game.hands[0]))

            stats["scores"].append(final_score)

        # 4. 计算最终指标
        summary = {
            "模型": model_name,
            "对手": opponent_type,
            "胜率": stats["wins"] / stats["total_episodes"],
            "平均得分": np.mean(stats["scores"]),
            "获胜平均步数": np.mean(stats["steps_to_win"]) if stats["steps_to_win"] else 0,
            "战败剩余张数": np.mean(stats["remain_cards_on_loss"]) if stats["remain_cards_on_loss"] else 0,
            "春天率": stats["springs"] / max(1, stats["wins"]),
            "炸弹频率": stats["bombs_played"] / stats["total_episodes"],
            "顶大违规率": stats["top_single_violations"] / max(1, stats["top_single_opportunities"]),
            "手牌强度胜率相关性": self._calc_correlation(stats["initial_hand_strength"], [1 if s > 0 else 0 for s in stats["scores"]])
        }

        self.results.append(summary)
        print(f"评估完成: 胜率 {summary['胜率']:.2%}, 平均分 {summary['平均得分']:.2f}, 顶大违规率 {summary['顶大违规率']:.2%}")
        return summary

    def _calc_correlation(self, x, y):
        if len(x) < 2: return 0
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0

    def save_report(self, output_dir: str):
        """保存评估报告并绘图"""
        if not self.results: return

        df = pd.DataFrame(self.results)
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存 CSV
        df.to_csv(os.path.join(output_dir, "eval_report.csv"), index=False, encoding='utf-8-sig')

        # 2. 绘制多维度对比图
        plt.figure(figsize=(15, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 子图 1: 胜率与春天率
        plt.subplot(2, 2, 1)
        for opp in df["对手"].unique():
            sub = df[df["对手"] == opp]
            x = np.arange(len(sub))
            plt.bar(x - 0.2, sub["胜率"] * 100, width=0.4, label=f"胜率 (vs {opp})")
            plt.bar(x + 0.2, sub["春天率"] * 100, width=0.4, label=f"春天率 (vs {opp})")
        plt.xticks(np.arange(len(df["模型"].unique())), df["模型"].unique())
        plt.ylabel("百分比 (%)")
        plt.title("竞技表现 (胜率 & 春天率)")
        plt.legend()

        # 子图 2: 平均得分与战败剩余
        plt.subplot(2, 2, 2)
        for opp in df["对手"].unique():
            sub = df[df["对手"] == opp]
            plt.bar(sub["模型"] + f"\n({opp})", sub["平均得分"])
        plt.ylabel("分数")
        plt.title("得分能力对比")

        # 子图 3: 策略质量 (顶大违规 & 炸弹频率)
        plt.subplot(2, 2, 3)
        for opp in df["对手"].unique():
            sub = df[df["对手"] == opp]
            x = np.arange(len(sub))
            plt.bar(x - 0.2, sub["顶大违规率"] * 100, width=0.4, label=f"顶大违规率 (vs {opp})")
            plt.bar(x + 0.2, sub["炸弹频率"] * 10, width=0.4, label=f"炸弹频率x10 (vs {opp})")
        plt.xticks(np.arange(len(df["模型"].unique())), df["模型"].unique())
        plt.ylabel("百分比 / 频率")
        plt.title("策略质量指标")
        plt.legend()

        # 子图 4: 效率 (平均步数)
        plt.subplot(2, 2, 4)
        for opp in df["对手"].unique():
            sub = df[df["对手"] == opp]
            plt.bar(sub["模型"] + f"\n({opp})", sub["获胜平均步数"])
        plt.ylabel("步数")
        plt.title("获胜效率 (步数越少越好)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "full_dimensions_comparison.png"))

        print(f"\n完整评估报告已生成至: {output_dir}")

if __name__ == "__main__":
    evaluator = ModelEvaluator(n_episodes=50)

    # 查找 models 文件夹中的模型
    model_dir = "models"
    models = [f for f in os.listdir(model_dir) if f.endswith(".zip")]

    for m in models:
        m_path = os.path.join(model_dir, m)
        # 对战随机
        evaluator.evaluate_model(m_path, m, opponent_type="random")
        # 对战启发式
        evaluator.evaluate_model(m_path, m, opponent_type="heuristic")

    evaluator.save_report("src/evaluate/reports")
