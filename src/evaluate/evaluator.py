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

# 2. 设置环境变量，必须在导入 torch 之前
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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

    def evaluate_model(self, model_path: str, model_name: str, opponent_path: str):
        """
        评估模型之间的对弈。
        model_path: 待评估模型路径
        opponent_path: 对手模型路径
        """
        opp_name = os.path.basename(opponent_path)
        print(f"\n--- 正在评估: {model_name} VS {opp_name} ---")

        # 1. 初始化环境与模型对手
        base_env = PokerEnv()
        opponents = [ModelAgent(opponent_path), ModelAgent(opponent_path)]
        env = SingleAgentWrapper(base_env, opponents=opponents)

        # 2. 初始化待评估 Agent
        try:
            agent = ModelAgent(model_path, deterministic=True)
        except Exception as e:
            print(f"无法初始化 Agent {model_name}: {e}")
            return None

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
                # 获取 mask
                mask = info.get("action_mask")

                # 检查顶大规则机会 (下家报单且轮到我出牌)
                next_player = (env.unwrapped.game.current_player + 1) % 3
                is_reported_single = (
                    len(env.unwrapped.game.hands[next_player]) == 1)

                # 使用 Agent 行动
                action = agent.act(obs, mask, env.unwrapped.game)

                # 评估策略质量: 顶大规则执行情况
                if is_reported_single:
                    play = env.unwrapped.action_space_manager.get_action(
                        action)
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
                # Truncated
                final_score = -100 - len(game.hands[0])
                stats["remain_cards_on_loss"].append(len(game.hands[0]))

            stats["scores"].append(final_score)

        # 4. 计算最终指标
        summary = {
            "模型": model_name,
            "对手": opp_name,
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
        print(
            f"评估完成: 胜率 {summary['胜率']:.2%}, 平均分 {summary['平均得分']:.2f}, 顶大违规率 {summary['顶大违规率']:.2%}")
        return summary

    def _calc_correlation(self, x, y):
        if len(x) < 2:
            return 0
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0

    def save_report(self, output_dir: str):
        """保存评估报告并绘图"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存 CSV
        df.to_csv(os.path.join(output_dir, "eval_report.csv"),
                  index=False, encoding='utf-8-sig')

        # 2. 绘制多维度对比图
        plt.figure(figsize=(15, 10))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 子图 1: 胜率与春天率
        plt.subplot(2, 2, 1)
        for opp in df["对手"].unique():
            sub = df[df["对手"] == opp]
            x = np.arange(len(sub))
            plt.bar(x - 0.2, sub["胜率"] * 100,
                    width=0.4, label=f"胜率 (vs {opp})")
            plt.bar(x + 0.2, sub["春天率"] * 100,
                    width=0.4, label=f"春天率 (vs {opp})")
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
            plt.bar(x - 0.2, sub["顶大违规率"] * 100,
                    width=0.4, label=f"顶大违规率 (vs {opp})")
            plt.bar(x + 0.2, sub["炸弹频率"] * 10, width=0.4,
                    label=f"炸弹频率x10 (vs {opp})")
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
    # 减少对局数以加快对比速度
    evaluator = ModelEvaluator(n_episodes=50)

    model_dir = "models"
    # 支持 PPO (.zip) 和 MCCFR (.pth, .pkl)
    all_files = os.listdir(model_dir)

    # 1. 定义基准模型 (通常是之前最强的 PPO)
    benchmark_file = "ppo_poker_final.zip"
    benchmark_path = os.path.join(model_dir, benchmark_file)

    if not os.path.exists(benchmark_path):
        # 如果找不到 Final PPO，选一个最新的 .zip
        zips = [f for f in all_files if f.endswith(".zip")]
        if zips:
            benchmark_file = zips[0]
            benchmark_path = os.path.join(model_dir, benchmark_file)
        else:
            print("错误: 找不到任何基准模型 (.zip)")
            sys.exit(1)

    # 2. 筛选待测评模型 (所有 .pth 和 .pkl)
    test_models = [f for f in all_files if (f.endswith(
        ".pth") or f.endswith(".pkl")) and f != benchmark_file]

    print(f"=== 开始模型对弈测评 ===")
    print(f"基准对手: {benchmark_file}")
    print(f"待测模型: {test_models}")

    for m in test_models:
        m_path = os.path.join(model_dir, m)
        evaluator.evaluate_model(m_path, m, benchmark_path)

    # 3. 如果有多个 .pth 模型，也可以进行它们之间的内战
    pths = [f for f in test_models if f.endswith(".pth")]
    if len(pths) >= 2:
        print("\n--- 进行 MCCFR 模型内战 ---")
        # 让最新的对战次新的
        pths.sort(key=lambda x: os.path.getmtime(
            os.path.join(model_dir, x)), reverse=True)
        evaluator.evaluate_model(os.path.join(
            model_dir, pths[0]), f"{pths[0]} (最新)", os.path.join(model_dir, pths[1]))

    evaluator.save_report("src/evaluate/reports")
