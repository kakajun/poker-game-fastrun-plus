import gymnasium as gym
import numpy as np
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

# 修复 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 确保项目根目录在 path 中，以便导入 src 下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from src.env.poker_env import PokerEnv
from src.env.single_agent_wrapper import SingleAgentWrapper

def mask_fn(env: gym.Env) -> np.ndarray:
    """获取环境的动作掩码"""
    return env.action_masks()

def plot_training_curves(log_dir: str, out_dir: str):
    """绘制训练曲线（奖励和回合长度）并保存为图片"""
    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"日志文件未找到: {monitor_file}")
        return

    df = pd.read_csv(monitor_file, comment='#')
    if df.empty:
        print("日志文件为空，无法绘制曲线。")
        return

    ep = np.arange(1, len(df) + 1)
    rewards = df["r"].to_numpy()
    lengths = df["l"].to_numpy()

    # 计算滑动平均
    window = max(1, min(50, len(df)//10 if len(df) > 100 else 10))
    rewards_ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean().to_numpy()
    lengths_ma = pd.Series(lengths).rolling(window=window, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(12, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # 绘制奖励曲线
    plt.subplot(2, 1, 1)
    plt.plot(ep, rewards, alpha=0.4, label="奖励")
    plt.plot(ep, rewards_ma, color="C1", label=f"滑动平均 (窗口={window})")
    plt.xlabel("回合 (Episode)")
    plt.ylabel("奖励 (Reward)")
    plt.title("训练过程奖励曲线")
    plt.legend()

    # 绘制回合长度曲线
    plt.subplot(2, 1, 2)
    plt.plot(ep, lengths, alpha=0.4, label="回合长度")
    plt.plot(ep, lengths_ma, color="C1", label=f"滑动平均 (窗口={window})")
    plt.xlabel("回合 (Episode)")
    plt.ylabel("长度 (Length)")
    plt.title("训练过程回合长度曲线")
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()
    print(f"训练曲线图已保存至: {os.path.join(out_dir, 'training_curves.png')}")

def save_eval_image(win_rate: float, avg_reward: float, out_dir: str, n_episodes: int):
    """将评估结果生成柱状图并保存"""
    plt.figure(figsize=(8, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    bars = plt.bar(["胜率"], [win_rate * 100.0], color="C0")
    plt.ylim(0, 100)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.1f}%", ha="center")

    plt.title("模型评估结果")
    txt = f"评估回合数: {n_episodes}\n平均奖励: {avg_reward:.2f}"
    plt.gcf().text(0.7, 0.5, txt, fontsize=11, bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_summary.png"))
    plt.close()
    print(f"评估结果图已保存至: {os.path.join(out_dir, 'evaluation_summary.png')}")

from stable_baselines3.common.callbacks import BaseCallback

class SelfPlaySaveCallback(BaseCallback):
    """
    定期保存模型快照，以便在 Self-Play 中使用。
    """
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self.save_path, f"ppo_poker_snapshot_{timestamp}")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saved self-play snapshot to {path}")
        return True

def train():
    """主训练函数"""
    # 1. 创建目录
    run_dir = os.path.join("logs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 2. 创建环境
    env = PokerEnv()
    # 启用混合对手模式，包括 30% 概率的 Self-Play
    env = SingleAgentWrapper(env, mixed_opponents=True, self_play_prob=0.3)

    # 3. 添加动作掩码和监控器 Wrapper
    env = ActionMasker(env, mask_fn)
    env = Monitor(env, run_dir)

    # 4. 创建 PPO 模型
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=run_dir,
        learning_rate=2e-4,
        n_steps=4096,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=[256, 256, 256])
    )

    # 5. 设置 Callback (每 50,000 步保存一次快照用于 Self-Play)
    snapshot_callback = SelfPlaySaveCallback(check_freq=50000, save_path="models")

    # 6. 开始训练
    total_timesteps = 500000
    print(f"开始 Self-Play 训练，总步数: {total_timesteps}...")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=snapshot_callback
    )

    # 6. 保存最终模型
    model_name = f"ppo_poker_{time.strftime('%Y%m%d-%H%M%S')}"
    model_path = os.path.join("models", model_name)
    model.save(model_path)
    print(f"训练完成，最终模型已保存至: {model_path}.zip")

    # 7. 评估并可视化
    print("\n开始生成训练和评估结果图...")
    plot_training_curves(run_dir, run_dir)
    win_rate, avg_reward = evaluate(model, env)
    save_eval_image(win_rate, avg_reward, run_dir, 100)

def evaluate(model, env, n_episodes=100):
    """评估模型性能"""
    print(f"\n模型评估中，总回合数: {n_episodes}...")
    wins = 0
    total_reward = 0

    for i in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            # 获取合法的动作掩码
            action_masks = get_action_masks(env)

            # 模型预测
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

        # 检查是否获胜
        # 通过 unwrap 获取最底层的游戏环境实例
        game = env.unwrapped.game
        hero_id = 0 # 我们知道 Hero (训练对象) 的 ID 是 0

        if game.winner == hero_id:
            wins += 1

    win_rate = wins / n_episodes
    avg_reward = total_reward / n_episodes

    print(f"评估完成。")
    print(f"胜率: {win_rate * 100:.2f}%")
    print(f"平均奖励: {avg_reward:.2f}")
    return win_rate, avg_reward

if __name__ == "__main__":
    # 创建日志和模型目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train()
