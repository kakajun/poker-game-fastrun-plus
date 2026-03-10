import os
# 必须在最顶部设置，防止多进程下 OMP 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from src.core.game import Game
from src.env.obs_encoder import ObsEncoder
from src.env.action_space import ActionSpace
# 修改为绝对路径引入
from src.algo.mccfr.deep_model import DeepRegretNet, DeepStrategyNet


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0

    def push(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.ptr] = sample
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


def worker_collect(num_games, model_state, queue):
    """
    独立的工作函数，用于多进程。
    """
    # 子进程中的初始化
    action_space = ActionSpace()
    encoder = ObsEncoder()
    regret_net = DeepRegretNet(42, action_space.size)
    if model_state:
        regret_net.load_state_dict(model_state)
    regret_net.eval()

    # 优化：使用 TorchScript 加速推理 (如果可能)
    # scripted_net = torch.jit.script(regret_net)

    for _ in range(num_games):
        game = Game()
        history = []

        while not game.is_over:
            p = game.current_player
            legal_actions = game.get_legal_actions()
            if not legal_actions:
                break

            # 性能优化：随机探索跳过网络推理
            # 在训练初期，随机探索不仅快，而且有助于发现新策略
            # 我们设置 30% 的概率直接随机选择动作，不走神经网络
            # 随着训练进行，这个比例可以动态调整，但为了简单起见，固定一个值
            if random.random() < 0.3:
                action = random.choice(legal_actions)
                a_id = action_space.get_id(action)
                # 随机动作的 obs 也可以收集，但为了质量，我们这里只收集模型推理的数据？
                # 不，随机数据也是有效的 Off-policy 数据。
                # 但我们需要记录 obs，所以还是得编码
                obs = encoder.encode(game, p)
            else:
                obs = encoder.encode(game, p)
                with torch.no_grad():
                    # 优化：不转 tensor，直接用 numpy (如果模型支持)
                    # 但 PyTorch 还是需要 tensor
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    pred_regrets = regret_net(obs_tensor).numpy().flatten()

                legal_ids = [action_space.get_id(a) for a in legal_actions]
                legal_ids = [idx for idx in legal_ids if idx != -1]

                if not legal_ids:
                    action = legal_actions[0]
                    a_id = -1
                else:
                    # Regret Matching
                    pos_regrets = np.maximum(pred_regrets[legal_ids], 0)
                    sum_pos = np.sum(pos_regrets)
                    if sum_pos > 0:
                        probs = pos_regrets / sum_pos
                    else:
                        probs = np.ones(len(legal_ids)) / len(legal_ids)

                    idx = np.random.choice(len(legal_ids), p=probs)
                    action = legal_actions[idx]
                    a_id = legal_ids[idx]

            game.step(action)
            history.append((obs, p, a_id))

        # 记录每步的收益
        scores = game.scores

        # 优化：批量放入队列
        # 减少 queue.put 的调用次数，直接放一个列表
        # 这需要修改主进程的接收逻辑，但我们可以先在这里优化
        # Queue 的 put 操作有锁开销，批量放能显著提升吞吐
        batch_samples = []
        for obs, p, a_id in history:
            if a_id != -1:
                batch_samples.append((obs, a_id, float(scores[p])))

        if batch_samples:
            queue.put(batch_samples)


class DeepCFRTrainer:
    """深度 CFR 训练器 (GPU 并发加速版)"""

    def __init__(self, num_workers: int = 4):
        # 确保 multiprocessing 在 Windows 下能正常工作
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.action_space = ActionSpace()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 42
        self.action_dim = self.action_space.size

        self.regret_net = DeepRegretNet(
            self.state_dim, self.action_dim).to(self.device)
        self.strategy_net = DeepStrategyNet(
            self.state_dim, self.action_dim).to(self.device)

        self.regret_optimizer = optim.Adam(
            self.regret_net.parameters(), lr=1e-3)
        self.regret_buffer = ReplayBuffer(100000)
        self.num_workers = num_workers

    def train_step(self, games_per_worker: int = 5):
        """单步训练：采样 -> 更新"""
        queue = mp.Queue()
        # 优化：只传输 state_dict，不传输整个模型对象
        model_state = {k: v.cpu()
                       for k, v in self.regret_net.state_dict().items()}

        processes = []
        for _ in range(self.num_workers):
            p = mp.Process(target=worker_collect, args=(
                games_per_worker, model_state, queue))
            p.start()
            processes.append(p)

        # 收集数据 (带超时防止阻塞)
        data_count = 0
        active_workers = self.num_workers

        # 优化：批量读取队列，减少锁竞争
        while active_workers > 0:
            try:
                # 尝试批量获取，或者设置较短的 timeout
                # 这里的逻辑是：只要进程还活着或者队列不为空，就继续读
                if not queue.empty():
                    # 现在 queue 中的元素是一个 batch list
                    batch_samples = queue.get_nowait()
                    for sample in batch_samples:
                        self.regret_buffer.push(sample)
                        data_count += 1
                else:
                    # 检查是否有进程结束
                    # 注意：join() 会阻塞，所以用 is_alive()
                    still_alive = 0
                    for p in processes:
                        if p.is_alive():
                            still_alive += 1

                    if still_alive < active_workers:
                        active_workers = still_alive

                    if active_workers == 0 and queue.empty():
                        break

                    time.sleep(0.001)  # 极短睡眠避免死循环占满 CPU
            except:
                continue

        for p in processes:
            p.join()

        return data_count

    def update_network(self, batch_size: int = 1024, updates: int = 1):
        """
        公开的更新接口
        updates: 在同一批数据上更新多少次 (或者采样多少个 batch)
        """
        if len(self.regret_buffer) < batch_size:
            return

        self.regret_net.train()

        for _ in range(updates):
            samples = self.regret_buffer.sample(batch_size)

            # 转换为 Tensor (一次性转换比逐个转换快)
            # 优化：使用 numpy stack 加速列表转换
            obs_np = np.array([s[0] for s in samples], dtype=np.float32)
            actions_np = np.array([s[1] for s in samples], dtype=np.int64)
            rewards_np = np.array([s[2] for s in samples], dtype=np.float32)

            obs_batch = torch.from_numpy(obs_np).to(self.device)
            action_batch = torch.from_numpy(actions_np).to(self.device)
            reward_batch = torch.from_numpy(rewards_np).to(self.device)

            pred_regrets = self.regret_net(obs_batch)

            target = pred_regrets.clone().detach()
            # 向量化赋值，比 for loop 快得多
            # target[arange, actions] = rewards
            target[torch.arange(batch_size), action_batch] = reward_batch

            loss = F.mse_loss(pred_regrets, target)

            self.regret_optimizer.zero_grad()
            loss.backward()
            self.regret_optimizer.step()

    def _update_network(self, batch_size: int = 128):
        # 保留旧接口兼容，但调用新逻辑
        self.update_network(batch_size, updates=1)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'regret_net': self.regret_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict()
        }, path)
        print(f"Deep CFR Model saved to {path}")
