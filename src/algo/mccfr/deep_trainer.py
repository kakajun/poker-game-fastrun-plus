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
    """
    高度优化的 Numpy 经验回放池。
    直接使用预分配的 Numpy 数组，避免 Python 列表与 Numpy 之间的转换开销。
    """
    def __init__(self, capacity: int, state_dim: int, action_dim: int = None, is_strategy: bool = False):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.is_strategy = is_strategy

        # 预分配内存
        self.obs = np.zeros((capacity, state_dim), dtype=np.float32)
        if is_strategy:
            self.targets = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(capacity, dtype=np.int64)
            self.rewards = np.zeros(capacity, dtype=np.float32)

    def push(self, sample):
        if self.is_strategy:
            obs, target = sample
            self.obs[self.ptr] = obs
            self.targets[self.ptr] = target
        else:
            obs, action, reward = sample
            self.obs[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)

        if self.is_strategy:
            return self.obs[indices], self.targets[indices]
        else:
            return self.obs[indices], self.actions[indices], self.rewards[indices]

    def __len__(self):
        return self.size


def worker_collect_continuous(worker_id, model_state_dict_ref, queue, stop_event):
    """
    持续运行的游戏采样工作进程。
    """
    # 子进程中的初始化
    action_space = ActionSpace()
    encoder = ObsEncoder()
    state_dim = 57
    regret_net = DeepRegretNet(state_dim, action_space.size)

    # 局部缓存模型状态
    current_state_dict = None

    while not stop_event.is_set():
        # 定期尝试更新模型权重 (从共享内存/队列获取，这里简化为从 ref 获取)
        # 注意：在真正的大规模训练中，这里会更复杂
        if model_state_dict_ref.value is not None:
             regret_net.load_state_dict(model_state_dict_ref.value)

        regret_net.eval()
        game = Game()
        history = []
        strategy_samples = []

        while not game.is_over:
            p = game.current_player
            legal_actions = game.get_legal_actions()
            if not legal_actions: break

            obs = encoder.encode(game, p)

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                pred_regrets = regret_net(obs_tensor).numpy().flatten()

            legal_ids = [action_space.get_id(a) for a in legal_actions]
            legal_ids = [idx for idx in legal_ids if idx != -1]

            if not legal_ids:
                action = legal_actions[0]
                a_id = -1
            else:
                pos_regrets = np.maximum(pred_regrets[legal_ids], 0)
                sum_pos = np.sum(pos_regrets)
                probs_legal = pos_regrets / sum_pos if sum_pos > 0 else np.ones(len(legal_ids)) / len(legal_ids)

                probs = np.zeros(action_space.size)
                probs[legal_ids] = probs_legal
                strategy_samples.append((obs, probs))

                idx = np.random.choice(len(legal_ids), p=probs_legal)
                action = legal_actions[idx]
                a_id = legal_ids[idx]

            game.step(action)
            history.append((obs, p, a_id))

        scores = game.scores
        batch_samples = []
        for obs, p, a_id in history:
            if a_id != -1: batch_samples.append(('regret', obs, a_id, float(scores[p])))
        for obs, probs in strategy_samples:
            batch_samples.append(('strategy', obs, probs))

        if batch_samples:
            queue.put(batch_samples)

import torch.nn.utils.prune as prune

class DeepCFRTrainer:
    """深度 CFR 训练器 (异步常驻进程版)"""

    def __init__(self, num_workers: int = 4):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.action_space = ActionSpace()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 57
        self.action_dim = self.action_space.size

        # 模型初始化
        self.regret_net = DeepRegretNet(self.state_dim, self.action_dim).to(self.device)
        self.strategy_net = DeepStrategyNet(self.state_dim, self.action_dim).to(self.device)
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=1e-3)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=1e-3)

        self.regret_buffer = ReplayBuffer(1000000, self.state_dim)
        self.strategy_buffer = ReplayBuffer(1000000, self.state_dim, self.action_dim, is_strategy=True)

        # 验证集缓冲区 (容量较小)
        self.val_regret_buffer = ReplayBuffer(50000, self.state_dim)
        self.val_strategy_buffer = ReplayBuffer(50000, self.state_dim, self.action_dim, is_strategy=True)

        # 异步进程管理
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.model_state_ref = self.manager.Value(object, None)
        self.stop_event = mp.Event()
        self.queue = mp.Queue(maxsize=1000) # 限制队列大小防止内存溢出
        self.processes = []

    def start_workers(self):
        """启动常驻采样进程"""
        self.stop_event.clear()
        self.model_state_ref.value = {k: v.cpu() for k, v in self.regret_net.state_dict().items()}
        for i in range(self.num_workers):
            p = mp.Process(target=worker_collect_continuous,
                         args=(i, self.model_state_ref, self.queue, self.stop_event))
            p.daemon = True # 随主进程退出
            p.start()
            self.processes.append(p)
        print(f"Started {self.num_workers} resident worker processes.")

    def stop_workers(self):
        """停止所有进程"""
        self.stop_event.set()
        for p in self.processes:
            p.join(timeout=1)
        self.processes = []

    def collect_and_update(self, batch_size: int = 8192, train_updates: int = 100):
        """
        从队列收集数据并进行一次密集的 GPU 训练。
        """
        # 1. 尽可能多地从队列读取数据
        new_data_count = 0
        val_split = 0.1 # 10% 数据作为验证集

        while not self.queue.empty():
            try:
                batch_samples = self.queue.get_nowait()
                for sample in batch_samples:
                    is_val = random.random() < val_split
                    if sample[0] == 'regret':
                        if is_val:
                            self.val_regret_buffer.push(sample[1:])
                        else:
                            self.regret_buffer.push(sample[1:])
                    else:
                        if is_val:
                            self.val_strategy_buffer.push(sample[1:])
                        else:
                            self.strategy_buffer.push(sample[1:])
                    new_data_count += 1
            except:
                break

        # 2. 如果数据够了，进行训练
        if len(self.regret_buffer) >= batch_size:
            self.update_network(batch_size, train_updates)
            # 更新共享的模型权重
            self.model_state_ref.value = {k: v.cpu() for k, v in self.regret_net.state_dict().items()}

        return new_data_count

    def validate_network(self, batch_size: int = 4096) -> Tuple[float, float]:
        """
        在验证集上计算 Loss，用于早停
        """
        regret_loss = 0.0
        strategy_loss = 0.0

        # 1. 验证遗憾网络
        if len(self.val_regret_buffer) >= batch_size:
            self.regret_net.eval()
            with torch.no_grad():
                obs_np, action_np, reward_np = self.val_regret_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                action_batch = torch.from_numpy(action_np).to(self.device)
                reward_batch = torch.from_numpy(reward_np).to(self.device)

                pred_regrets = self.regret_net(obs_batch)
                target = pred_regrets.clone().detach()
                target[torch.arange(batch_size), action_batch] = reward_batch

                loss = F.mse_loss(pred_regrets, target)
                regret_loss = loss.item()

        # 2. 验证策略网络
        if len(self.val_strategy_buffer) >= batch_size:
            self.strategy_net.eval()
            with torch.no_grad():
                obs_np, target_probs_np = self.val_strategy_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                target_probs = torch.from_numpy(target_probs_np).to(self.device)

                pred_probs = self.strategy_net(obs_batch)
                loss = F.mse_loss(pred_probs, target_probs)
                strategy_loss = loss.item()

        return regret_loss, strategy_loss

    def prune_model(self, amount: float = 0.2):
        """
        对模型进行非结构化剪枝，移除权重较小的连接。
        amount: 剪枝比例 (0.0 - 1.0)
        """
        print(f"Applying L1 Unstructured Pruning (amount={amount})...")

        # 对 Regret Net 的全连接层剪枝
        for module in [self.regret_net.fc1, self.regret_net.fc2, self.regret_net.fc3, self.regret_net.head]:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight") # 永久化剪枝结果

        # 对 Strategy Net 的全连接层剪枝
        for module in [self.strategy_net.fc1, self.strategy_net.fc2, self.strategy_net.fc3, self.strategy_net.head]:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

        print("Model pruning completed.")

    def update_network(self, batch_size: int = 1024, updates: int = 1):
        """
        更新 Regret Net 和 Strategy Net
        """
        # 1. 更新遗憾网络
        if len(self.regret_buffer) >= batch_size:
            self.regret_net.train()
            for _ in range(updates):
                obs_np, action_np, reward_np = self.regret_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                action_batch = torch.from_numpy(action_np).to(self.device)
                reward_batch = torch.from_numpy(reward_np).to(self.device)

                pred_regrets = self.regret_net(obs_batch)
                target = pred_regrets.clone().detach()
                target[torch.arange(batch_size), action_batch] = reward_batch

                loss = F.mse_loss(pred_regrets, target)
                self.regret_optimizer.zero_grad()
                loss.backward()
                self.regret_optimizer.step()

        # 2. 更新策略网络 (Deep CFR 的核心：学习平均策略)
        if len(self.strategy_buffer) >= batch_size:
            self.strategy_net.train()
            for _ in range(updates):
                obs_np, target_probs_np = self.strategy_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                target_probs = torch.from_numpy(target_probs_np).to(self.device)

                pred_probs = self.strategy_net(obs_batch)
                # 使用 KL 散度或 MSE 损失让策略网络学习 Regret Matching 的结果
                loss = F.mse_loss(pred_probs, target_probs)

                self.strategy_optimizer.zero_grad()
                loss.backward()
                self.strategy_optimizer.step()

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
