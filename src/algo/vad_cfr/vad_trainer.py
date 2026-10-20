from src.algo.vad_cfr.deep_model import DeepRegretNet, DeepStrategyNet
from src.env.action_space import ActionSpace
from src.env.obs_encoder import ObsEncoder
from src.core.game import Game
from typing import List, Dict, Tuple, Optional
import time
import random
import numpy as np
import torch.nn.utils.prune as prune
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
# Must set before importing torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ReplayBuffer:
    """
    Optimized Replay Buffer for VAD-CFR.
    Includes support for weighted sampling (volatility-adaptive).
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int = None, is_strategy: bool = False):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.is_strategy = is_strategy

        # Pre-allocate memory
        self.obs = np.zeros((capacity, state_dim), dtype=np.float32)
        # Store weights for VAD mechanisms (e.g., regret magnitude)
        self.weights = np.zeros(capacity, dtype=np.float32)

        if is_strategy:
            self.targets = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(capacity, dtype=np.int64)
            self.rewards = np.zeros(capacity, dtype=np.float32)

    def push(self, sample):
        if self.is_strategy:
            obs, target, weight = sample
            self.obs[self.ptr] = obs
            self.targets[self.ptr] = target
            self.weights[self.ptr] = weight
        else:
            obs, action, reward, weight = sample
            self.obs[self.ptr] = obs
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.weights[self.ptr] = weight

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        # Uniform sampling for now, but weights are returned for loss calculation
        indices = np.random.randint(0, self.size, size=batch_size)

        if self.is_strategy:
            return self.obs[indices], self.targets[indices], self.weights[indices]
        else:
            return self.obs[indices], self.actions[indices], self.rewards[indices], self.weights[indices]

    def __len__(self):
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0


def worker_collect_vad(worker_id, model_state_dict_ref, queue, stop_event):
    """
    VAD-CFR Worker Process.
    Implements Asymmetric Boosting and Regret-Magnitude Weighting.
    """
    action_space = ActionSpace()
    encoder = ObsEncoder()
    state_dim = encoder.shape[0]  # Should be 60
    regret_net = DeepRegretNet(state_dim, action_space.size)

    while not stop_event.is_set():
        if model_state_dict_ref.value is not None:
            regret_net.load_state_dict(model_state_dict_ref.value)

        regret_net.eval()
        game = Game()
        history = []
        strategy_samples = []

        # Epsilon-greedy exploration factor
        epsilon = 0.05

        while not game.is_over:
            p = game.current_player
            legal_actions = game.get_legal_actions()
            if not legal_actions:
                break

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
                # 1. Asymmetric Boosting (VAD-CFR)
                raw_regrets = pred_regrets[legal_ids]
                boosted_regrets = np.where(
                    raw_regrets > 0, raw_regrets * 1.2, raw_regrets)

                pos_regrets = np.maximum(boosted_regrets, 0)
                sum_pos = np.sum(pos_regrets)

                if sum_pos > 0:
                    probs_legal = pos_regrets / sum_pos
                else:
                    probs_legal = np.ones(len(legal_ids)) / len(legal_ids)

                # Add Epsilon-Greedy exploration
                probs_legal = (1 - epsilon) * probs_legal + epsilon * \
                    (np.ones(len(legal_ids)) / len(legal_ids))
                probs_legal = probs_legal / np.sum(probs_legal)

                probs = np.zeros(action_space.size)
                probs[legal_ids] = probs_legal

                # 2. Regret-Magnitude Weighting (VAD-CFR)
                regret_magnitude = np.max(np.abs(raw_regrets)) if len(
                    raw_regrets) > 0 else 0.0
                strategy_samples.append((obs, probs, regret_magnitude))

                idx = np.random.choice(len(legal_ids), p=probs_legal)
                action = legal_actions[idx]
                a_id = legal_ids[idx]

            game.step(action)
            # Record if it was a 'Pass' action (Play of type PASS)
            is_pass = (a_id == 0)  # 0 is typically PASS in ActionSpace
            history.append((obs, p, a_id, is_pass))

        scores = game.scores
        batch_samples = []

        # 3. Discounted Reward & Reward Shaping
        # Later actions are more important. Weight = (step / total_steps)^2
        total_steps = len(history)

        for step, (obs, p, a_id, is_pass) in enumerate(history):
            if a_id != -1:
                # Base reward from game outcome
                reward = float(scores[p])

                # Reward Shaping:
                # - Small bonus for playing cards (reducing hand size)
                # - Small penalty for passing (especially if others are about to win)
                if not is_pass:
                    reward += 0.5
                else:
                    reward -= 0.1

                # Discounting: Later steps have higher weight in the buffer
                # This helps the network focus on the 'climax' of the game
                weight = ((step + 1) / total_steps) ** 2

                batch_samples.append(('regret', obs, a_id, reward, weight))

        for obs, probs, weight in strategy_samples:
            batch_samples.append(('strategy', obs, probs, weight))

        if batch_samples:
            queue.put(batch_samples)


class VADCFRTrainer:
    """
    Volatility-Adaptive Discounted CFR (VAD-CFR) Trainer.
    Features:
    - Asymmetric Instantaneous Boosting (in Worker)
    - Hard Warm-Start for Policy Averaging
    - Volatility-Sensitive Discounting (via Loss Weighting)
    """

    def __init__(self, num_workers: int = 4):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.action_space = ActionSpace()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 60
        self.action_dim = self.action_space.size

        self.regret_net = DeepRegretNet(
            self.state_dim, self.action_dim).to(self.device)
        self.strategy_net = DeepStrategyNet(
            self.state_dim, self.action_dim).to(self.device)
        self.regret_optimizer = optim.Adam(
            self.regret_net.parameters(), lr=1e-3)
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=1e-3)

        self.regret_buffer = ReplayBuffer(1000000, self.state_dim)
        self.strategy_buffer = ReplayBuffer(
            1000000, self.state_dim, self.action_dim, is_strategy=True)

        self.val_regret_buffer = ReplayBuffer(50000, self.state_dim)
        self.val_strategy_buffer = ReplayBuffer(
            50000, self.state_dim, self.action_dim, is_strategy=True)

        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.model_state_ref = self.manager.Value(object, None)
        self.stop_event = mp.Event()
        self.queue = mp.Queue(maxsize=1000)
        self.processes = []

        self.iteration = 0
        # VAD-CFR Hard Warm-Start (Reduced from 500)
        self.warm_start_iter = 100

        # Volatility Tracking
        self.volatility_ewma = 0.0
        self.alpha_volatility = 0.1

    def start_workers(self):
        self.stop_event.clear()
        self.model_state_ref.value = {
            k: v.cpu() for k, v in self.regret_net.state_dict().items()}
        for i in range(self.num_workers):
            p = mp.Process(target=worker_collect_vad,
                           args=(i, self.model_state_ref, self.queue, self.stop_event))
            p.daemon = True
            p.start()
            self.processes.append(p)
        print(f"Started {self.num_workers} VAD-CFR worker processes.")

    def stop_workers(self):
        self.stop_event.set()
        for p in self.processes:
            p.join(timeout=1)
        self.processes = []

    def collect_and_update(self, batch_size: int = 8192, train_updates: int = 100):
        new_data_count = 0
        val_split = 0.1

        # Collect data
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
                        # 3. Hard Warm-Start (VAD-CFR)
                        # Only add to strategy buffer after warm-start period
                        if self.iteration >= self.warm_start_iter:
                            if is_val:
                                self.val_strategy_buffer.push(sample[1:])
                            else:
                                self.strategy_buffer.push(sample[1:])
                    new_data_count += 1
            except:
                break

        # Train
        if len(self.regret_buffer) >= batch_size:
            self.update_network(batch_size, train_updates)
            self.model_state_ref.value = {
                k: v.cpu() for k, v in self.regret_net.state_dict().items()}
            self.iteration += 1

        return new_data_count

    def update_network(self, batch_size: int = 1024, updates: int = 1):
        # 1. Update Regret Net
        if len(self.regret_buffer) >= batch_size:
            self.regret_net.train()
            for _ in range(updates):
                obs_np, action_np, reward_np, weight_np = self.regret_buffer.sample(
                    batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                action_batch = torch.from_numpy(action_np).to(self.device)
                reward_batch = torch.from_numpy(reward_np).to(self.device)
                weights = torch.from_numpy(weight_np).to(self.device)

                # Track volatility
                current_volatility = torch.std(reward_batch).item()
                self.volatility_ewma = (
                    1 - self.alpha_volatility) * self.volatility_ewma + self.alpha_volatility * current_volatility

                # Volatility-Adaptive Learning Rate
                lr_scale = 1.0 / (1.0 + 0.1 * self.volatility_ewma)
                for param_group in self.regret_optimizer.param_groups:
                    param_group['lr'] = 1e-3 * lr_scale

                pred_regrets = self.regret_net(obs_batch)
                target = pred_regrets.clone().detach()
                target[torch.arange(batch_size), action_batch] = reward_batch

                # Weighted MSE Loss for Regret Net (Implementing Discounting)
                # Ensure weights are normalized
                w_norm = weights / (torch.mean(weights) + 1e-8)
                loss = torch.mean(w_norm.unsqueeze(
                    1) * (pred_regrets - target) ** 2)

                self.regret_optimizer.zero_grad()
                loss.backward()
                self.regret_optimizer.step()

        # 2. Update Strategy Net
        if len(self.strategy_buffer) >= batch_size:
            self.strategy_net.train()
            # Similarly, adjust strategy learning rate
            lr_scale = 1.0 / (1.0 + 0.05 * self.volatility_ewma)
            for param_group in self.strategy_optimizer.param_groups:
                param_group['lr'] = 1e-3 * lr_scale

            for _ in range(updates):
                obs_np, target_probs_np, weight_np = self.strategy_buffer.sample(
                    batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                target_probs = torch.from_numpy(
                    target_probs_np).to(self.device)
                weights = torch.from_numpy(weight_np).to(self.device)

                pred_probs = self.strategy_net(obs_batch)

                # Weighted MSE Loss based on Regret Magnitude (VAD-CFR)
                # Ensure weights are normalized to prevent exploding gradients
                w_norm = weights / (torch.mean(weights) + 1e-8)
                loss = torch.mean(w_norm.unsqueeze(
                    1) * (pred_probs - target_probs) ** 2)

                self.strategy_optimizer.zero_grad()
                loss.backward()
                self.strategy_optimizer.step()

    def validate_network(self, batch_size: int = 4096) -> Tuple[float, float]:
        """
        Compute validation loss
        """
        regret_loss = 0.0
        strategy_loss = 0.0

        # 1. Validate Regret Net
        if len(self.val_regret_buffer) >= batch_size:
            self.regret_net.eval()
            with torch.no_grad():
                obs_np, action_np, reward_np, _ = self.val_regret_buffer.sample(
                    batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                action_batch = torch.from_numpy(action_np).to(self.device)
                reward_batch = torch.from_numpy(reward_np).to(self.device)

                pred_regrets = self.regret_net(obs_batch)
                target = pred_regrets.clone().detach()
                target[torch.arange(batch_size), action_batch] = reward_batch

                loss = F.mse_loss(pred_regrets, target)
                regret_loss = loss.item()

        # 2. Validate Strategy Net
        if len(self.val_strategy_buffer) >= batch_size:
            self.strategy_net.eval()
            with torch.no_grad():
                obs_np, target_probs_np, weight_np = self.val_strategy_buffer.sample(
                    batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                target_probs = torch.from_numpy(
                    target_probs_np).to(self.device)

                pred_probs = self.strategy_net(obs_batch)
                loss = F.mse_loss(pred_probs, target_probs)
                strategy_loss = loss.item()

        return regret_loss, strategy_loss

    def prune_model(self, amount: float = 0.2):
        """
        Pruning to compress model size and reduce overfitting.
        """
        print(f"Applying L1 Unstructured Pruning (amount={amount})...")

        # Prune Regret Net (ResNet structure)
        for module in [self.regret_net.fc_in, self.regret_net.res1.fc, self.regret_net.res2.fc, self.regret_net.fc_mid, self.regret_net.head]:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

        # Prune Strategy Net
        for module in [self.strategy_net.fc_in, self.strategy_net.res1.fc, self.strategy_net.res2.fc, self.strategy_net.fc_mid, self.strategy_net.head]:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

        print("Model pruning completed.")

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'regret_net': self.regret_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'iteration': self.iteration,
            'volatility': self.volatility_ewma
        }, path)
        print(
            f"VAD-CFR Model saved to {path} (Iter: {self.iteration}, Volatility: {self.volatility_ewma:.4f})")

    def load_model(self, path: str):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            # Use strict=True to detect architecture mismatch early
            self.regret_net.load_state_dict(
                checkpoint['regret_net'], strict=True)
            self.strategy_net.load_state_dict(
                checkpoint['strategy_net'], strict=True)
            self.iteration = checkpoint.get('iteration', 0)
            self.volatility_ewma = checkpoint.get('volatility', 0.0)
            print(
                f"VAD-CFR Model loaded from {path} (Iter: {self.iteration}, Volatility: {self.volatility_ewma:.4f})")
            return True
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            print("Architecture might have changed. Starting with fresh weights.")
            return False
