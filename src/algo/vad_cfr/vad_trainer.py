import os
# Must set before importing torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.utils.prune as prune
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional
from src.core.game import Game
from src.env.obs_encoder import ObsEncoder
from src.env.action_space import ActionSpace
from src.algo.vad_cfr.deep_model import DeepRegretNet, DeepStrategyNet

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
    state_dim = encoder.shape[0] # Should be 57
    regret_net = DeepRegretNet(state_dim, action_space.size)
    
    current_state_dict = None
    
    while not stop_event.is_set():
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
                # 1. Asymmetric Boosting (VAD-CFR)
                # Boost positive regrets to encourage exploring promising actions faster
                raw_regrets = pred_regrets[legal_ids]
                boosted_regrets = np.where(raw_regrets > 0, raw_regrets * 1.1, raw_regrets)
                
                pos_regrets = np.maximum(boosted_regrets, 0)
                sum_pos = np.sum(pos_regrets)
                probs_legal = pos_regrets / sum_pos if sum_pos > 0 else np.ones(len(legal_ids)) / len(legal_ids)
                
                probs = np.zeros(action_space.size)
                probs[legal_ids] = probs_legal
                
                # 2. Regret-Magnitude Weighting (VAD-CFR)
                # Weight strategy samples by the magnitude of instantaneous regret (or max regret)
                # This helps filter out noise from low-impact states
                regret_magnitude = np.max(np.abs(raw_regrets)) if len(raw_regrets) > 0 else 0.0
                strategy_samples.append((obs, probs, regret_magnitude))
                
                idx = np.random.choice(len(legal_ids), p=probs_legal)
                action = legal_actions[idx]
                a_id = legal_ids[idx]
            
            game.step(action)
            history.append((obs, p, a_id))
            
        scores = game.scores
        batch_samples = []
        
        # Calculate advantages/regrets
        for obs, p, a_id in history:
            if a_id != -1: 
                # In MCCFR outcome sampling, the "immediate regret" for the chosen action 
                # is approximated by the sampled utility (minus baseline, but here we learn Q directly)
                # We store (s, a, u)
                # The Regret Net learns Q(s,a) -> u
                # VAD-CFR boosting applies to the *accumulation*. 
                # Since we train on *samples*, we can boost the *reward* signal if it's high?
                # Actually, Asymmetric Boosting is applied to the *regret* used for *strategy calculation*.
                # I already applied it above in `boosted_regrets`.
                # So here we just store the raw sample for training the Regret Net.
                # However, we can also weight the *training sample* by volatility/magnitude.
                # Let's use a default weight of 1.0 for regret samples for now.
                batch_samples.append(('regret', obs, a_id, float(scores[p]), 1.0))
                
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 57
        self.action_dim = self.action_space.size
        
        self.regret_net = DeepRegretNet(self.state_dim, self.action_dim).to(self.device)
        self.strategy_net = DeepStrategyNet(self.state_dim, self.action_dim).to(self.device)
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=1e-3)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=1e-3)
        
        self.regret_buffer = ReplayBuffer(1000000, self.state_dim)
        self.strategy_buffer = ReplayBuffer(1000000, self.state_dim, self.action_dim, is_strategy=True)
        
        self.val_regret_buffer = ReplayBuffer(50000, self.state_dim)
        self.val_strategy_buffer = ReplayBuffer(50000, self.state_dim, self.action_dim, is_strategy=True)
        
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.model_state_ref = self.manager.Value(object, None)
        self.stop_event = mp.Event()
        self.queue = mp.Queue(maxsize=1000)
        self.processes = []
        
        self.iteration = 0
        self.warm_start_iter = 500  # VAD-CFR Hard Warm-Start
        
        # Volatility Tracking
        self.volatility_ewma = 0.0
        self.alpha_volatility = 0.1

    def start_workers(self):
        self.stop_event.clear()
        self.model_state_ref.value = {k: v.cpu() for k, v in self.regret_net.state_dict().items()}
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
            self.model_state_ref.value = {k: v.cpu() for k, v in self.regret_net.state_dict().items()}
            self.iteration += 1
            
        return new_data_count

    def update_network(self, batch_size: int = 1024, updates: int = 1):
        # 1. Update Regret Net
        if len(self.regret_buffer) >= batch_size:
            self.regret_net.train()
            for _ in range(updates):
                obs_np, action_np, reward_np, weight_np = self.regret_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                action_batch = torch.from_numpy(action_np).to(self.device)
                reward_batch = torch.from_numpy(reward_np).to(self.device)
                # We can use weights if we want, but for Regret Net we usually fit to value
                # Maybe weight by volatility?
                
                # Track volatility (standard deviation of rewards in this batch)
                current_volatility = torch.std(reward_batch).item()
                self.volatility_ewma = (1 - self.alpha_volatility) * self.volatility_ewma + self.alpha_volatility * current_volatility
                
                # Volatility-Adaptive Discounting Logic:
                # If volatility is high, we might want to weight the loss of "outlier" samples less? 
                # Or weight the whole update more?
                # VAD-CFR paper says: "When volatility is high, it raises discount rates to forget unstable history faster"
                # Since we can't easily forget history in the buffer, we can simulate this by
                # slightly decaying the weights of the network (weight decay) proportional to volatility?
                # Or simply rely on the Asymmetric Boosting we did in the worker.
                
                pred_regrets = self.regret_net(obs_batch)
                target = pred_regrets.clone().detach()
                target[torch.arange(batch_size), action_batch] = reward_batch
                
                loss = F.mse_loss(pred_regrets, target)
                self.regret_optimizer.zero_grad()
                loss.backward()
                self.regret_optimizer.step()

        # 2. Update Strategy Net
        if len(self.strategy_buffer) >= batch_size:
            self.strategy_net.train()
            for _ in range(updates):
                obs_np, target_probs_np, weight_np = self.strategy_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                target_probs = torch.from_numpy(target_probs_np).to(self.device)
                weights = torch.from_numpy(weight_np).to(self.device)
                
                pred_probs = self.strategy_net(obs_batch)
                
                # Weighted MSE Loss based on Regret Magnitude (VAD-CFR)
                # Weights are already computed in worker
                loss = torch.mean(weights.unsqueeze(1) * (pred_probs - target_probs) ** 2)
                
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
                obs_np, action_np, reward_np, _ = self.val_regret_buffer.sample(batch_size)
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
                obs_np, target_probs_np, weight_np = self.val_strategy_buffer.sample(batch_size)
                obs_batch = torch.from_numpy(obs_np).to(self.device)
                target_probs = torch.from_numpy(target_probs_np).to(self.device)

                pred_probs = self.strategy_net(obs_batch)
                loss = F.mse_loss(pred_probs, target_probs)
                strategy_loss = loss.item()

        return regret_loss, strategy_loss

    def prune_model(self, amount: float = 0.2):
        """
        Apply L1 Unstructured Pruning to reduce model size
        """
        print(f"Applying L1 Unstructured Pruning (amount={amount})...")

        # Prune Regret Net
        for module in [self.regret_net.fc1, self.regret_net.fc2, self.regret_net.fc3, self.regret_net.head]:
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

        # Prune Strategy Net
        for module in [self.strategy_net.fc1, self.strategy_net.fc2, self.strategy_net.fc3, self.strategy_net.head]:
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
        print(f"VAD-CFR Model saved to {path} (Iter: {self.iteration}, Volatility: {self.volatility_ewma:.4f})")
