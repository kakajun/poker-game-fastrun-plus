import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class DeepRegretNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DeepRegretNet, self).__init__()
        # 大幅提升网络宽度，从 256 增加到 1024，充分利用 GPU
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.head = nn.Linear(512, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.head(x)

class DeepStrategyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DeepStrategyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.head = nn.Linear(512, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.head(x), dim=-1)

class ModelWrapper:
    """
    模型包装器，处理 GPU/CPU 转换
    """
    def __init__(self, state_dim: int, action_dim: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.regret_net = DeepRegretNet(state_dim, action_dim).to(self.device)
        self.strategy_net = DeepStrategyNet(state_dim, action_dim).to(self.device)
        
    def predict_regrets(self, state_vec: np.ndarray) -> np.ndarray:
        self.regret_net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_vec).to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            regrets = self.regret_net(x)
            return regrets.cpu().numpy().flatten()
            
    def predict_strategy(self, state_vec: np.ndarray) -> np.ndarray:
        self.strategy_net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_vec).to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            strategy = self.strategy_net(x)
            return strategy.cpu().numpy().flatten()

    def save(self, path: str):
        torch.save({
            'regret_net': self.regret_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict()
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.regret_net.load_state_dict(checkpoint['regret_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
