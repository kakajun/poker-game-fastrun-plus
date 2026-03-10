import numpy as np

class RandomAgent:
    """
    随机出牌策略。
    """
    def __init__(self, action_space_size=None):
        self.action_space_size = action_space_size
        
    def act(self, obs, mask):
        # 随机选择一个合法的动作
        legal_actions = np.where(mask == 1)[0]
        if len(legal_actions) == 0:
            # 理论上不应发生，如果 mask 全 0，说明无路可走
            return 0 # Pass
        return np.random.choice(legal_actions)
