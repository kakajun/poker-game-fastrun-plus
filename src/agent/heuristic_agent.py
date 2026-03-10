import numpy as np
from src.env.action_space import ActionSpace
from src.core.hand_type import HandType

class HeuristicAgent:
    """
    启发式策略智能体。
    基于简单规则：
    1. 优先出点数小的牌（节省大牌）。
    2. 除非必要，否则不轻易动用炸弹。
    3. 如果有多种非炸弹选择，选点数最小的。
    """
    def __init__(self):
        self.action_space_manager = ActionSpace()
        
    def act(self, obs, mask):
        legal_action_ids = np.where(mask == 1)[0]
        if len(legal_action_ids) == 0:
            return 0 # Pass
            
        # 1. 分离 Pass、非炸弹动作、炸弹动作
        pass_id = 0
        normal_actions = []
        bomb_actions = []
        
        for aid in legal_action_ids:
            if aid == 0:
                continue
            
            play = self.action_space_manager.get_action(aid)
            if play.is_bomb:
                bomb_actions.append((aid, play))
            else:
                normal_actions.append((aid, play))
                
        # 2. 策略逻辑
        # 如果有普通动作，选点数最小的（即 max_rank 最小）
        if normal_actions:
            # 按 max_rank 升序排列
            normal_actions.sort(key=lambda x: x[1].max_rank)
            return normal_actions[0][0]
            
        # 如果只有炸弹动作，也选点数最小的
        if bomb_actions:
            bomb_actions.sort(key=lambda x: x[1].max_rank)
            return bomb_actions[0][0]
            
        # 只有 Pass 可选
        return pass_id
