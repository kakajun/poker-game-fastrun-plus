import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, List, Any
import random

from src.core.game import Game
from src.core.hand_type import HandType, Play
from src.env.action_space import ActionSpace
from src.env.obs_encoder import ObsEncoder

class PokerEnv(gym.Env):
    """
    跑得快强化学习环境
    Obs: [MyHand(52), OthersCount(2), LastPlay(62), PlayedHistory(52)] -> 168维
    Action: Discrete(200+) (Type-Len-MaxRank)
    Reward: Win=+100, Lose=-Remain, Bomb=+20
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        
        # 1. 核心组件
        self.game = Game()
        self.action_space_manager = ActionSpace()
        self.obs_encoder = ObsEncoder()
        
        # 2. 定义 Gym Space
        # Action: 离散 ID
        self.action_space = spaces.Discrete(self.action_space_manager.size)
        
        # Observation: Box(0, 1)
        # 维度由 encoder 决定
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=self.obs_encoder.shape, 
            dtype=np.float32
        )
        
        # 3. 运行时状态
        self.current_player_idx = 0
        self.steps_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # 重新开始游戏
        # 注意：Game 内部有 shuffle，我们也可以传入 seed
        self.game = Game(seed=seed)
        self.steps_count = 0
        
        # 谁是当前玩家？Game 会自动决定（持有红桃3者）
        self.current_player_idx = self.game.current_player
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步。
        注意：这是一个 Multi-Agent 环境。
        step() 只是让当前玩家执行动作。
        返回值 obs 是 *下一个* 玩家的观测。
        reward 是 *当前* 玩家获得的奖励。
        """
        self.steps_count += 1
        
        # 1. 解析动作
        # action_id -> Abstract Play
        abstract_play = self.action_space_manager.get_action(action_id)
        
        if abstract_play is None:
            # 非法动作 ID (理论上不应发生，如果使用了 Mask)
            # 给予重罚并结束? 或者 Pass?
            # 视为 Pass
            abstract_play = self.action_space_manager.pass_action
            
        # 2. 转换为具体动作 (Concrete Play)
        # 从当前手牌中找到匹配 abstract_play 的具体牌
        # 如果找不到，说明是非法动作（模型输出了手牌里没有的牌型）
        concrete_play = self._concretize_action(abstract_play)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if concrete_play is None:
            # 非法动作（例如手里没牌却想出牌）
            reward = -100.0
            truncated = True
            info = self._get_info()
            info["error"] = f"Illegal Action: Failed to concretize {abstract_play.type.name} len={abstract_play.length} max={abstract_play.max_rank}"
            return self._get_obs(), reward, terminated, truncated, info
            
        # 3. 执行动作
        try:
            is_over, events = self.game.step(concrete_play)
            
            # 4. 计算奖励
            # 基础奖励
            if is_over:
                terminated = True
                if self.game.winner == self.current_player_idx:
                    reward = 100.0
                else:
                    # 输了，扣除剩余牌数
                    # reward = - len(hand)
                    # 但 step 返回的是 *执行动作的玩家* 的奖励
                    # 如果我走了一步导致游戏结束，那一定是我赢了（我出完了）
                    # 除非是某种强制结束？
                    # 正常情况下，step 导致 is_over=True，意味着当前玩家出完了最后一张牌。
                    # 所以当前玩家一定是赢家。
                    reward = 100.0
            else:
                # 中间奖励
                # 炸弹奖励？
                # Game 内部已经计算了 bomb_scores。
                # 我们可以获取 bomb_scores 的变化量。
                # 但 Game.bomb_scores 是累计值。
                # 可以在 Game 中增加 step_reward 返回？
                # 或者在这里比较差异。
                # 合法出牌奖励，鼓励积极出牌
                if concrete_play.type != HandType.PASS:
                    reward += 2.0
                    
                # 炸弹奖励
                if concrete_play.type == HandType.BOMB:
                    reward += 10.0  # 适当降低即时奖励，防止乱炸，由 Game 逻辑处理大分
                
                # 控牌奖励：如果这一手牌让对手都过牌了，下一次还是我出牌
                # 这个逻辑在 step 后判断比较好，或者在 Wrapper 中判断

                    
        except ValueError as e:
            # 引擎抛错（例如管不住上家）
            reward = -100.0
            truncated = True
            info = self._get_info()
            info["error"] = str(e)
            return self._get_obs(), reward, terminated, truncated, info
            
        # 5. 更新状态
        # step 后，self.game.current_player 已经指向了下一个玩家
        self.current_player_idx = self.game.current_player
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return self.obs_encoder.encode(self.game, self.current_player_idx)

    def _get_info(self) -> Dict:
        """
        返回 Action Mask 等辅助信息
        """
        # 计算 Action Mask
        # 获取当前玩家所有合法动作
        legal_plays = self.game.get_legal_actions()
        
        mask = np.zeros(self.action_space_manager.size, dtype=np.int8)
        
        for play in legal_plays:
            aid = self.action_space_manager.get_id(play)
            if aid != -1 and 0 <= aid < self.action_space_manager.size:
                mask[aid] = 1
            elif play.type == HandType.PASS:
                mask[0] = 1
            
        return {
            "action_mask": mask,
            "player_id": self.current_player_idx
        }

    def _concretize_action(self, abstract_play: Play) -> Optional[Play]:
        """
        将抽象动作转换为具体动作。
        """
        # Pass 特殊处理
        if abstract_play.type == HandType.PASS:
            # 再次校验 Game 是否允许 Pass
            legal_plays = self.game.get_legal_actions()
            for p in legal_plays:
                if p.type == HandType.PASS:
                    return p
            return None # 不允许 Pass


            
        # 搜索匹配的合法动作
        legal_plays = self.game.get_legal_actions()
        candidates = []
        for p in legal_plays:
            if (p.type == abstract_play.type and 
                p.length == abstract_play.length and 
                p.max_rank == abstract_play.max_rank):
                candidates.append(p)
                
        if not candidates:
            return None
            
        # 如果有多个候选（如同点数不同花色），选第一个
        # legal_plays 通常由 ActionGenerator 生成，顺序是确定的
        # ActionGenerator 内部用 combinations，通常是按顺序的
        # 我们可以选 candidates[0]
        return candidates[0]

    def render(self):
        if self.render_mode == "human":
            print(f"--- Step {self.steps_count} ---")
            print(f"Player: {self.current_player_idx}")
            print(f"Hand: {self.game.hands[self.current_player_idx]}")
            print(f"Last Play: {self.game.last_play}")
