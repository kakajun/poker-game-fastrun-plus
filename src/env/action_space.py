import itertools
from typing import List, Dict, Tuple, Optional
import pickle
import os

from src.core.card import Card, Rank, Suit
from src.core.hand_type import HandType, Play
from src.core.action_generator import ActionGenerator

class ActionSpace:
    """
    全局离散动作空间管理。
    将所有可能的合法出牌动作映射为唯一 ID。
    """
    
    def __init__(self):
        self.action_to_id: Dict[str, int] = {}
        self.id_to_action: Dict[int, Play] = {}
        self.pass_action = Play(HandType.PASS, [], 0, 0)
        self.size = 0
        
        if not self.load():
            print("Building Action Space (this may take a while)...")
            self.build()
            self.save()
        print(f"Action Space ready. Size: {self.size}")

    def get_id(self, play: Play) -> int:
        """获取动作 ID (Abstract)"""
        if play.type == HandType.PASS:
            return 0
        key = self._get_key(play)
        return self.action_to_id.get(key, -1)
        
    def get_action(self, action_id: int) -> Optional[Play]:
        """根据 ID 获取动作模板 (No cards)"""
        if action_id == 0:
            return self.pass_action
        return self.id_to_action.get(action_id)

    def _get_key(self, play: Play) -> str:
        """生成唯一键: Type_Len_Max (Abstract)"""
        # 确保 max_rank 是 int
        rank_val = play.max_rank
        if hasattr(rank_val, 'value'):
            rank_val = rank_val.value
        return f"{play.type.name}_{play.length}_{rank_val}"

    def build(self):
        """构建动作空间"""
        self.action_to_id = {}
        self.id_to_action = {}
        idx = 0
        
        # 0. Pass
        self._add_action_abstract(self.pass_action.type, self.pass_action.length, self.pass_action.max_rank)
        idx += 1
        
        # 构造全牌堆 (45张)
        full_deck_cards = []
        # 3-Q (4张)
        for r in range(3, 13):
            for s in Suit:
                full_deck_cards.append(Card(Rank(r), s))
        # K (3张)
        full_deck_cards.append(Card(Rank.KING, Suit.CLUB))
        full_deck_cards.append(Card(Rank.KING, Suit.HEART))
        full_deck_cards.append(Card(Rank.KING, Suit.SPADE))
        # A (1张)
        full_deck_cards.append(Card(Rank.ACE, Suit.SPADE))
        # 2 (1张)
        full_deck_cards.append(Card(Rank.TWO, Suit.HEART))
        
        print("Generating all actions from full deck...")
        # ActionGenerator.get_all_actions 会生成基于当前手牌的所有组合。
        # 如果给它 45 张牌，它会生成所有可能的顺子、连对、飞机等。
        # 但这有一个问题：它只生成 *存在于手牌中* 的组合。
        # 对于顺子，比如 3-4-5-6-7，如果有 4 张 3，4 张 4...
        # 它会生成所有花色组合的顺子吗？
        # ActionGenerator._generate_sequences 是递归枚举所有单张组合。
        # 是的，它会生成所有具体的牌组合。
        # 这可能会导致组合爆炸！
        # 45张牌的全排列太大了。
        # 但这里只生成合法的牌型。
        # 比如顺子 3-7。每张牌有4种花色。4^5 = 1024 种顺子。
        # 连对 33-44-55。每对有 C(4,2)=6 种。6^3 = 216 种。
        # 飞机 333-444。每三张有 C(4,3)=4 种。4^2 = 16 种。
        # 飞机带翅膀。带单牌：4^2 * 40 * 39 ... 太大了。
        # 如果我们把所有具体的动作都列出来，空间可能会达到几百万。
        # 这对于 DQN/PPO 来说太大了。
        
        # 解决方案：抽象动作空间 (Abstract Action Space)
        # 我们可以只根据牌型、长度、关键点数来定义动作，而不关心具体的花色组合？
        # 但是，出牌必须指定具体的牌。
        # 如果模型输出 "出顺子3-7"，具体的花色怎么定？
        # 可以由规则引擎自动选择（例如选花色最小的）？
        # 或者模型输出具体的牌？
        # 这是一个经典难题。
        # 在斗地主AI (DouZero) 中，动作空间是具体的。
        # 他们怎么处理的？
        # DouZero 使用了动作编码 + 蒙特卡洛采样？
        # 或者动作空间其实没那么大？
        # 让我们估算一下。
        # 单张: 45
        # 对子: ~40 * 6 = 240
        # 三张: ~40 * 4 = 160
        # 三带一: 160 * 40 = 6400 (带单牌)
        # 顺子: 4^5 * 8 (长度5-12) ≈ 1000 * 8 = 8000? 不止。
        # 顺子长度5: 4^5 = 1024. 起点3-10 (8个). 8192.
        # 顺子长度6: 4^6 = 4096. 起点3-9 (7个). 28672.
        # ... 总计可能几十万。
        
        # 看来不能用全具体动作空间。
        # 必须用抽象动作空间。
        # ID = Type_Length_MaxRank
        # 例如：顺子_5_7 (34567)。
        # 当模型选择这个动作时，环境自动选择具体的牌（例如花色最小的组合）。
        # 这会损失一些策略性（比如留红桃2配顺子），但在跑得快里，通常只有大小关键，花色不关键（除了首出红桃3）。
        # 我们可以加一个规则：优先保留大牌/特殊牌？或者随机选择？
        # 建议：优先出花色小的牌，保留花色大的（如黑桃、红桃）。
        # 这样动作空间就小多了：
        # 单张: 13
        # 对子: 13
        # 三张: 13
        # 三带一: 13 (带单牌? 带谁不重要，重要的是带了单牌。但带的单牌大小会影响。)
        # 三带一应该细化为：三张点数 + 被带单牌点数。 13 * 12 = 156.
        # 顺子: 长度(5-12) * 起点. ~几十个.
        # 连对: 长度 * 起点. ~几十个.
        # 飞机: 长度 * 起点.
        # 飞机带翅膀: 长度 * 起点 * 翅膀点数组合? (这就复杂了)
        
        # 重新考虑：
        # 如果我们用 "抽象动作"，那么状态空间也应该是抽象的吗？
        # 不，状态空间包含具体手牌。
        # 动作选择 "出顺子3-7"。环境从手牌里找出一个顺子3-7。
        # 如果有多个（花色不同），选哪一个？
        # 简单的策略：选花色ID最小的组合。
        # 这样模型不需要关心花色。
        
        # 动作ID定义：
        # 0: Pass
        # 1-13: Single 3..2
        # 14-26: Pair 3..2
        # 27-39: Triple 3..2
        # 40-195: Triple 3..2 + Single 3..2 (13 * 12)
        # ...
        # 顺子: Len(5..12) * MaxRank.
        # 连对: Len(2..10) * MaxRank.
        # 飞机: Len(2..6) * MaxRank.
        # 飞机带翅膀: Len * MaxRank * WingRank?
        # 飞机带翅膀比较麻烦，因为翅膀可以是任意单牌。
        # 比如 333444 + 5 + 6. 
        # 动作需要指定 5 和 6 吗？
        # 是的，否则不知道带谁。
        # 如果只指定主体，翅膀自动选最小？这可能导致把关键牌带出去了。
        # 比如我有 333444 5 6 2. 我想带 5 6，留 2。
        # 如果自动选最小，就是带 5 6。
        # 如果我有 333444 5 2. 我只能带 5 2.
        # 所以动作必须包含翅膀信息。
        
        # 这样动作空间还是很大。
        # 也许我们应该限制模型输出：
        # 1. 牌型 (Type + Length + MainRank)
        # 2. 具体的牌由 Mask 决定？
        # 不，标准 PPO 输出是 Discrete(N)。
        
        # 回到具体动作空间，但通过 Action Mask 限制。
        # 如果全空间太大，无法构建 Softmax。
        # 几十万维的 Softmax 太慢。
        
        # 妥协方案：
        # 使用 DouZero 的方法：动作编码为矩阵，不是 Discrete ID。
        # 但我们想用标准 Gym + Stable Baselines3 (PPO)。
        # PPO 支持 MultiDiscrete？
        # 或者：
        # 将动作拆分为：
        # 1. 主要类型与主牌 (Type-Len-MainRank) -> ID1
        # 2. 被带牌 (Kicker) -> ID2 (如果有)
        
        # 或者：
        # 简化动作空间，只保留 "Type-Len-MainRank"。
        # 对于三带一、飞机带翅膀，"自动选择最小的翅膀"。
        # 这是一个很强的假设，但能极大减小动作空间，且对 AI 性能影响可能有限（初级阶段）。
        # 只有在残局需要保牌时会有影响。
        # 我们可以接受这个妥协。
        
        # 修正后的动作ID定义 (Abstract Action Space):
        # Pass: 1
        # Single: 13 (3..2)
        # Pair: 13
        # Triple: 13
        # Triple+Single: 13 (只指定三张点数，单张自动选最小)
        # Bomb: 13 (3..2, A, 2? No bomb 2/A) -> 10 (3..Q)
        # Straight:
        #   Len 5: 3-7 .. 10-A (8)
        #   Len 6: 3-8 .. 9-A (7)
        #   ...
        #   Len 12: 3-A (1)
        #   Total ~ 40
        # DoubleSeq:
        #   Len 2: 3344 .. KKAA (10)
        #   ...
        #   Len 10: (1)
        #   Total ~ 40
        # Airplane:
        #   Len 2: 333444 .. KKKAAA (10)
        #   ...
        # Airplane+Wings:
        #   Same as Airplane. Wings auto-selected.
        
        # 总大小估计：
        # 1 + 13*4 + 10 + 40 + 40 + 20 + 20 ≈ 200 左右！
        # 这个空间非常小，非常适合 PPO 训练！
        # 唯一的代价是：不能指定被带的牌（翅膀），总是带最小的单牌。
        # 这是一个非常好的起点。
        
        print("Building Abstract Action Space...")
        
        self._add_action_abstract(HandType.PASS, 0, 0)
        
        # Single, Pair, Triple, Triple+Single, Bomb
        # Ranks: 3(3)..2(15)
        # Bomb: 3..Q (12)
        for r in range(3, 16):
            # Single
            self._add_action_abstract(HandType.SINGLE, 1, r)
            # Pair
            self._add_action_abstract(HandType.PAIR, 1, r)
            # Triple
            self._add_action_abstract(HandType.TRIPLE, 1, r)
            # Triple+Single
            self._add_action_abstract(HandType.TRIPLE_WITH_SINGLE, 1, r)
            # Bomb (only 3-12)
            if r <= 12:
                self._add_action_abstract(HandType.BOMB, 1, r)
        
        # Straight (Len 5-12, MaxRank)
        # MaxRank range: 3+Len-1 .. 14(A)
        # Min MaxRank = 3+5-1 = 7. Max MaxRank = 14.
        for length in range(5, 13):
            min_max_r = 3 + length - 1
            max_max_r = 14
            for r in range(min_max_r, max_max_r + 1):
                self._add_action_abstract(HandType.STRAIGHT, length, r)
                
        # DoubleSequence (Len 2-10)
        # Len 2: 3344 (Max 4) .. KKAA (Max 14)
        for length in range(2, 11):
            min_max_r = 3 + length - 1
            max_max_r = 14
            for r in range(min_max_r, max_max_r + 1):
                self._add_action_abstract(HandType.DOUBLE_SEQUENCE, length, r)
                
        # Airplane (Len 2-6)
        for length in range(2, 7):
            min_max_r = 3 + length - 1
            max_max_r = 14
            for r in range(min_max_r, max_max_r + 1):
                self._add_action_abstract(HandType.AIRPLANE, length, r)
                
        # Airplane+Wings (Len 2-6)
        for length in range(2, 7):
            min_max_r = 3 + length - 1
            max_max_r = 14
            for r in range(min_max_r, max_max_r + 1):
                self._add_action_abstract(HandType.AIRPLANE_WITH_WINGS, length, r)
                
        self.size = idx
        self.action_to_id = self.action_to_id
        self.id_to_action = self.id_to_action
        
        # 更新 size
        self.size = len(self.action_to_id)

    def _add_action_abstract(self, type: HandType, length: int, max_rank: int):
        # Key: Type_Length_MaxRank
        key = f"{type.name}_{length}_{max_rank}"
        if key not in self.action_to_id:
            idx = len(self.action_to_id)
            self.action_to_id[key] = idx
            # Action object here is a template (no cards)
            self.id_to_action[idx] = Play(type, [], length, max_rank)

    def save(self):
        """保存到文件"""
        path = os.path.join(os.path.dirname(__file__), 'action_space.pkl')
        with open(path, 'wb') as f:
            pickle.dump((self.action_to_id, self.id_to_action), f)

    def load(self) -> bool:
        """从文件加载"""
        path = os.path.join(os.path.dirname(__file__), 'action_space.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    self.action_to_id, self.id_to_action = pickle.load(f)
                # 更新 pass_action
                self.pass_action = self.id_to_action[0]
                self.size = len(self.action_to_id)
                return True
            except:
                return False
        return False
