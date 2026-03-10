from typing import List, Dict, Optional, Tuple
from src.core.card import Card, Rank, Suit
from src.core.deck import Deck
from src.core.hand_type import HandType, Play
from src.core.action_generator import ActionGenerator
from src.core.evaluator import HandEvaluator


class Game:
    """跑得快游戏引擎"""

    def __init__(self, seed: int = None):
        self.deck = Deck(seed)
        self.hands: List[List[Card]] = [[], [], []]
        self.current_player: int = 0

        # 桌面状态
        self.last_play: Optional[Play] = None
        self.last_play_player: int = -1

        # 统计
        self.cards_played_count: List[int] = [0, 0, 0]  # 每个玩家出牌次数 (用于春天判定)
        self.bomb_scores: List[int] = [0, 0, 0]  # 炸弹得分 (实时结算)
        self.played_card_ids: set = set()  # 已出牌 ID 集合 (用于记牌)

        # 结果
        self.winner: int = -1
        self.is_over: bool = False
        self.scores: List[int] = [0, 0, 0]  # 最终得分

        # 初始发牌
        self.start_game()

    def start_game(self):
        """开始新游戏：洗牌、发牌、确定首出"""
        self.deck.shuffle()
        h1, h2, h3 = self.deck.deal()

        # 排序手牌：按点数降序，同点数按花色 (黑>红>梅>方)
        # Rank.value: 2=15 > ... > 3=3
        # Suit.value: Spade=0, Heart=1, Club=2, Diamond=3
        # key: (rank, -suit) -> Rank大在前，Suit小在前
        def sort_key(c): return (c.rank.value, -c.suit.value)

        h1.sort(key=sort_key, reverse=True)
        h2.sort(key=sort_key, reverse=True)
        h3.sort(key=sort_key, reverse=True)

        self.hands = [h1, h2, h3]

        # 寻找红桃3
        h3_card = Card(Rank.THREE, Suit.HEART)
        start_idx = -1
        for i in range(3):
            if h3_card in self.hands[i]:
                start_idx = i
                break

        if start_idx == -1:
            # 理论上不可能，除非牌堆不对
            raise ValueError("No Heart 3 found in hands!")

        self.current_player = start_idx
        self.last_play = None
        self.last_play_player = start_idx  # 初始状态视为轮到首家任意出牌

        # 重置统计
        self.cards_played_count = [0, 0, 0]
        self.bomb_scores = [0, 0, 0]
        self.played_card_ids = set()
        self.winner = -1
        self.is_over = False
        self.scores = [0, 0, 0]

    def get_legal_actions(self) -> List[Play]:
        """获取当前玩家的合法动作"""
        if self.is_over:
            return []

        hand = self.hands[self.current_player]

        # 判断是否是新的一轮 (自由出牌)
        # 如果 last_play_player == current_player，说明其他人都过了，或者这是首轮
        is_new_round = (self.last_play_player == self.current_player) or (
            self.last_play is None)

        target = None if is_new_round else self.last_play

        # 获取动作
        if is_new_round:
            actions = ActionGenerator.get_all_actions(hand)
        else:
            actions = ActionGenerator.get_legal_actions(hand, target)
            if not actions:
                return [Play(HandType.PASS, [], length=0, max_rank=0)]

        # 特殊规则过滤
        # 1. 首出必须包含红桃3 (如果是首轮且还没出过牌)
        h3_card = Card(Rank.THREE, Suit.HEART)
        # 判断是否是首轮：所有玩家出牌次数均为0?
        # 或者简单的：如果持有红桃3，且当前是自由出牌，且还没有打出过?
        # 其实只要持有红桃3，这一手必须出红桃3 (因为红桃3必首出)
        # 只有持有者才受限。
        if h3_card in hand:
            filtered = []
            for a in actions:
                if h3_card in a.cards:
                    filtered.append(a)
            actions = filtered

        # 3. 顶大规则 (报单时上家出最大单)
        # 检查下家 (next player) 手牌数
        next_player = (self.current_player + 1) % 3
        if len(self.hands[next_player]) == 1:
            # 下家报单
            # 找出我手牌最大的单张 rank
            if hand:
                max_rank_in_hand = hand[-1].rank  # Hand is sorted

                filtered = []
                for a in actions:
                    if a.type == HandType.SINGLE:
                        # 只有最大单张才保留
                        if a.max_rank == max_rank_in_hand:
                            filtered.append(a)
                    else:
                        filtered.append(a)
                actions = filtered

        return actions

    def step(self, action: Play) -> Tuple[bool, List[str]]:
        """
        执行动作
        Returns: (is_over, events)
        """
        events = []

        # 校验合法性 (简单校验，假设调用者已调用 get_legal_actions)
        # 但为了安全，这里应该再次校验?
        # 为性能考虑，暂跳过深度校验，只做基本检查

        if action.type == HandType.PASS:
            # Pass
            if self.last_play is None:
                raise ValueError("Cannot pass on free turn")

            # 轮转
            self.current_player = (self.current_player + 1) % 3

            # 检查是否一轮结束
            if self.current_player == self.last_play_player:
                # 回到出牌者，新一轮
                self.last_play = None

            return self.is_over, events

        # 出牌
        # 从手牌移除
        current_hand = self.hands[self.current_player]
        for c in action.cards:
            if c not in current_hand:
                raise ValueError(f"Card {c} not in hand")
            current_hand.remove(c)  # List remove is O(N) but N is small
            self.played_card_ids.add(c.id)  # 记录已出的牌 ID

        self.cards_played_count[self.current_player] += 1

        # 炸弹得分
        if action.type == HandType.BOMB:
            # 立即结算炸弹分
            # 每家输10，打出者得20
            # 无论是否被管住?
            # mainRule.md: "打出炸弹且成功（未被更大的炸弹管住）：每家输10分，打出者得20分。如果炸弹被更大的炸弹管住，前一个炸弹不得分。"
            # 这意味着炸弹分是"暂存"的，直到一轮结束才能确定?
            # 或者：立即加分。如果被管住，再扣回来?
            # 简单做法：立即加分。如果被管住，倒扣前一个人的分。

            # 逻辑：
            # 当前打出炸弹 -> +20, 其他 -10.
            # 检查 last_play 是否是炸弹
            if self.last_play and self.last_play.type == HandType.BOMB:
                # 被管住了，前一个炸弹不得分 (扣回)
                # 前一个玩家 (last_play_player) -20, 其他 +10
                prev = self.last_play_player
                self.bomb_scores[prev] -= 20
                for i in range(3):
                    if i != prev:
                        self.bomb_scores[i] += 10
                events.append(f"Bomb blocked! Player {prev} score reverted.")

            # 当前炸弹得分
            curr = self.current_player
            self.bomb_scores[curr] += 20
            for i in range(3):
                if i != curr:
                    self.bomb_scores[i] -= 10
            events.append(f"Player {curr} played Bomb! Score +20.")

        # 更新桌面
        self.last_play = action
        self.last_play_player = self.current_player

        # 检查胜利
        if len(current_hand) == 0:
            self.winner = self.current_player
            self.is_over = True
            self._calculate_final_scores()
            events.append(f"Game Over! Winner: Player {self.winner}")
            return True, events

        # 轮转
        self.current_player = (self.current_player + 1) % 3

        return False, events

    def _calculate_final_scores(self):
        """计算最终得分"""
        # 基础分
        remain_counts = [len(h) for h in self.hands]
        base_scores = [0, 0, 0]

        winner_score = 0
        for i in range(3):
            if i != self.winner:
                # 输家扣分 = 剩余张数
                loss = remain_counts[i]

                # 保本规则：如果只剩1张牌，不算输，不扣分
                if loss == 1:
                    loss = 0

                # 春天 (关门) 判定
                # 如果赢家出完，且输家一张没出 (cards_played_count[i] == 0)
                # 输家输双倍?
                # mainRule.md: "赢家1个得30分... 春天家输30分"
                # 这似乎是固定分?
                # "剩余1张牌 = 1分" 是基础分。
                # "春天：如果赢家出完牌时，另外另外玩家一张牌都未出... 赢家1个得30分"
                # 这意味着春天是 *额外* 奖励? 还是替代?
                # 通常跑得快规则：春天 = 剩余张数翻倍? 或者固定奖励?
                # 规则原文:
                # "赢家得分 = 另外两家剩余牌数之和"
                # "特殊奖励分... 春天... 赢家1个得30分"
                # 这看起来是 *额外* 加分。
                # 也就是说：
                # Score = (Base Loss) + (Bomb Score) + (Spring Score)

                spring_score = 0
                if self.cards_played_count[i] == 0:
                    spring_score = 30  # 春天罚分

                total_loss = loss + spring_score
                base_scores[i] -= total_loss
                winner_score += total_loss

        base_scores[self.winner] = winner_score

        # 总分 = 基础分 + 炸弹分
        for i in range(3):
            self.scores[i] = base_scores[i] + self.bomb_scores[i]
