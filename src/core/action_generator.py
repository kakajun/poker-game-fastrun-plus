from typing import List, Tuple, Dict
from itertools import combinations
from src.core.card import Card, Rank
from src.core.hand_type import HandType, Play
from src.core.evaluator import HandEvaluator


class ActionGenerator:
    """合法动作生成器"""

    @staticmethod
    def get_all_actions(cards: List[Card]) -> List[Play]:
        """
        生成当前手牌所有合法的出牌动作（不含Pass）
        """
        actions = []

        # 预处理：按Rank分组
        rank_map = {}
        for c in cards:
            if c.rank not in rank_map:
                rank_map[c.rank] = []
            rank_map[c.rank].append(c)

        ranks = sorted(rank_map.keys())

        # 1. 单张
        for r in ranks:
            for c in rank_map[r]:
                actions.append(
                    Play(HandType.SINGLE, [c], length=1, max_rank=r))

        # 2. 对子
        for r in ranks:
            cs = rank_map[r]
            if len(cs) >= 2:
                for combo in combinations(cs, 2):
                    actions.append(
                        Play(HandType.PAIR, list(combo), length=1, max_rank=r))

        # 3. 三张 (不带)
        # 跑得快通常不允许三张不带，除非是最后一手
        # 这里先生成，由规则层过滤？
        # 或者我们允许三张不带（作为3带0）？
        # mainRule.md: "三张: 三张点数相同的牌（通常接不带）"
        # 假设允许。
        for r in ranks:
            cs = rank_map[r]
            if len(cs) >= 3:
                for combo in combinations(cs, 3):
                    actions.append(
                        Play(HandType.TRIPLE, list(combo), length=1, max_rank=r))

        # 4. 炸弹
        for r in ranks:
            cs = rank_map[r]
            if len(cs) == 4:
                actions.append(Play(HandType.BOMB, cs, length=1,
                               max_rank=r, is_bomb=True))

        # 5. 三带一
        # 找出所有三张
        triples = []
        for r in ranks:
            cs = rank_map[r]
            if len(cs) >= 3:
                for combo in combinations(cs, 3):
                    triples.append((r, list(combo)))

        for r_trip, trip_cards in triples:
            # 找单张（不能是三张里的牌）
            # 可以是同点数的剩余牌（如果有4张）
            # 也可以是其他点数的牌
            remain_cards = [c for c in cards if c not in trip_cards]
            
            # 5.1 生成三带一
            for c in remain_cards:
                play_cards = trip_cards + [c]
                play_cards.sort()
                actions.append(Play(HandType.TRIPLE_WITH_SINGLE,
                               play_cards, length=1, max_rank=r_trip))
            
            # 5.2 生成三带二
            if len(remain_cards) >= 2:
                for combo in combinations(remain_cards, 2):
                    play_cards = trip_cards + list(combo)
                    play_cards.sort()
                    actions.append(Play(HandType.TRIPLE_WITH_TWO,
                                   play_cards, length=1, max_rank=r_trip))

        # 6. 顺子 (>=5)
        # 寻找所有连续 Rank 序列
        # 不含2
        valid_ranks = [r for r in ranks if r != Rank.TWO]
        if len(valid_ranks) >= 5:
            # 暴力枚举起点和长度
            for i in range(len(valid_ranks)):
                for length in range(5, len(valid_ranks) - i + 1):
                    seq_ranks = valid_ranks[i: i+length]
                    # 检查是否连续
                    if seq_ranks[-1] - seq_ranks[0] == length - 1:
                        # 生成顺子组合
                        # 每个Rank取一张
                        # 这是一个笛卡尔积问题。如果某Rank有2张，组合数翻倍。
                        # 使用递归生成
                        ActionGenerator._generate_sequences(
                            seq_ranks, rank_map, [], actions, HandType.STRAIGHT
                        )

        # 7. 连对 (>=2对)
        # 不含2
        pair_ranks = [r for r in ranks if r !=
                      Rank.TWO and len(rank_map[r]) >= 2]
        if len(pair_ranks) >= 2:
            for i in range(len(pair_ranks)):
                for length in range(2, len(pair_ranks) - i + 1):  # Start from 2
                    seq_ranks = pair_ranks[i: i+length]
                    if seq_ranks[-1].value - seq_ranks[0].value == length - 1:
                        ActionGenerator._generate_sequences_pairs(
                            seq_ranks, rank_map, [], actions
                        )

        # 8. 飞机 (>=2三张)
        # 不含2
        triple_ranks = [r for r in ranks if r !=
                        Rank.TWO and len(rank_map[r]) >= 3]
        if len(triple_ranks) >= 2:
            for i in range(len(triple_ranks)):
                for length in range(2, len(triple_ranks) - i + 1):
                    seq_ranks = triple_ranks[i: i+length]
                    if seq_ranks[-1] - seq_ranks[0] == length - 1:
                        # 生成飞机（不带）
                        ActionGenerator._generate_sequences_triples(
                            seq_ranks, rank_map, [], actions, False  # No wings
                        )
                        # 生成飞机带翅膀
                        ActionGenerator._generate_sequences_triples(
                            seq_ranks, rank_map, [], actions, True, cards  # With wings
                        )

        return actions

    @staticmethod
    def _generate_sequences(ranks, rank_map, current_cards, actions, type):
        if not ranks:
            actions.append(Play(type, sorted(current_cards), length=len(
                current_cards), max_rank=current_cards[-1].rank))
            return

        r = ranks[0]
        for c in rank_map[r]:
            # 确保每张牌只用一次（虽然这里ranks是unique的，但递归需要）
            ActionGenerator._generate_sequences(
                ranks[1:], rank_map, current_cards + [c], actions, type)

    @staticmethod
    def _generate_sequences_pairs(ranks, rank_map, current_cards, actions):
        if not ranks:
            # max_rank is the rank of last pair
            # length is number of pairs (len(current_cards)/2)
            actions.append(Play(HandType.DOUBLE_SEQUENCE, sorted(current_cards), length=len(
                current_cards)//2, max_rank=current_cards[-1].rank))
            return

        r = ranks[0]
        cs = rank_map[r]
        for combo in combinations(cs, 2):
            ActionGenerator._generate_sequences_pairs(
                ranks[1:], rank_map, current_cards + list(combo), actions)

    @staticmethod
    def _generate_sequences_triples(ranks, rank_map, current_cards, actions, with_wings, full_hand=None):
        if not ranks:
            # Base triples are formed.
            # If no wings, just add.
            if not with_wings:
                actions.append(Play(HandType.AIRPLANE, sorted(current_cards), length=len(
                    current_cards)//3, max_rank=current_cards[-1].rank))
            else:
                # Add wings
                # Number of triples = len(current_cards)//3
                n = len(current_cards) // 3

                # Wings can be n singles OR n pairs
                # Find remaining cards
                used_cards_set = set(current_cards)
                remain_cards = [
                    c for c in full_hand if c not in used_cards_set]

                # Generate wings (Singles)
                if len(remain_cards) >= n:
                    for wing_combo in combinations(remain_cards, n):
                        final_cards = current_cards + list(wing_combo)
                        final_cards.sort()
                        # 注意：这里我们生成了所有可能的带单牌组合。
                        # 如果 wings 是对子（如55），这里会被当作2个单牌（5,5）。
                        # 对于规则来说，带2个单牌是合法的。
                        # 但如果规则强制要求：带对子时必须是对子？
                        # 通常跑得快规则：带n单 或 带n对。
                        # 如果带n单，随便n张。
                        # 如果带n对，必须n个对子。
                        # 我们的代码目前覆盖了“带n单”的所有情况（包括55）。
                        # 是否需要专门生成“带n对”的情况（需2n张）？
                        # 如果需要，应该单独生成。
                        actions.append(Play(
                            HandType.AIRPLANE_WITH_WINGS, final_cards, length=n, max_rank=current_cards[-1].rank))

                # Generate wings (Pairs)
                # 寻找剩下的对子
                # rank_map needs to be re-calculated or passed?
                # simpler to re-calc from remain_cards
                # ... skip for now
            return

        r = ranks[0]
        cs = rank_map[r]
        for combo in combinations(cs, 3):
            ActionGenerator._generate_sequences_triples(
                ranks[1:], rank_map, current_cards + list(combo), actions, with_wings, full_hand)

    @staticmethod
    def get_legal_actions(cards: List[Card], target: Play = None) -> List[Play]:
        """
        根据上家出牌 target，生成所有合法动作。
        如果 target 为 None，返回所有动作。
        如果 target 不为 None，返回能管住的动作 + (Pass if allowed? Engine decides Pass)
        注意：此函数只返回能管住的动作，不包含 Pass。
        """
        all_actions = ActionGenerator.get_all_actions(cards)

        if target is None:
            return all_actions

        legal_actions = []
        for action in all_actions:
            if HandEvaluator.can_beat(action, target):
                legal_actions.append(action)

        return legal_actions
