from typing import List, Optional
from src.core.card import Card, Rank
from src.core.hand_type import HandType, Play


class HandEvaluator:
    """牌型评估器"""

    @staticmethod
    def evaluate(cards: List[Card]) -> Optional[Play]:
        """
        评估一组牌的类型。
        如果合法，返回 Play 对象；否则返回 None。
        """
        if not cards:
            return None

        # 排序：rank 优先
        cards.sort()

        # 根据张数快速分流
        count = len(cards)

        if count == 1:
            return HandEvaluator._check_single(cards)
        elif count == 2:
            return HandEvaluator._check_pair(cards)
        elif count == 3:
            return HandEvaluator._check_triple(cards)
        elif count == 4:
            # 可能是炸弹、三带一、或连对(2对)
            play = HandEvaluator._check_bomb(cards)
            if play: return play
            play = HandEvaluator._check_triple_with_single(cards)
            if play: return play
            play = HandEvaluator._check_double_sequence(cards)
            if play: return play
        elif count == 5:
            # 顺子 或 三带二
            play = HandEvaluator._check_straight(cards)
            if play:
                return play
            return HandEvaluator._check_triple_with_two(cards)

        # 6张及以上: 顺子, 连对, 飞机, 飞机带翅膀
        play = HandEvaluator._check_straight(cards)
        if play:
            return play

        play = HandEvaluator._check_double_sequence(cards)
        if play:
            return play

        play = HandEvaluator._check_airplane(cards)
        if play:
            return play

        play = HandEvaluator._check_airplane_with_wings(cards)
        if play:
            return play

        return None

    @staticmethod
    def can_beat(current: Play, target: Play) -> bool:
        """
        判断 current 是否能管住 target。
        假设 target 是上家出的牌，current 是我方出的牌。
        """
        if not current or not target:
            return False

        # 炸弹判断
        if current.type == HandType.BOMB:
            if target.type != HandType.BOMB:
                return True
            else:
                return current.max_rank > target.max_rank

        if target.type == HandType.BOMB:
            return False

        # 非炸弹必须同类型且同张数
        if current.type != target.type:
            return False

        if current.length != target.length:
            # 特殊情况：三带一 vs 三带一对？通常不行
            # 飞机带翅膀 vs 飞机带翅膀（对子）：长度不同不行
            return False

        # 同类型同张数，比大小
        return current.max_rank > target.max_rank

    @staticmethod
    def _check_single(cards: List[Card]) -> Play:
        return Play(HandType.SINGLE, cards, length=1, max_rank=cards[0].rank)

    @staticmethod
    def _check_pair(cards: List[Card]) -> Optional[Play]:
        if cards[0].rank == cards[1].rank:
            return Play(HandType.PAIR, cards, length=1, max_rank=cards[0].rank)
        return None

    @staticmethod
    def _check_triple(cards: List[Card]) -> Optional[Play]:
        if cards[0].rank == cards[1].rank == cards[2].rank:
            return Play(HandType.TRIPLE, cards, length=1, max_rank=cards[0].rank)
        return None

    @staticmethod
    def _check_bomb(cards: List[Card]) -> Optional[Play]:
        if len(cards) != 4:
            return None
        if cards[0].rank == cards[1].rank == cards[2].rank == cards[3].rank:
            return Play(HandType.BOMB, cards, length=1, max_rank=cards[0].rank, is_bomb=True)
        return None

    @staticmethod
    def _check_triple_with_single(cards: List[Card]) -> Optional[Play]:
        # 3带1 (4张)
        rank_counts = {}
        for c in cards:
            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

        if len(rank_counts) != 2:
            return None

        triple_rank = None
        single_rank = None
        for r, count in rank_counts.items():
            if count == 3:
                triple_rank = r
            elif count == 1:
                single_rank = r

        if triple_rank and single_rank:
            return Play(HandType.TRIPLE_WITH_SINGLE, cards, length=1, max_rank=triple_rank)
        return None

    @staticmethod
    def _check_triple_with_two(cards: List[Card]) -> Optional[Play]:
        # 3带2 (5张)
        rank_counts = {}
        for c in cards:
            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

        # 必须是两组或三组Rank (例如 333+4+5 或 333+44)
        if not (2 <= len(rank_counts) <= 3):
            return None

        triple_rank = None
        for r, count in rank_counts.items():
            if count >= 3:
                triple_rank = r
                break

        if triple_rank:
            return Play(HandType.TRIPLE_WITH_TWO, cards, length=1, max_rank=triple_rank)
        return None

    @staticmethod
    def _check_straight(cards: List[Card]) -> Optional[Play]:
        # 顺子: >=5 张, 连续, 不能含2
        if len(cards) < 5:
            return None

        # 检查是否包含2
        for c in cards:
            if c.rank == Rank.TWO:
                return None

        # 检查连续性
        for i in range(len(cards) - 1):
            if cards[i+1].rank - cards[i].rank != 1:
                return None

        return Play(HandType.STRAIGHT, cards, length=len(cards), max_rank=cards[-1].rank)

    @staticmethod
    def _check_double_sequence(cards: List[Card]) -> Optional[Play]:
        # 连对: 2对及以上连续对子 (4张, 6张, 8张...)
        if len(cards) < 4 or len(cards) % 2 != 0:
            return None

        pairs = []
        for i in range(0, len(cards), 2):
            if cards[i].rank != cards[i+1].rank:
                return None
            if cards[i].rank == Rank.TWO:
                return None
            pairs.append(cards[i].rank)

        # 检查对子连续性
        for i in range(len(pairs) - 1):
            if pairs[i+1] - pairs[i] != 1:
                return None

        return Play(HandType.DOUBLE_SEQUENCE, cards, length=len(pairs), max_rank=pairs[-1])

    @staticmethod
    def _check_airplane(cards: List[Card]) -> Optional[Play]:
        # 飞机: >= 2个三张 (6张, 9张...)
        if len(cards) < 6 or len(cards) % 3 != 0:
            return None

        triples = []
        for i in range(0, len(cards), 3):
            if not (cards[i].rank == cards[i+1].rank == cards[i+2].rank):
                return None
            if cards[i].rank == Rank.TWO:
                return None
            triples.append(cards[i].rank)

        # 检查连续性
        for i in range(len(triples) - 1):
            if triples[i+1] - triples[i] != 1:
                return None

        return Play(HandType.AIRPLANE, cards, length=len(triples), max_rank=triples[-1])

    @staticmethod
    def _check_airplane_with_wings(cards: List[Card]) -> Optional[Play]:
        # 飞机带翅膀: n个三张 + n个单牌 (4*n 张) 或 n个对子 (5*n 张)
        count = len(cards)

        # Case 1: 带单牌 (n * 4)
        if count % 4 == 0:
            n = count // 4
            if n >= 2:
                play = HandEvaluator._find_airplane_core(cards, n)
                if play:
                    return play

        # Case 2: 带对子 (n * 5)
        if count % 5 == 0:
            n = count // 5
            if n >= 2:
                # 必须检查翅膀是否为对子
                # 1. 找到三张主体
                # 2. 剩下的牌是否组成 n 个对子
                # 这个逻辑比较复杂，因为三张主体也包含对子 (333 包含 33)
                # 简单做法: 统计 rank counts, 必须有 n 个 >=3, 且剩下的能组成对子

                # 重新实现带对子的检测逻辑
                play = HandEvaluator._find_airplane_core_with_pairs(cards, n)
                if play:
                    return play

        return None

    @staticmethod
    def _find_airplane_core(cards: List[Card], n: int) -> Optional[Play]:
        """寻找 n 连三张主体 (翅膀随意)"""
        rank_counts = {}
        for c in cards:
            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

        triple_candidates = [
            r for r, count in rank_counts.items() if count >= 3]
        if len(triple_candidates) < n:
            return None

        triple_candidates.sort()

        # 寻找连续序列
        for i in range(len(triple_candidates) - n + 1):
            seq = triple_candidates[i: i+n]
            if seq[-1] == Rank.TWO:
                continue

            is_seq = True
            for k in range(len(seq) - 1):
                if seq[k+1] - seq[k] != 1:
                    is_seq = False
                    break

            if is_seq:
                return Play(HandType.AIRPLANE_WITH_WINGS, cards, length=n, max_rank=seq[-1])
        return None

    @staticmethod
    def _find_airplane_core_with_pairs(cards: List[Card], n: int) -> Optional[Play]:
        """寻找 n 连三张主体，且翅膀必须是 n 个对子"""
        rank_counts = {}
        for c in cards:
            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

        # 所有牌必须至少是2张 (对子)
        # 因为翅膀是对子，主体是三张(也是对子+1)
        # 所以所有 rank 的 count 必须 >= 2
        for r, c in rank_counts.items():
            if c < 2:
                return None

        # 寻找主体
        triple_candidates = [
            r for r, count in rank_counts.items() if count >= 3]
        if len(triple_candidates) < n:
            return None

        triple_candidates.sort()

        # 寻找连续序列
        for i in range(len(triple_candidates) - n + 1):
            seq = triple_candidates[i: i+n]
            if seq[-1] == Rank.TWO:
                continue

            is_seq = True
            for k in range(len(seq) - 1):
                if seq[k+1] - seq[k] != 1:
                    is_seq = False
                    break

            if is_seq:
                # 找到了主体，现在检查剩下的牌是否全是对子
                # 移除主体 (n * 3 张)
                # 实际上只要所有 count 都是 2, 3, 4 (2对), 5(3+2), 6(3+3) 等
                # 且总张数匹配即可
                # 我们已经检查了所有 count >= 2
                # 主体消耗了 3*n 张
                # 剩下 2*n 张
                # 因为所有 count >= 2, 且总数正确,
                # 剩下的必然能组成对子?
                # 例如 333444 + 5566. 3(3), 4(3), 5(2), 6(2).
                # 移除 333, 444. 剩 55, 66. 是对子.
                # 例如 33334444 + 5555. (12张) n=2? 12 != 10.
                # 例如 333444 + 5555. (10张). 3(3), 4(3), 5(4).
                # 移除 333, 444. 剩 5555 (2对).
                # 是合法的.

                return Play(HandType.AIRPLANE_WITH_WINGS, cards, length=n, max_rank=seq[-1])

        return None
