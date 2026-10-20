"""45 张跑得快牌堆的紧凑表示。

花色无关，因此一张"牌"退化为点数。整手牌用 13 个 4bit 计数打包进一个 int:
    idx = rank - 3          (0..12 -> 3..15, 15 表示 2)
    hand = sum(count[idx] << (4*idx))

这样:
  * 状态天然规范化（同点数牌互换等价），分支数与哈希表都大幅缩小；
  * 复制/哈希/相等都是 int 操作；
  * 仍可与旧 API 的 card_id 互转（id = idx*4 + instance）。
"""

from typing import Iterable, List, Sequence, Tuple

NUM_RANKS = 13
RANK_MIN = 3
RANK_MAX = 15          # 15 == 2
NIBBLE = 4

# 3..Q 各 4 张, K 3 张, A 1 张, 2 1 张
RANK_COUNTS: Tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1, 1)
DECK_SIZE = sum(RANK_COUNTS)          # 45
FULL_DECK = sum(c << (NIBBLE * i) for i, c in enumerate(RANK_COUNTS))
MAX_HAND_IDX = 11                      # A -> 顺子/连对/飞机不含 2


def idx_of_rank(rank: int) -> int:
    return rank - RANK_MIN


def rank_of_idx(idx: int) -> int:
    return idx + RANK_MIN


def n_at(hand: int, idx: int) -> int:
    """取某个点数的张数。"""
    return (hand >> (NIBBLE * idx)) & 0xF


def add(hand: int, idx: int, k: int = 1) -> int:
    return hand + (k << (NIBBLE * idx))


def remove(hand: int, idx: int, k: int = 1) -> int:
    return hand - (k << (NIBBLE * idx))


def counts(hand: int) -> Tuple[int, ...]:
    return tuple((hand >> (NIBBLE * i)) & 0xF for i in range(NUM_RANKS))


def make(counts_seq: Sequence[int]) -> int:
    h = 0
    for i, c in enumerate(counts_seq):
        if c:
            h += int(c) << (NIBBLE * i)
    return h


def hand_from_counts(counts_seq: Sequence[int]) -> int:
    return make(counts_seq)


def total(hand: int) -> int:
    s = 0
    h = hand
    while h:
        s += h & 0xF
        h >>= NIBBLE
    return s


def has(hand: int, idx: int, k: int = 1) -> bool:
    return n_at(hand, idx) >= k


def subset(small: int, big: int) -> bool:
    """small 的每个点数张数都不超过 big。"""
    for i in range(NUM_RANKS):
        if n_at(small, i) > n_at(big, i):
            return False
    return True


def max_idx(hand: int) -> int:
    for i in range(NUM_RANKS - 1, -1, -1):
        if n_at(hand, i):
            return i
    return -1


def min_idx(hand: int) -> int:
    for i in range(NUM_RANKS):
        if n_at(hand, i):
            return i
    return -1


def hand_from_card_ids(card_ids: Iterable[int]) -> int:
    """旧 API 的 card_id -> 打包手牌。id = idx*4 + instance。"""
    seq = [0] * NUM_RANKS
    for cid in card_ids:
        cid = int(cid)
        idx = cid // 4
        if not (0 <= idx < NUM_RANKS):
            raise ValueError(f"card id out of range: {cid}")
        seq[idx] += 1
        if seq[idx] > RANK_COUNTS[idx]:
            raise ValueError(f"too many cards of rank {rank_of_idx(idx)}")
    return make(seq)


def card_ids(hand: int) -> List[int]:
    """打包手牌 -> card_id 列表（同点数按 instance 0,1,2.. 依次分配）。"""
    out: List[int] = []
    for i in range(NUM_RANKS):
        for inst in range(n_at(hand, i)):
            out.append(i * 4 + inst)
    return out


def deck_list() -> List[int]:
    """45 张牌按点数展开（每个元素是 idx）。"""
    out: List[int] = []
    for i, c in enumerate(RANK_COUNTS):
        out.extend([i] * c)
    return out


def parse(spec: str) -> int:
    """'3*4 5*1 14*1' -> 打包手牌（调试/测试/脚本用）。"""
    seq = [0] * NUM_RANKS
    for part in spec.split():
        r, k = part.split("*")
        seq[int(r) - RANK_MIN] = int(k)
    return make(seq)


def to_str(hand: int) -> str:
    names = {11: "J", 12: "Q", 13: "K", 14: "A", 15: "2"}
    parts = []
    for i in range(NUM_RANKS):
        r = rank_of_idx(i)
        label = names.get(r, str(r))
        parts.extend([label] * n_at(hand, i))
    return " ".join(parts)
