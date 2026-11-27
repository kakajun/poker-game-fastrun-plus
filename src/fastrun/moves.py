"""合法动作生成（生成式，不依赖固定动作表）。

必压规则的效果：存在任何能管住上家的动作时，PASS 不是合法动作。
顶大规则：下家报单（剩 1 张）时，所有单张动作被限制为手中最大点数的单张。
"""

from typing import List, Optional

from . import cards as C
from .config import DEFAULT_RULES, RuleConfig
from .rules import (Move, MoveType, PASS, can_beat)

RM = C.RANK_MIN


def _nib(idx: int, k: int) -> int:
    return k << (4 * idx)


# ---------------------------------------------------------------- 单牌型生成

def _singles(hand: int) -> List[Move]:
    return [Move(MoveType.SINGLE, 1, i + RM, _nib(i, 1))
            for i in range(C.NUM_RANKS) if C.n_at(hand, i) >= 1]


def _pairs(hand: int) -> List[Move]:
    return [Move(MoveType.PAIR, 1, i + RM, _nib(i, 2))
            for i in range(C.NUM_RANKS) if C.n_at(hand, i) >= 2]


def _triples(hand: int) -> List[Move]:
    return [Move(MoveType.TRIPLE, 1, i + RM, _nib(i, 3))
            for i in range(C.NUM_RANKS) if C.n_at(hand, i) >= 3]


def _bombs(hand: int) -> List[Move]:
    return [Move(MoveType.BOMB, 1, i + RM, _nib(i, 4))
            for i in range(C.NUM_RANKS) if C.n_at(hand, i) == 4]


def _tws(hand: int, min_rank: int = 0, cfg: RuleConfig = DEFAULT_RULES) -> List[Move]:
    """三带一。"""
    cnt = C.counts(hand)
    out: List[Move] = []
    for i in range(C.NUM_RANKS):
        if cnt[i] < 3 or i + RM <= min_rank:
            continue
        for j in range(C.NUM_RANKS):
            if j == i or cnt[j] < 1:
                continue
            out.append(Move(MoveType.TRIPLE_WITH_SINGLE, 1, i + RM,
                            _nib(i, 3) + _nib(j, 1)))
    return out


def _twt(hand: int, min_rank: int = 0, cfg: RuleConfig = DEFAULT_RULES) -> List[Move]:
    """三带二（任意两张，可成对也可两张单）。"""
    cnt = C.counts(hand)
    out: List[Move] = []
    for i in range(C.NUM_RANKS):
        if cnt[i] < 3 or i + RM <= min_rank:
            continue
        for j in range(C.NUM_RANKS):
            if j == i or cnt[j] < 1:
                continue
            for k in range(j, C.NUM_RANKS):
                if k == i:
                    continue
                if j == k:
                    if cnt[j] >= 2:
                        out.append(Move(MoveType.TRIPLE_WITH_TWO, 1, i + RM,
                                        _nib(i, 3) + _nib(j, 2)))
                elif cnt[k] >= 1:
                    out.append(Move(MoveType.TRIPLE_WITH_TWO, 1, i + RM,
                                    _nib(i, 3) + _nib(j, 1) + _nib(k, 1)))
    return out


def _straights(hand: int, length: Optional[int] = None) -> List[Move]:
    cnt = C.counts(hand)
    out: List[Move] = []
    lengths = [length] if length else range(5, 13)
    for L in lengths:
        if L < 5 or L > 12:
            continue
        for s in range(0, C.MAX_HAND_IDX + 2 - L):
            if all(cnt[s + t] >= 1 for t in range(L)):
                cards = 0
                for t in range(L):
                    cards += _nib(s + t, 1)
                out.append(Move(MoveType.STRAIGHT, L, s + L - 1 + RM, cards))
    return out


def _dseqs(hand: int, min_len: int = 3, length: Optional[int] = None) -> List[Move]:
    cnt = C.counts(hand)
    out: List[Move] = []
    lengths = [length] if length else range(min_len, 13)
    for L in lengths:
        if L < min_len or L > 12:
            continue
        for s in range(0, C.MAX_HAND_IDX + 2 - L):
            if all(cnt[s + t] >= 2 for t in range(L)):
                cards = 0
                for t in range(L):
                    cards += _nib(s + t, 2)
                out.append(Move(MoveType.DOUBLE_SEQ, L, s + L - 1 + RM, cards))
    return out


def _airplanes(hand: int, length: Optional[int] = None) -> List[Move]:
    cnt = C.counts(hand)
    out: List[Move] = []
    lengths = [length] if length else range(2, 13)
    for L in lengths:
        if L < 2 or L > 12:
            continue
        for s in range(0, C.MAX_HAND_IDX + 2 - L):
            if all(cnt[s + t] >= 3 for t in range(L)):
                cards = 0
                for t in range(L):
                    cards += _nib(s + t, 3)
                out.append(Move(MoveType.AIRPLANE, L, s + L - 1 + RM, cards))
    return out


# ---------------------------------------------------------------- 翅膀枚举

def _wings_single(avail, n: int, allow_dup: bool) -> List[int]:
    """从 avail 里取 n 张单牌（打包），允许/禁止同点数多张。"""
    out: List[int] = []

    def rec(i: int, left: int, packed: int):
        if left == 0:
            out.append(packed)
            return
        if i >= len(avail):
            return
        cap = avail[i] if allow_dup else min(avail[i], 1)
        for k in range(min(cap, left), -1, -1):
            rec(i + 1, left - k, packed + _nib(i, k))

    rec(0, n, 0)
    return out


def _wings_pairs(avail, n: int) -> List[int]:
    """从 avail 里取 n 个不同点数的对子。"""
    from itertools import combinations
    ranks = [i for i, a in enumerate(avail) if a >= 2]
    out: List[int] = []
    for combo in combinations(ranks, n):
        packed = 0
        for i in combo:
            packed += _nib(i, 2)
        out.append(packed)
    return out


def _apws(hand: int, cfg: RuleConfig = DEFAULT_RULES,
          length: Optional[int] = None, min_rank: int = 0) -> List[Move]:
    cnt = C.counts(hand)
    out: List[Move] = []
    lengths = [length] if length else range(2, 13)
    for L in lengths:
        if L < 2 or L > 12:
            continue
        for s in range(0, C.MAX_HAND_IDX + 2 - L):
            if any(cnt[s + t] < 3 for t in range(L)):
                continue
            max_r = s + L - 1 + RM
            if max_r <= min_rank:
                continue
            core = 0
            avail = list(cnt)
            for t in range(L):
                core += _nib(s + t, 3)
                avail[s + t] = 0          # 主体同点数的余牌不 wings（与旧实现一致）
            for w in _wings_single(avail, L, cfg.apw_single_dup):
                out.append(Move(MoveType.AIRPLANE_WITH_WINGS, L, max_r, core + w))
            if cfg.airplane_pair_wings:
                for w in _wings_pairs(avail, L):
                    out.append(Move(MoveType.AIRPLANE_WITH_WINGS, L, max_r, core + w))
    return out


# ---------------------------------------------------------------- 对外

def gen_lead(hand: int, cfg: RuleConfig = DEFAULT_RULES) -> List[Move]:
    """自由出牌时的全部动作（不含 PASS）。"""
    ms: List[Move] = []
    ms += _singles(hand)
    ms += _pairs(hand)
    if cfg.triple_can_lead:
        ms += _triples(hand)
    ms += _bombs(hand)
    ms += _tws(hand, 0, cfg)
    ms += _twt(hand, 0, cfg)
    ms += _straights(hand)
    ms += _dseqs(hand, cfg.min_pair_seq)
    ms += _airplanes(hand)
    ms += _apws(hand, cfg)
    return ms


def gen_beats(hand: int, target: Move,
              cfg: RuleConfig = DEFAULT_RULES) -> List[Move]:
    """能管住 target 的全部动作（不含 PASS、不含顶大过滤）。"""
    t = target.type
    ms: List[Move] = []

    if t == MoveType.SINGLE:
        ms = [m for m in _singles(hand) if m.max_rank > target.max_rank]
    elif t == MoveType.PAIR:
        ms = [m for m in _pairs(hand) if m.max_rank > target.max_rank]
    elif t == MoveType.TRIPLE:
        ms = [m for m in _triples(hand) if m.max_rank > target.max_rank]
    elif t == MoveType.TRIPLE_WITH_SINGLE:
        ms = _tws(hand, target.max_rank, cfg)
    elif t == MoveType.TRIPLE_WITH_TWO:
        ms = _twt(hand, target.max_rank, cfg)
    elif t == MoveType.STRAIGHT:
        ms = _straights(hand, target.length)
    elif t == MoveType.DOUBLE_SEQ:
        ms = _dseqs(hand, cfg.min_pair_seq, target.length)
    elif t == MoveType.AIRPLANE:
        ms = _airplanes(hand, target.length)
    elif t == MoveType.AIRPLANE_WITH_WINGS:
        ms = _apws(hand, cfg, target.length)
    elif t == MoveType.BOMB:
        ms = [m for m in _bombs(hand) if m.max_rank > target.max_rank]

    if t != MoveType.BOMB and cfg.must_beat_bomb:
        ms += _bombs(hand)

    # 兜底过滤，保证返回值一定真的能管住
    return [m for m in ms if can_beat(m, target)]


def legal_moves(hand: int, target: Optional[Move],
                next_player_cards: int,
                cfg: RuleConfig = DEFAULT_RULES) -> List[Move]:
    """完整合法动作：含 PASS 判定与顶大过滤。"""
    if target is None:
        ms = gen_lead(hand, cfg)
    else:
        ms = gen_beats(hand, target, cfg)

    if cfg.top_big_single and next_player_cards == 1 and hand:
        top_idx = C.max_idx(hand)
        top_rank = top_idx + RM if top_idx >= 0 else -1
        ms = [m for m in ms
              if m.type != MoveType.SINGLE or m.max_rank == top_rank]

    if target is not None and not ms:
        return [PASS]
    return ms
