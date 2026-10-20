"""游戏状态机（不可变）+ 计分。"""

import random
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from . import cards as C
from . import moves as M
from .config import DEFAULT_RULES, RuleConfig
from .rules import Move, MoveType


@dataclass(frozen=True)
class GameState:
    hands: Tuple[int, int, int]
    played: int = 0
    current: int = 0
    last_play: Optional[Move] = None
    last_player: int = 0
    tricks: Tuple[int, int, int] = (0, 0, 0)     # 出牌手数（春天判定）
    bombs: Tuple[int, int, int] = (0, 0, 0)      # 炸弹分（实时结算）
    over: bool = False
    winner: int = -1
    scores: Tuple[int, int, int] = (0, 0, 0)

    # ------------------------------------------------------------ 构造

    @classmethod
    def new(cls, seed: Optional[int] = None) -> "GameState":
        rng = random.Random(seed)
        deck = C.deck_list()
        rng.shuffle(deck)
        hands = []
        for p in range(3):
            cnt = [0] * C.NUM_RANKS
            for idx in deck[p * 15:(p + 1) * 15]:
                cnt[idx] += 1
            hands.append(C.make(cnt))
        cur = rng.randrange(3)
        return cls(hands=tuple(hands), current=cur, last_player=cur)

    @classmethod
    def from_hands(cls, hands: Sequence, current: int = 0,
                   last_play: Optional[Move] = None,
                   last_player: Optional[int] = None,
                   tricks: Sequence[int] = (0, 0, 0)) -> "GameState":
        hs = []
        for h in hands:
            if isinstance(h, int):
                hs.append(h)
            elif isinstance(h, str):
                hs.append(C.parse(h))
            else:
                hs.append(C.make(h))
        hs = tuple(hs)
        lp = current if last_player is None else last_player
        return cls(hands=hs, current=current, last_play=last_play,
                   last_player=lp, tricks=tuple(tricks))

    # ------------------------------------------------------------ 查询

    def hand(self, p: Optional[int] = None) -> int:
        return self.hands[self.current if p is None else p]

    def legal_moves(self, cfg: RuleConfig = DEFAULT_RULES) -> List[Move]:
        nxt = C.total(self.hands[(self.current + 1) % 3])
        return M.legal_moves(self.hands[self.current], self.last_play, nxt, cfg)

    def remaining_cards(self) -> Tuple[int, int, int]:
        return tuple(C.total(h) for h in self.hands)

    # ------------------------------------------------------------ 转移

    def apply(self, move: Move, cfg: RuleConfig = DEFAULT_RULES,
              validate: bool = False) -> "GameState":
        if self.over:
            raise ValueError("game is over")
        if move.type == MoveType.PASS:
            if self.last_play is None:
                raise ValueError("cannot pass when leading")
            nxt = (self.current + 1) % 3
            if nxt == self.last_player:
                return replace(self, current=nxt, last_play=None)
            return replace(self, current=nxt)

        p = self.current
        if not C.subset(move.cards, self.hands[p]):
            raise ValueError(f"cards not in hand: {move}")
        if validate and not any(x == move for x in self.legal_moves(cfg)):
            raise ValueError(f"illegal move: {move}")

        hand = self.hands[p] - move.cards
        played = self.played + move.cards
        tricks = list(self.tricks)
        tricks[p] += 1
        bombs = list(self.bombs)

        if move.type == MoveType.BOMB:
            if self.last_play is not None and self.last_play.type == MoveType.BOMB:
                prev = self.last_player
                bombs[prev] -= cfg.bomb_win
                for i in range(3):
                    if i != prev:
                        bombs[i] += cfg.bomb_lose
            bombs[p] += cfg.bomb_win
            for i in range(3):
                if i != p:
                    bombs[i] -= cfg.bomb_lose

        hands = list(self.hands)
        hands[p] = hand

        if hand == 0:
            scores = self._final_scores(hands, tricks, bombs, p, cfg)
            return replace(self, hands=tuple(hands), played=played,
                           tricks=tuple(tricks), bombs=tuple(bombs),
                           over=True, winner=p, scores=tuple(scores),
                           last_play=move, last_player=p)

        return replace(self, hands=tuple(hands), played=played,
                       tricks=tuple(tricks), bombs=tuple(bombs),
                       current=(p + 1) % 3, last_play=move, last_player=p)

    # ------------------------------------------------------------ 计分

    @staticmethod
    def _final_scores(hands, tricks, bombs, winner: int,
                      cfg: RuleConfig) -> List[int]:
        base = [0, 0, 0]
        gain = 0
        for i in range(3):
            if i == winner:
                continue
            remain = C.total(hands[i])
            loss = 0 if (cfg.protect_one_card and remain == 1) else remain
            if tricks[i] == 0:
                loss += cfg.spring_score
            base[i] -= loss
            gain += loss
        base[winner] = gain
        return [base[i] + bombs[i] for i in range(3)]

    # ------------------------------------------------------------ 调试

    def to_str(self) -> str:
        rows = []
        for p in range(3):
            mark = "*" if p == self.current else " "
            rows.append(f"{mark}P{p}[{C.total(self.hands[p]):2d}] {C.to_str(self.hands[p])}")
        rows.append(f"  last={self.last_play} by P{self.last_player}"
                    f"  scores={self.scores} winner={self.winner}")
        return "\n".join(rows)
