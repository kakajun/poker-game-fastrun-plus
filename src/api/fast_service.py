"""API ↔ fastrun 适配层（M5）。

要点:
  * 会话状态换成 `fastrun.GameState`（规则已按口径修正，含顶大/必炸/连对>=3）；
  * 对外 DTO 完全不变（CardModel/PlayModel/GameStateModel），前端零改动；
  * 牌 id 仍是 `idx*4 + instance`，与旧 `Card.from_id` 一致；
  * 只给人类玩家返回 legal_actions，避免把 AI 的可选牌型泄漏给前端/玩家。
"""

import uuid
from typing import Dict, List, Optional, Tuple

from ..eval.policy_m1 import M1Policy
from ..fastrun import cards as C
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move, MoveType, classify
from ..fastrun.state import GameState
from .models import (AIActionThinkingModel, AIThinkingModel, CardModel,
                     GameStateModel, PlayModel)

# 旧 HandType 的命名（前端/历史日志兼容）
TYPE_NAMES = {
    MoveType.PASS: "PASS",
    MoveType.SINGLE: "SINGLE",
    MoveType.PAIR: "PAIR",
    MoveType.TRIPLE: "TRIPLE",
    MoveType.TRIPLE_WITH_SINGLE: "TRIPLE_WITH_SINGLE",
    MoveType.TRIPLE_WITH_TWO: "TRIPLE_WITH_TWO",
    MoveType.STRAIGHT: "STRAIGHT",
    MoveType.DOUBLE_SEQ: "DOUBLE_SEQUENCE",
    MoveType.AIRPLANE: "AIRPLANE",
    MoveType.AIRPLANE_WITH_WINGS: "AIRPLANE_WITH_WINGS",
    MoveType.BOMB: "BOMB",
}

TYPE_CN = {
    MoveType.PASS: "过",
    MoveType.SINGLE: "单张",
    MoveType.PAIR: "对子",
    MoveType.TRIPLE: "三张",
    MoveType.TRIPLE_WITH_SINGLE: "三带一",
    MoveType.TRIPLE_WITH_TWO: "三带二",
    MoveType.STRAIGHT: "顺子",
    MoveType.DOUBLE_SEQ: "连对",
    MoveType.AIRPLANE: "飞机",
    MoveType.AIRPLANE_WITH_WINGS: "飞机带翅膀",
    MoveType.BOMB: "炸弹",
}

RANK_LABELS = {11: "J", 12: "Q", 13: "K", 14: "A", 15: "2"}


def rank_of_card_id(cid: int) -> int:
    return int(cid) // 4 + 3


def rank_label(rank: int) -> str:
    return RANK_LABELS.get(rank, str(rank))


def cards_to_models(cards_packed: int) -> List[CardModel]:
    return [CardModel(rank=rank_of_card_id(cid), id=cid)
            for cid in C.card_ids(cards_packed)]


def play_to_model(move: Move) -> PlayModel:
    return PlayModel(
        type=TYPE_NAMES.get(move.type, move.type.name),
        cards=cards_to_models(move.cards),
        length=move.length,
        max_rank=move.max_rank,
    )


def describe(move: Move, steps_after: int = -1) -> str:
    if move.is_pass:
        return "过"
    cards = " ".join(rank_label(rank_of_card_id(cid)) for cid in C.card_ids(move.cards))
    text = f"{TYPE_CN.get(move.type, move.type.name)} {cards}"
    if steps_after >= 0:
        text += f" · 剩{steps_after}手"
    return text


def packed_from_card_ids(card_ids: List[int]) -> int:
    return C.hand_from_card_ids([int(x) for x in card_ids])


def card_ids_for_move(hand: int, move: Move) -> List[int]:
    """从当前手牌里挑出这手牌对应的 card id（同点数任意张等价）。"""
    if move.is_pass:
        return []
    pool: Dict[int, List[int]] = {}
    for cid in C.card_ids(hand):
        pool.setdefault(cid // 4, []).append(cid)
    out: List[int] = []
    for idx in range(C.NUM_RANKS):
        k = C.n_at(move.cards, idx)
        if k:
            out.extend(pool.get(idx, [])[:k])
    return out


class FastSession:
    """一局对战：人类固定 0 号位，AI 为 1/2。"""

    def __init__(self, session_id: str, cfg: RuleConfig = DEFAULT_RULES):
        self.session_id = session_id
        self.cfg = cfg
        # 人类固定先手（前端默认 0 号位是人类，避免开局就是 AI 回合）
        self.state = GameState.new()
        self.state = GameState.from_hands(self.state.hands, current=0)
        self.human_player_idx = 0
        self.ai_player_indices = [1, 2]
        self.thinking: Optional[AIThinkingModel] = None

    # ---------------- 查询 ----------------

    def hand_models(self, p: int) -> List[CardModel]:
        return cards_to_models(self.state.hands[p])

    def legal_moves(self) -> List[Move]:
        return self.state.legal_moves(self.cfg)

    def to_dto(self, thinking: Optional[AIThinkingModel] = None) -> GameStateModel:
        st = self.state
        legal: List[PlayModel] = []
        if not st.over and st.current == self.human_player_idx:
            legal = [play_to_model(m) for m in self.legal_moves()]
        return GameStateModel(
            game_id=self.session_id,
            current_player=st.current,
            hands=[self.hand_models(p) for p in range(3)],
            last_play=play_to_model(st.last_play) if st.last_play else None,
            last_play_player=st.last_player,
            winner=st.winner if st.over else -1,
            is_over=st.over,
            scores=list(st.scores),
            cards_played_count=list(st.tricks),
            bomb_scores=list(st.bombs),
            legal_actions=legal,
            thinking=thinking,
        )

    # ---------------- 动作 ----------------

    def apply_human_cards(self, card_ids: List[int]) -> Move:
        """把前端选的牌 id 转成动作并落地；不合法抛 ValueError。"""
        st = self.state
        if st.over:
            raise ValueError("对局已结束")
        if st.current != self.human_player_idx:
            raise ValueError("还没轮到你")
        legal = self.legal_moves()

        if not card_ids:
            if not any(m.is_pass for m in legal):
                raise ValueError("能管必须管，不能过牌")
            move = legal[0]
            self.state = st.apply(move, self.cfg)
            return move

        packed = packed_from_card_ids(card_ids)
        if not C.subset(packed, st.hands[self.human_player_idx]):
            raise ValueError("牌不在手里")
        move = classify(packed, self.cfg)
        if move is None:
            raise ValueError("不是合法牌型")
        for m in legal:
            if (m.type == move.type and m.length == move.length
                    and m.max_rank == move.max_rank and m.cards == move.cards):
                self.state = st.apply(m, self.cfg)
                return m
        # 牌型对但管不住 / 顶大限制
        if st.last_play is not None and any(m.is_pass for m in legal):
            raise ValueError("这手牌管不住上家")
        raise ValueError("不合法（可能是顶大限制：下家报单时必须出最大单张）")

    def apply_ai_move(self, move: Move) -> None:
        self.state = self.state.apply(move, self.cfg)


class SessionManager:
    def __init__(self, cfg: RuleConfig = DEFAULT_RULES):
        self.cfg = cfg
        self.sessions: Dict[str, FastSession] = {}

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = FastSession(sid, self.cfg)
        return sid

    def get_session(self, sid: str) -> Optional[FastSession]:
        return self.sessions.get(sid)

    def remove_session(self, sid: str) -> None:
        self.sessions.pop(sid, None)
