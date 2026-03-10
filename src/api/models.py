from pydantic import BaseModel
from typing import List, Optional, Any

class CardModel(BaseModel):
    rank: int
    suit: int
    id: int # 0-51

class PlayModel(BaseModel):
    type: str # SINGLE, PAIR, ...
    cards: List[CardModel]
    length: int
    max_rank: int

class GameStateModel(BaseModel):
    game_id: str
    current_player: int
    hands: List[List[CardModel]] # 实际上只需要返回玩家的，AI 的可以隐藏（返回背面或只返回数量）
                                 # 但为了调试，先全部返回。前端自己决定显示。
    last_play: Optional[PlayModel]
    last_play_player: int
    winner: int
    is_over: bool
    scores: List[int]
    
    # 额外信息
    cards_played_count: List[int]
    bomb_scores: List[int]
    
    # 合法动作 (仅针对当前玩家，用于辅助前端)
    # 如果仅包含 PASS，说明要不起
    legal_actions: List[PlayModel]

class ActionRequest(BaseModel):
    # 玩家选择的牌的 IDs
    card_ids: List[int] 
    # 或者直接传 PlayModel? 
    # 前端可能不知道怎么组装 PlayModel (type, max_rank)。
    # 前端只知道选了哪些牌。
    # 后端需要根据 card_ids 推断 PlayModel。
    # 如果是 Pass，card_ids 为空。

class AIRequest(BaseModel):
    pass
