"""AI 服务（M2 策略：拆牌进度 + 确定性保权 + 残局精确解）。

替换旧的 VAD-CFR：
  * 旧模型与当前动作/观测维度不兼容，加载失败会被当成成功（历史坑）；
  * 现在的策略是纯规则 + 轻量搜索，决策 1~4ms，行为可解释。
"""

from typing import Dict, List, Optional, Tuple

from ..eval.decompose import min_steps
from ..eval.policy_m1 import M1Policy
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from .fast_service import describe, play_to_model
from .models import AIActionThinkingModel, AIThinkingModel

MODELS = ["m2-rule", "m2-search"]


class AIService:
    def __init__(self, cfg: RuleConfig = DEFAULT_RULES):
        self.cfg = cfg
        self.current_model = MODELS[0]
        self.policy = M1Policy(cfg, endgame_cards=8, endgame_k=6,
                               guaranteed_lead=(self.current_model == "m2-search"))

    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        return list(MODELS)

    def load_model(self, name: str) -> bool:
        if name not in MODELS:
            print(f"unknown model: {name}, keep {self.current_model}")
            return False
        self.current_model = name
        self.policy = M1Policy(self.cfg, endgame_cards=8, endgame_k=6,
                               guaranteed_lead=(name == "m2-search"))
        return True

    # ------------------------------------------------------------------

    def predict(self, state: GameState,
                return_thinking: bool = False
                ) -> Tuple[Move, Optional[AIThinkingModel]]:
        move = self.policy.act(state)
        thinking = self.build_thinking(state, move) if return_thinking else None
        return move, thinking

    # ------------------------------------------------------------------

    def build_thinking(self, state: GameState, chosen: Move) -> AIThinkingModel:
        ranked = self.policy.rank_moves(state)
        hand = state.hand()
        weights = [1.0 / (i + 1) for i in range(min(len(ranked), 5))]
        total = sum(weights) or 1.0
        top = []
        for i, m in enumerate(ranked[:5]):
            steps_after = -1
            if not m.is_pass:
                steps_after = min_steps(hand - m.cards, self.cfg)
            top.append(AIActionThinkingModel(
                play=describe(m, steps_after),
                regret=0.0,
                prob=round(weights[i] / total, 3),
                cards=[int(c.id) for c in play_to_model(m).cards],
            ))
        chosen_steps = -1
        if not chosen.is_pass:
            chosen_steps = min_steps(hand - chosen.cards, self.cfg)
        return AIThinkingModel(
            player_idx=state.current,
            top_actions=top,
            chosen_play=describe(chosen, chosen_steps),
        )
