"""AI 服务（M2 策略：拆牌进度 + 确定性保权 + 残局精确解）。

替换旧的 VAD-CFR：
  * 旧模型与当前动作/观测维度不兼容，加载失败会被当成成功（历史坑）；
  * 现在的策略是纯规则 + 轻量搜索，决策 1~4ms，行为可解释。
"""

import os
from typing import Dict, List, Optional, Tuple

from ..arena.numpy_agent import NumpyPolicyAgent
from ..eval.decompose import min_steps
from ..eval.policy_m1 import M1Policy
from ..fastrun.config import DEFAULT_RULES, RuleConfig
from ..fastrun.rules import Move
from ..fastrun.state import GameState
from .fast_service import describe, play_to_model
from .models import AIActionThinkingModel, AIThinkingModel

MODELS = ["m2-rule", "m2-search"]
POLICY_NAME = "policy-v4"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
# 纯 numpy 权重：运行镜像不需要 torch（.pt 只用于开发机导出/训练）
DEFAULT_POLICY_PATH = os.path.join(_ROOT, "models", "policy_v4.npz")


class AIService:
    def __init__(self, cfg: RuleConfig = DEFAULT_RULES,
                 policy_path: str = DEFAULT_POLICY_PATH):
        self.cfg = cfg
        self.policy_path = policy_path
        self.current_model = MODELS[0]
        self.policy = M1Policy(cfg, endgame_cards=8, endgame_k=6,
                               guaranteed_lead=False)
        self.neural = None
        # 训练模型存在时默认启用；加载失败自动留在 m2-rule，坏模型不上线
        if os.path.exists(self.policy_path):
            self.load_model(POLICY_NAME)

    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        models = list(MODELS)
        if os.path.exists(self.policy_path):
            models.insert(0, POLICY_NAME)
        return models

    def load_model(self, name: str) -> bool:
        if name == POLICY_NAME:
            if not os.path.exists(self.policy_path):
                print(f"policy checkpoint missing: {self.policy_path}")
                return False
            try:
                agent = NumpyPolicyAgent(self.cfg, npz_path=self.policy_path)
                agent._load()  # 显式验证 metadata/维度，坏模型不进入线上
            except Exception as exc:
                print(f"{POLICY_NAME} load failed: {exc}")
                return False
            self.current_model = name
            self.neural = agent
            self.policy = None
            return True
        if name not in MODELS:
            print(f"unknown model: {name}, keep {self.current_model}")
            return False
        self.current_model = name
        self.neural = None
        self.policy = M1Policy(self.cfg, endgame_cards=8, endgame_k=6,
                               guaranteed_lead=(name == "m2-search"))
        return True

    # ------------------------------------------------------------------

    def predict(self, state: GameState,
                return_thinking: bool = False
                ) -> Tuple[Move, Optional[AIThinkingModel]]:
        if self.neural is not None:
            move = self.neural.act(state)
        else:
            move = self.policy.act(state)
        thinking = self.build_thinking(state, move) if return_thinking else None
        return move, thinking

    # ------------------------------------------------------------------

    def build_thinking(self, state: GameState, chosen: Move) -> AIThinkingModel:
        hand = state.hand()
        if self.neural is not None:
            ms, prob, score = self.neural.evaluate(state)
            order = sorted(range(len(ms)),
                           key=lambda i: (-float(prob[i]), -float(score[i])))
            ranked = [ms[i] for i in order]
            weights = [float(prob[i]) for i in order[:5]]
            total = sum(weights) or 1.0
        else:
            ranked = self.policy.rank_moves(state)
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
