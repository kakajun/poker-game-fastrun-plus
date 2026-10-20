"""规则开关。默认值 = 需求方 2026-10-20 确认的口径。

改规则只改这里，不用动引擎逻辑；每次 checkpoint 都要记录 config_hash。
"""

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuleConfig:
    # --- 牌型 ---
    triple_can_lead: bool = True        # 规则1: 三张可单出
    min_pair_seq: int = 3               # 规则2: 连对最少 3 对
    airplane_pair_wings: bool = True    # 规则3: 飞机允许带对翅膀
    apw_single_dup: bool = True         # 规则3: 单翅膀允许带同点两张
    kicker_same_rank: bool = False      # 三带一/二 不能用三张自身的同点数牌

    # --- 强制规则 ---
    must_beat_bomb: bool = True         # 规则4: 炸弹也算"能管住"，能管必须管
    top_big_single: bool = True         # 规则5: 下家报单时单张必须是最大单张

    # --- 计分 ---
    bomb_win: int = 20                  # 炸弹成功: 出炸者 +20
    bomb_lose: int = 10                 # 其余每家 -10
    spring_score: int = 30              # 春天: 每个未出过牌的输家 -30
    protect_one_card: bool = True       # 保本: 剩 1 张基础分记 0

    def hash(self) -> str:
        raw = repr(sorted(asdict(self).items())).encode()
        return hashlib.sha256(raw).hexdigest()[:12]


DEFAULT_RULES = RuleConfig()
