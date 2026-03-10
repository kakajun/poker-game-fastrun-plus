# 阶段 2：AI 训练环境（Gym）实施方案（已根据实际代码更新）

## 目标与范围
- 将规则引擎封装为标准强化学习环境（Gym 接口），让智能体可与环境交互。
- 提供观测（Observation）、动作（Action）与奖励（Reward）定义，包含动作掩码（Action Mask）。

## 接口设计
- `reset(seed?) -> (obs, info)`
- `step(action_id) -> (obs, reward, terminated, truncated, info)`
- `action_mask`: 通过 `info` 或 `ActionMasker` 暴露。

## 状态表示（Observation）
- **编码方式**: `ObsEncoder` (src/env/obs_encoder.py)
- **总维度**: **116 维** (Box 空间, float32)
- **组成**:
  - **我的手牌 (52维)**: One-hot 编码，索引对应 Card ID (0-51)。
  - **其他两家剩余牌数 (2维)**: 数值归一化 (剩余张数 / 15.0)。
  - **当前桌面牌 (52维)**: 上一手打出的牌的 One-hot 编码 (Last Play Cards)。
  - **当前桌面牌型 (10维)**: 上一手牌型的 One-hot 编码 (Last Play Type, 对应 10 种 HandType)。
- **注意**: 与原计划相比，简化了对手历史出牌的记录，专注于当前盘面信息，以降低状态空间复杂度。

## 动作定义（Action）
- **动作空间类型**: `Discrete(N)` (离散空间)
- **动作管理**: `ActionSpace` (src/env/action_space.py)
- **抽象动作 (Abstract Action)**:
  - 为了避免组合爆炸，动作空间采用**抽象定义**：`Type_Length_MaxRank`。
  - 具体的牌（如花色组合、被带的翅膀牌）由环境在执行时自动选择（通常选择最小的可用牌）。
- **空间大小**: 约 **252 维**。
  - **Pass (1)**: ID 0
  - **Single (13)**: 3-2
  - **Pair (13)**: 3-2
  - **Triple (13)**: 3-2
  - **Triple+Single (13)**: 3-2 (自动带最小单)
  - **Bomb (10)**: 3-Q (无 K/A/2 炸弹)
  - **Straight (~40)**: 长度 5-12 × 合法 MaxRank
  - **DoubleSeq (~40)**: 长度 2-10 × 合法 MaxRank
  - **Airplane (~20)**: 长度 2-6 × 合法 MaxRank
  - **Airplane+Wings (~20)**: 长度 2-6 × 合法 MaxRank (自动带最小翅膀)
- **动作掩码 (Action Mask)**:
  - 环境根据当前手牌和规则（如必压、首出红桃3），计算当前合法的抽象动作子集。
  - 返回二进制向量，供 Maskable PPO 屏蔽非法动作。

## 奖励设计（Reward）
- **终局奖励**:
  - **胜利**: `+100`
  - **失败**: `-100` (原计划为 -剩余牌数，目前代码实现为非法动作 -100，输赢奖励逻辑需在 Wrapper 中确认，BaseEnv 仅返回 0 或 -100 非法)。
  - *注*: 实际训练中，`SingleAgentWrapper` 负责计算最终胜负奖励。
- **中间奖励**:
  - **非法动作**: `-100` 并结束回合 (Truncated)。
  - **炸弹**: `+20` (在 `Game` 逻辑中即时结算)。
  - **步数惩罚**: 每步 `-1` (鼓励快速出完，在 Wrapper 中实现)。
  - **出牌奖励**: 每次合法出牌 `+1` (鼓励积极出牌，在 Wrapper 中实现)。

## Baseline 智能体
- **RandomAgent**: 从 `action_mask` 允许的动作中随机采样。
- **PPO Agent**: 使用 Maskable PPO 算法进行训练。

## 关键类与文件
- `src/env/poker_env.py`: Gym 环境主入口。
- `src/env/obs_encoder.py`: 状态编码逻辑。
- `src/env/action_space.py`: 动作空间构建与 ID 映射。
- `src/env/single_agent_wrapper.py`: 单智能体包装器，处理奖励整形和对手模拟。

## 里程碑
- ✅ **M1**: 环境接口与 Gym 注册完成。
- ✅ **M2**: 抽象动作空间构建完成，有效降低了动作维度。
- ✅ **M3**: 状态编码器实现，支持 116 维向量输入。
- ✅ **M4**: 接入 Stable-Baselines3 进行 PPO 训练验证。
