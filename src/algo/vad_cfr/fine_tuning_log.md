# VAD-CFR Fine-Tuning Log

## Version 1.2 - 2026-03-13 (Current)

### 1. 架构升级 (Architectural Upgrades)

- **ResNet-style 网络结构**:
    - 将简单的 MLP 替换为包含 **Skip Connections (残差连接)** 和 **LayerNorm (层归一化)** 的 ResNet 结构。
    - 提高了深层网络的梯度流动性，增强了模型处理复杂局面（如长序列出牌）的能力。

- **状态编码增强 (Observation Enhancement)**:
    - 维度从 `57` 增加到 `60`。
    - **新增特征**:
        - `is_my_turn_to_lead`: 显式告知 AI 当前是否拥有发牌权。
        - `max_rank_in_hand`: 告知 AI 手中最大牌的点数，有助于决策是否保留大牌。
        - `opponent_reported_single`: 告知 AI 对手是否已经报单（报警），从而强制执行“顶大牌”规则。

### 2. 核心算法进化 (CFR Algorithm Evolution)

- **折扣奖励机制 (Discounted Reward Sampling)**:
    - 引入了时间维度的折扣权重 `weight = (step / total_steps)^2`。
    - **逻辑**: 扑克游戏中，后期的出牌对胜负的影响更直接。通过加大后期步骤在 Replay Buffer 中的权重，让网络重点学习“收尾”阶段的博弈，解决“前期打得好，后期掉链子”的问题。

- **奖励塑形 (Reward Shaping)**:
    - **改进**: 不再仅仅依赖最终的胜负分。
    - **逻辑**:
        - 每次出牌（减少手牌）获得 `+0.5` 的奖励。
        - 每次过牌（Pass）获得 `-0.1` 的惩罚。
    - **目的**: 引导 AI 更有目的地去减少手牌，而不是无意义地等待。

- **加权损失函数 (Weighted Loss Implementation)**:
    - 在 Regret 网络训练中引入了样本权重。
    - 配合折扣机制，使得高价值、关键时期的样本对梯度的贡献更大。

### 3. 训练流程优化 (Workflow Optimization)

- **Batch Size**: 保持 `32768` 以确保稳定性。
- **探索率**: 维持 `0.05` 的 Epsilon-Greedy 探索。
- **目标**: 依然锁定 **80%** 胜率，作为晋升 Champion 的硬性指标。
