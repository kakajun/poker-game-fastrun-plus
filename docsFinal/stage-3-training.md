# 阶段 3：AI 模型训练实施方案（已根据实际代码更新）

## 目标与范围
- 以阶段 2 的 Gym 环境为基础，训练一个能稳定战胜 `RandomAgent`，且与 `HeuristicAgent` 打平或占优的策略。
- 目前已实现基础训练管线，支持针对随机对手的强化学习训练及多维度效果评估。

## 算法选择
- **核心算法**: `MaskablePPO` (来自 `sb3_contrib`)。
- **选择理由**: “跑得快”游戏在不同状态下合法动作差异巨大，使用动作掩码 (Action Masking) 可以彻底消除非法动作带来的无效探索，大幅提升收敛速度。
- **策略网络**: `MlpPolicy` (多层感知机)，输入为 116 维观测向量，输出为 252 维动作空间的概率分布。

## 动作掩码融合
- 通过 `ActionMasker` (包装器) 在 `step` 前获取环境生成的 `action_mask`。
- `MaskablePPO` 内部在计算 Categorical 分布时自动对掩码位置进行屏蔽。

## 训练流程与超参数
- **代码实现**: [train_ppo.py](file:///e:/git/poker-game-fastrun/src/train_ppo.py)
- **核心参数**:
  - `learning_rate`: `3e-4` (基础学习率)
  - `n_steps`: `2048` (每次更新的采样步数)
  - `batch_size`: `64`
  - `gamma`: `0.99` (折扣因子)
  - `ent_coef`: `0.01` (熵系数，控制探索强度)
- **日志记录**: 使用 `Monitor` 记录每个回合的奖励和长度，保存为 `monitor.csv`。

## 评测与可视化 (已实现)
项目建立了多维度的评估体系，详见 [README.md](file:///e:/git/poker-game-fastrun/README.md) 及 [evaluator.py](file:///e:/git/poker-game-fastrun/src/evaluate/evaluator.py)。
- **训练曲线**: 自动生成 `training_curves.png`，展示奖励与回合长度的滑动平均变化。
- **春天挑战 (Spring Challenge)**: [spring_evaluator.py](file:///e:/git/poker-game-fastrun/src/evaluate/spring_evaluator.py) 专门测试模型在优势局下的压制力。
- **核心指标**:
  - **胜率 (Win Rate)**: 针对 `RandomAgent` 和 `HeuristicAgent` 的获胜比例。
  - **春天率 (Spring Rate)**: 让对手一张牌未出的对局占比。
  - **步数效率**: 获胜所需的平均步数。

## 对手池 (Opponent Pool)
- **当前状态**: `SingleAgentWrapper` 默认使用 `RandomAgent` 作为对手进行训练。
- **后续规划**: 引入自我对弈 (Self-Play)，即让模型与自身的历史版本进行对抗，以突破单纯模仿随机或固定规则的局限。

## 风险与里程碑
- ✅ **M1**: 训练脚手架搭建完成，支持 Maskable PPO。
- ✅ **M2**: 实现 `HeuristicAgent` 基准线与多维度评估系统。
- ⏳ **M3**: 优化超参数，针对 `HeuristicAgent` 达到 >50% 胜率。
- ⏳ **M4**: 引入 Self-Play 提升博弈深度。
