# VAD-CFR 架构与算法实现文档

本文档详细说明了本项目（跑得快 Poker AI）后端重构后的架构、核心逻辑及 VAD-CFR（Volatility-Adaptive Discounted Counterfactual Regret Minimization）算法的实现细节。

## 1. 核心算法：VAD-CFR

本项目采用 DeepMind 最新提出的 **VAD-CFR** 算法，这是一种通过 AlphaEvolve 自动进化出的新一代 CFR 变体。相比传统 Deep CFR，它引入了非直观的动态机制来处理不完全信息博弈中的波动性。

### 1.1 核心机制

VAD-CFR 在标准 Deep CFR 的基础上引入了以下三个关键改进：

1.  **非对称瞬时提升 (Asymmetric Instantaneous Boosting)**
    *   **原理**：在采样阶段，对于计算出的**正向遗憾值 (Positive Regret)**，乘以一个提升系数（本项目设置为 `1.1`）。
    *   **目的**：加速对潜在优势动作的探索。在训练早期，能够更快地放大有希望的动作的概率，从而跳出局部最优。
    *   **代码位置**：`src/algo/vad_cfr/vad_trainer.py` -> `worker_collect_vad`

2.  **波动适应折扣 (Volatility-Adaptive Discounting / Regret-Magnitude Weighting)**
    *   **原理**：在收集策略训练数据（Strategy Data）时，不仅仅根据迭代次数加权，而是根据**瞬时遗憾的幅度 (Regret Magnitude)** 进行加权。
    *   **目的**：自动过滤低信息量的状态（即动作之间差异不大、遗憾值很小的状态），聚焦于关键决策点（高遗憾值状态）。
    *   **代码位置**：`src/algo/vad_cfr/vad_trainer.py` -> `worker_collect_vad` (采样时计算 `regret_magnitude`)

3.  **硬热启动 (Hard Warm-Start)**
    *   **原理**：设定一个预热期（如前 500 次迭代），在此期间**只训练遗憾网络 (Regret Net)**，**不收集也不训练策略网络 (Strategy Net)**。
    *   **目的**：在遗憾网络尚未收敛、对局面判断尚不准确时，避免错误的策略数据污染平均策略网络。
    *   **代码位置**：`src/algo/vad_cfr/vad_trainer.py` -> `collect_and_update`

## 2. 系统架构

后端采用 **异步分布式采样 + 集中式 GPU 训练** 的架构，以最大化利用多核 CPU 和 GPU 资源。

### 2.1 模块结构

```text
src/
├── algo/
│   └── vad_cfr/
│       ├── deep_model.py   # 神经网络定义 (DeepRegretNet, DeepStrategyNet)
│       ├── model.py        # 推理封装 (VADCFRModel)
│       └── vad_trainer.py  # 训练核心 (Trainer, Worker, ReplayBuffer)
├── api/
│   ├── ai_service.py       # AI 服务层 (加载模型, 提供预测接口)
│   └── app.py              # FastAPI 接口
├── core/                   # 游戏核心引擎 (规则, 牌型判断)
├── env/                    # 强化学习环境封装 (ObsEncoder, ActionSpace)
└── train_vad_cfr.py        # 训练入口脚本
```

### 2.2 数据流向

1.  **采样阶段 (CPU Workers)**：
    *   多个 Worker 进程并行运行 `Game` 实例。
    *   使用 `Regret Net`（从共享内存加载最新权重）预测当前局面的遗憾值。
    *   应用 **Asymmetric Boosting** 处理遗憾值，生成当前策略。
    *   根据策略采样动作，与环境交互。
    *   收集两种数据存入共享队列 `Queue`：
        *   **Regret Data**: `(state, action, reward)` —— 用于训练 Regret Net。
        *   **Strategy Data**: `(state, probability_distribution, weight)` —— 用于训练 Strategy Net（仅在热启动后收集）。

2.  **训练阶段 (Main Process / GPU)**：
    *   主进程从 `Queue` 中取出数据，存入 `ReplayBuffer`（区分 Regret Buffer 和 Strategy Buffer）。
    *   **Regret Update**: 从 Regret Buffer 采样，计算 MSE Loss 更新 `DeepRegretNet`。
    *   **Strategy Update**: 从 Strategy Buffer 采样，根据 **Regret Magnitude Weight** 计算加权 MSE Loss，更新 `DeepStrategyNet`。
    *   定期将更新后的模型权重同步到共享内存，供 Worker 使用。

## 3. 关键逻辑代码解析

### 3.1 网络模型 (`deep_model.py`)

*   **DeepRegretNet**: 输入状态向量 (57维)，输出每个动作的预估遗憾值 (Q-value 差值)。
*   **DeepStrategyNet**: 输入状态向量，输出动作的概率分布 (Softmax)。
*   **特点**: 使用了较宽的全连接层 (1024 -> 1024 -> 512) 以捕捉复杂的博弈动态。

### 3.2 训练器 (`vad_trainer.py`)

*   **ReplayBuffer**: 优化过的 Numpy 缓冲区，支持存储权重 `weights`。
*   **worker_collect_vad**:
    ```python
    # 伪代码逻辑
    pred_regrets = regret_net(obs)
    # 1. 非对称提升
    boosted_regrets = where(pred > 0, pred * 1.1, pred)
    # 2. 计算当前策略
    probs = compute_regret_matching(boosted_regrets)
    # 3. 计算权重 (波动适应)
    weight = max(abs(raw_regrets))
    # 4. 存储
    queue.put(StrategySample(obs, probs, weight))
    ```
*   **update_network**:
    ```python
    # 伪代码逻辑
    # 策略网络更新使用加权 Loss
    loss = mean(weights * (pred_probs - target_probs)^2)
    ```

### 3.3 推理服务 (`ai_service.py`)

*   **VADCFRModel**: 封装了 `DeepStrategyNet`。
*   **预测逻辑**:
    1.  加载 `.pth` 模型权重。
    2.  接收 `Game` 对象，编码为状态向量。
    3.  使用 `Strategy Net` 预测概率分布。
    4.  过滤掉非法动作，重新归一化概率。
    5.  选择概率最大的动作 (Deterministic) 或按概率采样 (Stochastic)。

## 4. 性能与优势

*   **收敛速度**: 相比 MCCFR，VAD-CFR 的非对称提升机制使其能更快发现并锁定优势策略。
*   **稳定性**: 波动适应权重机制使其在策略震荡剧烈的阶段降低学习率（通过权重），在策略稳定时增加权重，从而训练出更鲁棒的模型。
*   **纯粹性**: 移除了 PPO 等异构算法，专注于非完全信息博弈的纳什均衡逼近。

---
*文档生成时间: 2026-03-12*
