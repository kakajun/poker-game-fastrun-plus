# 跑得快 (Run Fast) AI & API 项目策划方案

## 1. 项目概述
本项目旨在开发一个基于深度学习的"跑得快"（3人版，45张牌）游戏服务端。核心目标是训练一个能像人类一样思考的AI模型，并通过API接口对外提供对战服务，允许人类玩家接入挑战。

## 2. 可行性分析 (Feasibility Analysis)

### 2.1 技术可行性 (High)
*   **游戏逻辑**：跑得快规则相对固定，状态空间虽然大（牌型组合多），但远小于围棋，适合强化学习（Reinforcement Learning, RL）。
*   **AI模型**：使用 **Deep Q-Network (DQN)** 或 **PPO (Proximal Policy Optimization)** 是目前业界处理此类不完全信息博弈（Imperfect Information Game）的标准解法。
*   **开发语言**：Python 是最佳选择，它既是 AI 领域的通用语言（PyTorch/TensorFlow），也是优秀的后端语言（FastAPI/Django）。

### 2.2 难点与挑战 (Challenges)
1.  **牌型编码 (State Representation)**：如何把手中的牌、桌上的牌、历史出牌记录转换成神经网络能理解的 "0101" 矩阵。
2.  **合法动作生成 (Legal Action Generation)**：跑得快有"必压"规则（能管必须管），这要求游戏引擎必须能极其准确地计算出当前所有合法的出牌组合。
3.  **模型训练时间**：深度强化学习需要大量的自我对弈（Self-Play）来积累经验，训练初期AI会非常笨，需要耐心和算力。

### 2.3 结论
**完全可行**。作为一个学习型项目，它能涵盖 **游戏逻辑算法**、**深度学习**、**后端API开发** 三大核心领域，非常适合进阶。

---

## 3. 技术架构 (Technical Architecture)

我们将项目分为三个核心模块：

```mermaid
graph TD
    User[玩家/前端] -->|HTTP/WebSocket| API[API 服务层 (FastAPI)]
    API -->|Action| GameEngine[游戏核心引擎 (Rules)]
    API -->|State| AI[AI 推理模型 (PyTorch)]
    AI -->|Decision| GameEngine
```

### 3.1 核心技术栈
*   **语言**: Python 3.9+
*   **Web框架**: FastAPI (高性能，适合异步IO)
*   **AI框架**: PyTorch (灵活性高，适合研究)
*   **数据处理**: NumPy (矩阵运算)
*   **训练环境**: Gym (OpenAI 标准强化学习接口)

---

## 4. 开发计划 (Development Roadmap)

### 4.0 阶段文档索引
- 阶段 1：规则引擎实施方案（纯规则，无 AI）— [stage-1-engine.md](file:///f:/git/poker-game-fastrun/docs/stage-1-engine.md)
- 阶段 2：AI 训练环境实施方案（Gym）— [stage-2-env.md](file:///f:/git/poker-game-fastrun/docs/stage-2-env.md)
- 阶段 3：AI 模型训练实施方案 — [stage-3-training.md](file:///f:/git/poker-game-fastrun/docs/stage-3-training.md)
- 阶段 4：API 服务封装实施方案 — [stage-4-api.md](file:///f:/git/poker-game-fastrun/docs/stage-4-api.md)
- 阶段 5：测试、优化、上线实施方案 — [stage-5-release.md](file:///f:/git/poker-game-fastrun/docs/stage-5-release.md)
- 阶段 6：前端界面设计与实现 — [stage-6-ui.md](file:///f:/git/poker-game-fastrun/docs/stage-6-frontend.md)


### 第一阶段：游戏核心引擎 (The Foundation)
**目标**：实现一个只要给它指令，它就能按规则运行的"裁判"。
*   [x] 定义牌的数据结构（Card, Hand）。
*   [x] 实现牌型判断算法（单张、对子、顺子、炸弹等）。
*   [x] **关键点**：实现 `get_legal_actions(state)` 函数，根据当前桌面牌型，计算出玩家所有合法的出牌选择（包含"过"）。
*   [x] 实现游戏状态流转（发牌、出牌、结算、下一位）。

### 第二阶段：AI 环境搭建 (The Gym)
**目标**：让 AI 能"玩"这个游戏。
*   [x] 封装 Gym Environment：实现 `reset()` (开始新游戏) 和 `step(action)` (执行动作并返回奖励)。
*   [x] 定义 **State (状态)**：
    *   核心特征：我的手牌（52维）、其他两家剩余牌数（2维）、当前桌面牌（52+10维）。
*   [x] 定义 **Action (动作)**：
    *   抽象动作空间 (Abstract Action Space)：Pass + Type_Len_Max (共252维)。
    *   具体化逻辑：从手牌中自动匹配符合抽象特征的具体牌。
*   [x] 定义 **Reward (奖励)**：
    *   赢了 +100
    *   输了 -剩余牌数 (Step 奖励 +1)
    *   打出炸弹 +20
    *   非法出牌 -100 (Truncated)

### 第三阶段：AI 训练 (The Brain)
**目标**：训练一个能打赢随机策略的 AI。
*   [x] 搭建训练管线：
    *   使用 Stable Baselines 3 (SB3) 的 PPO 算法。
    *   实现 `SingleAgentWrapper`：将多人环境包装为单人环境，对手使用 RandomAgent。
    *   使用 `MaskablePPO` 处理非法动作掩码。
*   [x] 训练与评估：
    *   在 CPU 上训练 100,000 步。
    *   评估 100 局对抗随机对手的胜率。
    *   **成果**：胜率达到 **64.00%** (随机基准 33%)，平均奖励 +37.30。
*   [ ] (可选) 进阶训练：Self-Play（自我对弈），让 AI 对抗过去的自己，提升水平。

### 第四阶段：API 服务封装 (The Service)
**目标**：将核心逻辑和 AI 模型封装为 HTTP API，供前端调用。
*   [x] 技术选型：FastAPI + Uvicorn。
*   [x] 接口定义：
    *   `POST /game/start`: 创建新游戏，返回 Session ID。
    *   `GET /game/{id}/state`: 获取当前游戏状态（手牌、出牌区、分数）。
    *   `POST /game/{id}/action`: 玩家出牌接口。
    *   `POST /game/{id}/ai`: 触发 AI 决策并执行。
*   [x] 会话管理：实现简单的内存 SessionManager。
*   [x] AI 集成：封装 `AIService`，加载 PPO 模型并进行推理。
*   [x] 成果：API 服务已启动，并通过了初步测试。

### 第五阶段：测试、优化、上线 (The Polish)
**目标**：打造一个稳定、高性能、无 Bug 的产品。
*   [ ] **单元测试 (Unit Test)**：覆盖核心逻辑（如牌型判断、胜负判定）。
*   [ ] **集成测试 (Integration Test)**：测试 API 接口、数据库连接、AI 模型加载。
*   [ ] **性能测试 (Performance Test)**：模拟高并发场景，优化响应速度。
*   [ ] **文档编写 (Documentation)**：编写 API 文档、部署文档、用户手册。
*   [ ] **上线部署 (Deployment)**：配置服务器、域名、SSL 证书，发布上线。

### 第六阶段：前端界面设计 (The Face)
**目标**：构建一个直观、响应式的 Web 界面。
*   [x] 框架：Vue 3 + Vite + TypeScript + Tailwind CSS (或自定义 CSS)。
*   [x] 核心组件：
    *   `Card.vue`: 渲染单张扑克牌（支持选中状态）。
    *   `Hand.vue`: 展示玩家手牌，支持点击选牌。
    *   `GamePage.vue`: 游戏主界面，布局玩家位置、出牌区、控制按钮。
*   [x] 状态管理：使用 Pinia 管理游戏状态 (`src/store/game.ts`)。
*   [x] 前后端对接：
    *   创建 `src/api/game.ts` 封装 Axios 请求。
    *   重构 `src/store/game.ts`，将纯前端逻辑替换为 API 调用。
    *   实现数据模型适配（后端 int ID -> 前端 string ID）。
*   [ ] (可选) 动画效果：发牌动画、出牌飞入效果。

### 总结 (Summary)
