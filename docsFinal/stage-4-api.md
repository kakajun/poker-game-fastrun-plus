# 阶段 4：API 服务封装实施方案（已根据实际代码更新）

## 目标与范围
- 将游戏引擎与已训练的 AI 策略封装为 Web 服务，支持前端 Vue 界面进行对战。
- 核心功能：对局生命周期管理、玩家动作校验、AI 推理集成、多会话并发支持。

## 架构实现
- **Web 框架**: FastAPI
- **会话管理**: [session_manager.py](file:///e:/git/poker-game-fastrun/src/api/session_manager.py)
  - `GameSession`: 维护单个对局的 `Game` 实例、玩家身份（人类 vs AI）。
  - `SessionManager`: 负责创建、获取和回收 UUID 标识的会话。
- **AI 服务**: [ai_service.py](file:///e:/git/poker-game-fastrun/src/api/ai_service.py)
  - 负责加载 PPO 模型并执行推理，将环境观测映射为游戏动作。

## 接口设计 (RESTful API)
- **POST `/game/start`**
  - 功能：创建并初始化一个新的游戏会话。
  - 返回：完整的 `GameStateModel`（含 `game_id`）。
- **GET `/game/{game_id}/state`**
  - 功能：获取指定对局的当前状态。
  - 返回：`GameStateModel`（包含手牌、上一次出牌、合法动作列表等）。
- **POST `/game/{game_id}/action`**
  - 功能：接收人类玩家的出牌动作。
  - 入参：`ActionRequest` (包含 `card_ids`)。
  - 校验：后端会自动验证所选牌是否组成合法牌型，以及是否能管住上家。
- **POST `/game/{game_id}/ai`**
  - 功能：触发 AI 进行出牌。
  - 返回：AI 行动后的最新游戏状态。

## 数据模型 (Schema)
详见 [models.py](file:///e:/git/poker-game-fastrun/src/api/models.py)：
- **CardModel**: `rank`, `suit`, `id` (0-51)。
- **PlayModel**: `type` (牌型名称), `cards`, `length`, `max_rank`。
- **GameStateModel**:
  - 包含 `hands` (所有玩家手牌)、`last_play` (上一手信息)。
  - `legal_actions`: 当前玩家所有可行的 `PlayModel` 列表，供前端高亮可用牌。
  - `scores`, `bomb_scores`: 实时得分统计。

## 规则校验逻辑
- 在 `/game/{game_id}/action` 接口中，后端会调用 `HandEvaluator` 识别玩家出的牌型。
- 通过 `game.get_legal_actions()` 确保玩家动作完全符合“跑得快”规则（包含必压、顶大等约束）。

## 运行与部署
- **本地启动**: `python -m uvicorn src.api.app:app --reload`
- **跨域支持**: 已集成 `CORSMiddleware`，支持前端开发环境 (Vite) 跨域请求。

## 里程碑
- ✅ **M1**: FastAPI 基础框架与路由搭建完成。
- ✅ **M2**: 会话管理与模型序列化逻辑实现。
- ✅ **M3**: 动作校验与 AI 推理集成跑通。
- ✅ **M4**: 支持前端 Vue 界面完整对局流程。
