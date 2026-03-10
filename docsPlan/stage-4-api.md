# 阶段 4：API 服务封装实施方案

## 目标与范围
- 将引擎与已训练策略对外封装为服务，支持人类或外部程序通过 HTTP/WebSocket 对战。
- 范围：对局生命周期管理、请求/响应契约、并发会话隔离、错误码与基本安全。

## 架构概览
- Web 框架：FastAPI（规划阶段仅约定接口，不落地实现）。
- 会话模型：GameSession（含状态机、玩家座位、AI 实例/推理句柄）。
- 推理模型：从阶段 3 加载冻结权重，提供同步/异步推理接口。

## 接口设计（HTTP v1）
- POST /game/start
  - 入参：{ mode: "pve"|"pvp", seats?: 3, engine_config?: {...} }
  - 出参：{ game_id, seat_id, state }
- POST /game/action
  - 入参：{ game_id, seat_id, action } // action 与阶段 2 的 ActionId 对齐
  - 出参：{ ok: true, state, events? }
- GET /game/state
  - 入参：{ game_id }
  - 出参：{ state, legal_actions, current_player }
- POST /game/ai
  - 入参：{ game_id, seat_id } // 请求 AI 为该座位出牌
  - 出参：{ action, policy?, value? }
- DELETE /game/end
  - 入参：{ game_id }
  - 出参：{ ok: true }

## WebSocket（可选 v2）
- 路径：/ws/game/{game_id}
- 事件：
  - server -> client：state_update、legal_actions、events、game_over
  - client -> server：action

## 模型与状态编码
- 与阶段 2 保持一致：state/action 的结构与编码完全复用，避免双份维护。
- legal_actions：直接由引擎生成；前端可用于可视化/客户端校验。

## 错误码与健壮性
- 400：参数非法（座位越界、非法 ActionId 等）
- 404：game_id 不存在或已结束
- 409：状态冲突（非当前轮或重复提交）
- 500：未知错误（包含 AI 推理异常）

## 并发与资源
- GameSession 限流（最大并发局数/玩家数），空闲回收。
- 推理队列与超时（例如 200ms 超时回退 RuleAgent）。

## 安全与可观测性
- 基础鉴权（Token/API Key）；请求限流与简单风控。
- 日志：请求日志、关键事件（炸弹、春天）、错误栈。
- 指标：QPS、平均响应时间、对局时长、推理耗时。

## 验收与产出
- Swagger 接口文档（由注解生成）与示例请求/响应。
- 集成测试用例：覆盖对局启动、行动、结束、异常路径。

## 里程碑
- M1：接口契约冻结；状态/动作协议与阶段 2 对齐。
- M2：对局管理/并发模型设计完毕；错误码与日志规范确定。
- M3：联通训练模型与引擎的集成方案评审通过。
