# 阶段 6：前端界面开发（Vue 3 + Human-AI 交互）实施方案（已根据实际代码更新）

## 1. 目标
构建一个基于 Web 的图形化界面，让用户能够直观地与 AI 进行对战。前端通过 API 与后端游戏引擎交互，展示游戏进程、手牌、出牌动画及最终结算。

## 2. 技术栈
*   **框架**: Vue 3 (Composition API)
*   **构建工具**: Vite
*   **状态管理**: Pinia (管理游戏会话、玩家手牌、出牌记录、托管状态等)
*   **网络请求**: Axios (封装于 `gameApi`，支持对局启动、状态获取、动作提交及 AI 触发)
*   **样式**: Tailwind CSS (用于响应式布局与 UI 样式)
*   **资源**:
    *   **扑克牌图片**: `front/src/assets/porerImg/` 下的 1-52.png。
    *   **映射逻辑**: 在 `useGameStore` 中通过 `mapCard` 函数将后端的 `Card ID` (0-51) 转换为前端可识别的 Rank/Suit 对象，并映射到图片。

## 3. 核心功能模块实现

### 3.1 状态管理与自动流程 (Pinia)
实现于 [game.ts](file:///e:/git/poker-game-fastrun/front/src/store/game.ts)：
*   **状态同步 (syncState)**: 将后端返回的 `GameStateModel` 实时映射为前端视图状态，包括各家手牌数量、桌面最后一次出牌等。
*   **AI 自动推进 (processAiTurn)**: 当轮到 AI 玩家时，前端会自动触发 `/game/ai` 接口，并增加 800ms 的人为延迟，提升对战真实感。
*   **自动跳过 (Auto-pass)**: 当人类玩家“要不起”（唯一合法动作为 PASS）时，系统会在 1s 后自动跳过，优化游戏节奏。
*   **托管模式 (Auto-play)**: 支持人类玩家开启托管，由 AI 代理出牌（设点 3s 思考延迟）。

### 3.2 游戏主界面 (Game Board)
实现于 [GamePage.vue](file:///e:/git/poker-game-fastrun/front/src/pages/GamePage.vue)：
*   **布局结构**:
    *   **顶部/侧边**: 展示 Bot 1 和 Bot 2 的头像、剩余牌数及出牌状态。
    *   **底部**: 展示人类玩家手牌及“提示”、“出牌”、“过”等交互按钮。
    *   **中央**: 核心对战场地，分区域展示各家打出的牌型。
*   **交互逻辑**:
    *   **选牌**: 点击手牌上浮，通过 `Hand` 组件收集选中的卡牌。
    *   **反馈**: 实时显示“Your Turn!”或“Thinking...”状态。

### 3.3 结算系统 (Settlement)
*   **触发机制**: 后端返回 `is_over: true` 时自动弹出。
*   **展示内容**: 胜负大标题、赢家昵称、再来一局按钮。

## 4. 接口对接 (API Integration)
实现于 [game.ts](file:///e:/git/poker-game-fastrun/front/src/api/game.ts)：
*   使用 Vite 的 `server.proxy` 将 `/api` 请求转发至后端的 8000 端口。
*   封装了 `startGame`, `getState`, `playerAction`, `triggerAi` 四个核心方法。

## 5. 里程碑总结
*   ✅ **M1**: 完成 Vue 3 + Vite + Tailwind 基础架构搭建。
-   ✅ **M2**: 实现扑克牌资源映射与 `Card`/`Hand` 基础组件。
-   ✅ **M3**: 完成 Pinia 状态机开发，实现与后端的全流程数据同步。
-   ✅ **M4**: 增加 AI 自动推进与 800ms 交互延迟优化。
-   ✅ **M5**: 实现“自动跳过”与“玩家托管”高级功能，显著提升游戏体验。
