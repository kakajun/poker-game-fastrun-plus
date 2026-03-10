# 游戏核心引擎 (Core Engine)

本目录包含跑得快游戏的所有核心逻辑实现，不依赖任何 AI 框架。

## 模块说明

*   **card.py**: 定义扑克牌 (`Card`)、花色 (`Suit`)、点数 (`Rank`)。
*   **deck.py**: 定义牌堆 (`Deck`)，负责生成45张牌、洗牌、发牌。
*   **hand_type.py**: 定义牌型枚举 (`HandType`) 和出牌对象 (`Play`)。
*   **evaluator.py**: 牌型评估器 (`HandEvaluator`)，负责识别牌型、比较牌力 (`can_beat`)。
*   **action_generator.py**: 动作生成器 (`ActionGenerator`)，负责生成所有合法出牌动作。
*   **game.py**: 游戏主控类 (`Game`)，负责管理游戏状态、轮转、计分、胜利判定。

## 使用示例

```python
from src.core.game import Game
from src.core.hand_type import HandType, Play

# 初始化游戏
game = Game(seed=42)

# 获取当前玩家合法动作
actions = game.get_legal_actions()

# 执行动作
play = actions[0]
is_over, events = game.step(play)

if is_over:
    print(f"Winner: {game.winner}")
    print(f"Scores: {game.scores}")
```
