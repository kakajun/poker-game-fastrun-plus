# Poker Game FastRun AI (跑得快 AI)

这是一个基于强化学习 (PPO) 的跑得快游戏 AI 项目。
包含核心游戏引擎、Gym 训练环境、PPO 训练管线、FastAPI 后端服务以及 Vue 3 前端界面。

## 🎯 项目成果

- **AI 胜率**: 64% (vs 随机策略)
- **训练环境**: OpenAI Gym + Maskable PPO (处理非法动作)
- **技术栈**: Python 3.9, PyTorch, Stable-Baselines3, FastAPI, Vue 3, TypeScript

## 📂 目录结构

```
poker-game-fastrun/
├── src/                # 后端源码
│   ├── core/           # 游戏核心逻辑 (规则、发牌、判胜)
│   ├── env/            # Gym 环境 (ActionSpace, Observation, Reward)
│   ├── agent/          # AI 代理 (RandomAgent, PPO Wrapper)
│   ├── api/            # FastAPI 服务接口
│   └── train_ppo.py    # PPO 训练脚本
├── front/              # 前端源码 (Vue 3 + Vite)
├── models/             # 训练好的模型文件
├── tests/              # 单元测试
└── environment.yml     # Conda 环境配置
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 Conda。

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate poker-rl

# 如果需要手动安装核心依赖 (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install sb3-contrib stable-baselines3 shimmy fastapi uvicorn pydantic numpy
```

```bash
# 自建环境
conda activate .\.venv
```

### 2. 启动后端服务

后端提供游戏逻辑和 AI 推理接口。

#### 本地运行
```bash
# 在项目根目录下运行
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Docker 运行 (推荐)
```bash
# 构建镜像
docker build -t poker-backend .

# 运行容器
docker run -d -p 8000:8000 --name poker-ai poker-backend
```

服务启动后，API 文档位于: `http://localhost:8000/docs`


### 3. 启动前端界面

前端提供可视化的游戏交互。

```bash
# 进入前端目录
cd front

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

浏览器访问: `http://localhost:5173` (或终端显示的端口)

### 4. 训练 AI (可选)

如果你想重新训练模型，需要安装额外的训练依赖：

```bash
# 安装训练专用依赖 (包含 tensorboard, stable-baselines3, gymnasium 等)
pip install -r requirementsTrain.txt

# 运行训练脚本 (使用 CPU 训练约需 5-10 分钟)
python src/train_ppo.py
```

模型将保存到 `models/ppo_poker_final.zip`。

## 📊 模型评估 (Model Evaluation)

项目提供了多维度的模型评估工具，用于量化 AI 的竞技水平和策略质量。

### 1. 评估维度
- **竞技表现 (Competitive)**: 统计胜率、平均得分、春天达成率及得分分布。
- **策略质量 (Strategy)**:
  - **炸弹使用效率**: 炸弹的打出频率及成功率。
  - **关键牌控制**: 对 2、A、K 等大牌的使用时机。
- **获胜效率 (Efficiency)**: 获胜所需的平均步数（步数越少说明牌型组合越优）。
- **鲁棒性 (Robustness)**: 分析起手手牌强度与最终胜率的相关性。

### 2. 春天挑战 (Spring Challenge)
专门测试 AI 在绝对优势局下的压制能力。系统会为 AI 注入一组“必胜神牌”（包含红桃3首出长顺子、大连对、炸弹等），测试其是否能按照正确顺序打出“春天”（对手一张牌未出）。

### 3. 运行评估
```bash
# 设置环境变量并运行通用评估 (对比 models 文件夹下的所有模型)
$env:PYTHONPATH = '.'; .\.venv\python.exe src\evaluate\evaluator.py

# 运行“打春天”专项能力评估
$env:PYTHONPATH = '.'; .\.venv\python.exe src\evaluate\spring_evaluator.py
```

评估结果（CSV 报表与可视化对比图）将自动生成在 `src/evaluate/reports/` 目录下。

## 🧠 核心逻辑

### 状态空间 (Observation)
- **手牌 (52维)**: One-hot 编码
- **剩余牌数 (2维)**: 对手手牌数量
- **上家出牌 (62维)**: 牌型、牌值、长度

### 动作空间 (Action)
- **252 维离散空间**: 包含 Pass 和所有可能的合法牌型抽象 (Type + Length + MaxRank)。
- **Masking**: 使用 Action Masking 技术屏蔽非法动作，加速训练收敛。

### 奖励函数 (Reward)
- **胜负**: +100 / -100
- **步数惩罚**: 每步 -1 (鼓励快速出完)
- **炸弹奖励**: +20
- **输家惩罚**: -剩余牌数

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License
