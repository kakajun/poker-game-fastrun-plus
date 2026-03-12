# Poker Game FastRun AI (跑得快 AI)

这是一个基于前沿博弈论算法 **VAD-CFR (Volatility-Adaptive Discounted Counterfactual Regret Minimization)** 的跑得快游戏 AI 项目。
VAD-CFR 是由 DeepMind 通过 AlphaEvolve 自动进化出的新一代算法，专门针对非完全信息博弈中的波动性进行了优化，相比传统 CFR 具有更快的收敛速度和更强的策略稳定性。

包含核心游戏引擎、VAD-CFR 训练管线、FastAPI 后端服务以及 Vue 3 前端界面。

## 🎯 项目成果

- **核心算法**: VAD-CFR (波动适应折扣反事实遗憾最小化)
- **技术亮点**: 非对称瞬时提升 (Asymmetric Boosting) + 波动适应折扣 (Volatility-Adaptive Discounting) + 硬热启动 (Hard Warm-Start)
- **技术栈**: Python 3.9, PyTorch (GPU 加速), FastAPI, Vue 3, TypeScript

## 📂 目录结构

```
poker-game-fastrun/
├── src/                # 后端源码
│   ├── algo/           # 算法实现
│   │   └── vad_cfr/    # VAD-CFR 核心逻辑 (网络模型、训练器)
│   ├── core/           # 游戏核心逻辑 (规则、发牌、判胜)
│   ├── env/            # 环境封装 (ActionSpace, Observation)
│   ├── agent/          # AI 代理 (ModelAgent)
│   ├── api/            # FastAPI 服务接口
│   └── train_vad_cfr.py # VAD-CFR 训练脚本
├── front/              # 前端源码 (Vue 3 + Vite)
├── models/             # 训练好的模型文件 (.pth)
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # 根据 CUDA 版本选择
pip install fastapi uvicorn pydantic numpy pandas matplotlib
```

### 2. 启动后端服务

后端提供游戏逻辑和 AI 推理接口。

```bash
# 在项目根目录下运行
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
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

浏览器访问: `http://localhost:5173`

### 4. 训练 AI (VAD-CFR)

项目采用 VAD-CFR 算法进行自博弈训练。支持多进程异步采样与 GPU 集中训练。

```bash
# 运行 VAD-CFR 训练脚本
$env:PYTHONPATH="."; python src/train_vad_cfr.py
```

- **输出**: `models/vad_cfr.pth` (包含 `_best` 和 `_pruned` 版本)
- **特点**:
    - **Asymmetric Boosting**: 对有潜力的动作进行 1.1 倍概率放大。
    - **Volatility-Adaptive**: 根据遗憾值的幅度动态调整样本权重。
    - **Hard Warm-Start**: 前 500 轮仅训练遗憾网络，避免早期噪声干扰。

## 📊 模型评估 (Model Evaluation)

项目提供了多维度的模型评估工具，用于量化 AI 的竞技水平和策略质量。

### 1. 评估维度
- **竞技表现**: 胜率、平均得分、春天率。
- **策略质量**: 顶大规则遵守率、炸弹使用频率。
- **效率**: 获胜平均步数。

### 2. 运行评估
```bash
# 运行模型对弈测评 (自动对比 models 目录下的 .pth 模型)
$env:PYTHONPATH="."; python src/evaluate/evaluator.py
```

评估结果（CSV 报表与可视化对比图）将自动生成在 `src/evaluate/reports/` 目录下。

## 🧠 核心逻辑

### 状态空间 (Observation)
- **57 维精简向量**: 移除了花色属性，仅保留点数逻辑。
- **手牌 (15维)**: 各点数 (3-2) 的张数计数。
- **剩余牌数 (2维)**: 对手手牌数量归一化。
- **上家出牌 (15维)**: 上一手打出的各点数张数。
- **上家牌型 (10维)**: 牌型的 One-hot 编码。
- **已出牌统计 (15维)**: **关键信息**，全场已打出的各点数张数统计（用于算牌）。

### 动作空间 (Action)
- **252 维离散空间**: 包含 Pass 和所有可能的合法牌型抽象。
- **Masking**: 使用 Action Masking 技术屏蔽非法动作。

### 奖励函数 (Reward)
- **胜负**: +100 / -100
- **步数惩罚**: 每步 -1 (鼓励快速出完)
- **炸弹奖励**: +20 (实时结算)
- **输家惩罚**: -剩余牌数 (基础分)

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License
