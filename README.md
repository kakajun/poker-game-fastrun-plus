# Poker Game FastRun AI（跑得快 AI）

三人跑得快 AI 工程：规则引擎、合法信息集编码、全动作策略网、numpy 推理服务与前端页面。

规则口径见 `mainRule.md`：48 张无花色牌（3-K 各 4 张、A 3 张、2 1 张），每人 16 张，**必压 + 顶大 + 无下注**。计分含基础分、炸弹（+20/-10）与春天（±30）。

> 历史文档 `VAD_CFR_Architecture.md` / `MCCFR_TRANSFORMATION_PLAN.md` 记录的是早期探索，不代表当前线上策略。当前部署的是 `policy_v4` 策略网。

## 当前线上策略

| 项 | 值 |
|---|---|
| 模型 | `models/policy_v4.pt` → 运行期用 `models/policy_v4.npz` |
| 网络 | `PolicyNetV3`（flatten 历史，**纯 MLP**） |
| 输入 | state 97 维 + 有序历史 128×8 + 每个合法动作 28 维 |
| 决策 | 全部合法动作打分；净分优先 + 胜率守卫 `win_guard=0.005`；残局用精确解（`endgame_cards=14`, `k=6`） |
| 推理 | **numpy 前向，运行镜像不含 torch** |
| gate 实测 | 对 greedy 胜率 61.2%（+4.58pt，CI [0.029, 0.062]）；对 m2 净分 +0.50（CI [0.121, 0.882]） |

完整评测与阈值见 `models/policy_v4.pt.json` 的 `gate` 字段。

## 快速开始

### 方式 A：直接跑镜像（推荐）

镜像已打包为工程根目录的 `poker-fastrun.tar.gz`（约 83MB，加载后 356MB），单容器同时提供前端页面和 API：

```bash
docker load -i poker-fastrun.tar.gz
docker run -d -p 9551:80 --restart unless-stopped --name poker-fastrun poker-fastrun:latest
```

浏览器打开 `http://localhost:9551`。后端 API 见下表；FastAPI 文档仅在容器内 `http://127.0.0.1:8000/docs`，未对外映射。

### 方式 B：从源码构建镜像

```bash
# 国内网络：用 daocloud 镜像源
docker build --build-arg BASE_PREFIX=docker.m.daocloud.io/library/ -t poker-fastrun:latest .

# GitHub Actions / 海外网络：默认走 docker.io 官方镜像
docker build -t poker-fastrun:latest .
```

多阶段构建：前端 `vite build` → `dist/`，运行时 Python 3.11-slim + nginx，nginx 托管静态页并把 `/api/` 反代到本地 uvicorn（8000）。history 路由刷新由 `try_files ... /index.html` 兜住。

### 方式 C：本地开发（前后端分离）

后端需要 Python 与 numpy/fastapi/uvicorn/pydantic；**只有重新导出权重或训练时才需要 torch**：

```bash
pip install fastapi uvicorn pydantic numpy
PYTHONPATH=. python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

前端（Node 18+）：

```bash
cd front
npm install
npm run dev          # http://localhost:5173，/api 代理到 8000
```

## API

| 端点 | 说明 |
|---|---|
| `GET /models` | 可用模型：`policy-v4`、`m2-rule`、`m2-search` |
| `POST /game/start` | 开局，body `{"model_name":"policy-v4"}` |
| `GET /game/{id}/state` | 当前状态 |
| `POST /game/{id}/action` | 人类出牌，body `{"card_ids":[...]}` |
| `POST /game/{id}/ai` | AI 走一步，返回状态与 `thinking.top_actions` |

## 重新导出模型权重

训练产出 `.pt` 后，必须重新导出 npz，运行镜像才能用上新权重：

```bash
# 开发机（需 torch）
PYTHONPATH=. python tools/export_npz.py models/policy_v4.pt

# 校验 numpy 前向与 torch 一致
PYTHONPATH=. python tools/verify_numpy_net.py models/policy_v4.pt
```

导出脚本只支持 `history_mode="flatten"`。`.pt` 不会被打进运行镜像。

## 目录

```text
src/
├── fastrun/          # 不可变状态机、规则、动作生成
├── core/             # 旧引擎（可变形，不用于 CFR/搜索）
├── eval/             # 拆牌手数、控制权、计分目标、M1 规则策略
├── net/              # encode_v3 (97/28)、PolicyNetV3、numpy_model
├── arena/            # Agent：规则/搜索/策略网/NumpyPolicyAgent
├── search/           # PIMC、残局 max-n 精确解
├── api/              # FastAPI 服务
└── train_*.py        # 训练入口
tools/                # 竞技场、数据生成、权重导出与校验
front/                # Vue 3 + Vite
deploy/               # nginx.conf、start.sh
models/               # policy_v4.pt / .npz / 元数据
```

## 评测与 CI

- 竞技场：`tools/arena_m1.py`、`tools/arena_m3.py`，按 deal 聚类做 paired bootstrap。
- 模型评测：`src/evaluate/evaluator.py`，报表输出到 `src/evaluate/reports/`。
- CI：push 到 `main` 或打 `v*.*.*` 标签时，GitHub Actions 构建 amd64/arm64 镜像并推送到 `ghcr.io`。

## 注意事项

- 规则是 3 人且「保本」使计分非严格零和，因此**不能声称收敛到纳什均衡**；当前指标是净分与胜率。
- 容器默认没有鉴权。公网部署时请限制安全组来源 IP，或在前面加带认证的反向代理。
- `PLAN.md` 是 v2 迭代计划，含验收门槛与消融矩阵；改动策略请先读它。

## 许可证

MIT
