# 本地构建传 --build-arg BASE_PREFIX=docker.m.daocloud.io/library/ 走国内源；
# CI/海外默认用 docker.io 官方镜像（且官方镜像支持 amd64/arm64 多架构）。
ARG BASE_PREFIX=

# ================= 1. 前端构建 =================
FROM ${BASE_PREFIX}node:26-bookworm-slim AS frontend
WORKDIR /build

# 先只拷 package.json，让依赖层可缓存
COPY front/package.json ./
RUN npm install --no-audit --no-fund --loglevel=error

COPY front/ ./
# 直接走 vite，跳过 vue-tsc 类型检查（类型错误不该阻断出镜像）
RUN npx --no-install vite build


# ================= 2. 运行时：后端 + nginx =================
# 不装 torch：策略网是纯 MLP，权重已导出成 numpy 可读的 .npz
ARG BASE_PREFIX=
FROM ${BASE_PREFIX}python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

# 后端运行时实际只用到这四个（不是 requirements.txt 里的全部：
# pandas/tqdm/rich 是训练与评估用，不进运行镜像）
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        numpy \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY src/ ./src/
# 只带 numpy 权重；.pt 是 torch checkpoint，运行期不需要
COPY models/policy_v4.npz models/policy_v4.npz.json ./models/

COPY --from=frontend /build/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/start.sh /usr/local/bin/start.sh
RUN sed -i 's/\r$//' /etc/nginx/conf.d/default.conf /usr/local/bin/start.sh \
    && chmod +x /usr/local/bin/start.sh

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1/api/models > /dev/null || exit 1

CMD ["/usr/local/bin/start.sh"]
