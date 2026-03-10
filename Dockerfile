# 使用轻量级 Python 3.9 基础镜像
FROM python:3.9-slim

# 设置环境变量，优化构建体积
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# 设置工作目录
WORKDIR /app

# 安装系统依赖并清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. 明确安装 PyTorch CPU 版本 (这是减小体积的关键，约 200MB)
# 对于 x86_64 使用专门的 CPU 索引，对于 ARM64 (aarch64) 直接从 PyPI 安装
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio; \
    fi

# 2. 复制依赖文件并安装其他库
COPY requirements.txt .
# 移除 requirements.txt 中的 torch 引导，避免冲突
RUN sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 3. 复制后端代码
COPY src/ ./src/

# 4. 复制训练好的模型
COPY models/ ./models/

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
