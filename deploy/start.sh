#!/bin/sh
set -eu

# 单容器内同时跑后端与 nginx：uvicorn 只监听回环，对外只暴露 80
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info &
UVICORN_PID=$!

shutdown() {
    kill -TERM "$UVICORN_PID" 2>/dev/null || true
    nginx -s quit 2>/dev/null || true
    wait "$UVICORN_PID" 2>/dev/null || true
}
trap shutdown TERM INT

# 等模型加载完再开 nginx，避免首屏 /api 请求拿 502（最多等 90s）
i=0
while [ "$i" -lt 90 ]; do
    if curl -fsS http://127.0.0.1:8000/models >/dev/null 2>&1; then
        echo "[start] backend ready after ${i}s"
        break
    fi
    i=$((i + 1))
    sleep 1
done

nginx -g 'daemon off;' &
NGINX_PID=$!
wait "$NGINX_PID"
shutdown
