#!/usr/bin/env bash
# 启动 easyrag 后端：先检测 8020 端口，已被占用则跳过启动。
#
# 用法:
#   scripts/start-backend.sh              # 交互/手动：后台 nohup 拉起，日志 logs/backend.log
#   LAUNCHD=1 scripts/start-backend.sh    # launchd 前台模式：exec 进程由 launchd 监管（KeepAlive）
#
# 环境变量:
#   PORT     监听端口，默认 8020
#   PG_WAIT  等待 PostgreSQL(5432) 就绪的最长秒数，默认 60；0 = 不等
set -euo pipefail

PORT="${PORT:-8020}"
PG_WAIT="${PG_WAIT:-60}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/backend.log"
PID_FILE="$LOG_DIR/backend.pid"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "后端已在运行（端口 $PORT 被占用）:"
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
    exit 0
fi

mkdir -p "$LOG_DIR"
cd "$ROOT"

[ -f .env ] && set -a && source .env && set +a

# 开机自启时 PG 可能还没起来，先等端口就绪再启动，避免一轮崩溃重启。
if [ "$PG_WAIT" != "0" ]; then
    waited=0
    while ! lsof -nP -iTCP:5432 -sTCP:LISTEN >/dev/null 2>&1; do
        if [ "$waited" -ge "$PG_WAIT" ]; then
            echo "ERROR: 等待 PostgreSQL(5432) 超时 ${PG_WAIT}s，未启动后端" >&2
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
fi

BIN="$ROOT/target/release/easyrag"

if [ "${LAUNCHD:-0}" = "1" ]; then
    # launchd 前台模式： exec 替换自身，进程退出后由 launchd 按 KeepAlive 重启。
    if [ -x "$BIN" ]; then
        exec "$BIN"
    fi
    exec cargo run --release -p easyrag-app
fi

if [ ! -x "$BIN" ]; then
    echo "未找到 $BIN，先编译（cargo build --release -p easyrag-app）..."
    cargo build --release -p easyrag-app
fi

nohup "$BIN" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "启动中，日志: $LOG_FILE"

for _ in $(seq 1 30); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "后端已就绪: http://localhost:$PORT"
        exit 0
    fi
    sleep 2
done

echo "ERROR: 60 秒内端口 $PORT 未就绪，请查看日志: $LOG_FILE" >&2
exit 1
