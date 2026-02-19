#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SERVER="$ROOT/python/server.py"
PORT=18888
PID_FILE="$ROOT/.python_server.pid"

# ── 颜色输出 ──────────────────────────────────────────────
info()  { echo -e "\033[34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[31m[ERR ]\033[0m  $*" >&2; }

# ── 退出时清理 Python 进程 ─────────────────────────────────
cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            info "停止 Python 服务（PID $PID）..."
            kill "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT INT TERM

# ── 检查 Python（优先 3.10+，回退 python3/python）──────────
info "检查 Python 环境..."
PYTHON=""
for candidate in python3.11 python3.10 python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 10 ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done
if [[ -z "$PYTHON" ]]; then
    error "未找到 Python 3.10+，请先安装（brew install python@3.11）"
    exit 1
fi
ok "Python $($PYTHON --version)"

# ── 检查 Python 依赖 ───────────────────────────────────────
info "检查 Python 依赖..."
if ! "$PYTHON" -c "import fastapi, uvicorn, torch" &>/dev/null; then
    warn "缺少依赖，正在安装..."
    REQUIREMENTS="$ROOT/requirements.txt"
    [[ "$(uname)" == "Darwin" ]] && REQUIREMENTS="$ROOT/requirements-mac.txt"
    "$PYTHON" -m pip install -r "$REQUIREMENTS" -q --break-system-packages 2>/dev/null || \
    "$PYTHON" -m pip install -r "$REQUIREMENTS" -q
    ok "Python 依赖安装完成"
else
    ok "Python 依赖已就绪"
fi

# ── 检查 Node / npm ────────────────────────────────────────
info "检查 Node 环境..."
if ! command -v npm &>/dev/null; then
    error "未找到 npm，请先安装 Node.js 18+"
    exit 1
fi
ok "Node $(node --version)"

# ── 安装 Node 依赖 ─────────────────────────────────────────
if [[ ! -d "$ROOT/node_modules" ]]; then
    info "安装 Node 依赖..."
    cd "$ROOT" && npm install -q
    ok "Node 依赖安装完成"
fi

# ── 端口检测：找一个可用端口 ───────────────────────────────
find_free_port() {
    local port=$1
    while lsof -iTCP:"$port" -sTCP:LISTEN -t &>/dev/null; do
        OCCUPIED_PID=$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1)
        OCCUPIED_CMD=$(ps -p "$OCCUPIED_PID" -o comm= 2>/dev/null || echo "unknown")
        warn "端口 $port 已被占用（PID $OCCUPIED_PID / $OCCUPIED_CMD）"
        port=$((port + 1))
    done
    echo "$port"
}

info "检测端口占用..."
PORT=$(find_free_port "$PORT")
ok "使用端口 $PORT"

# ── 启动 Python 后端 ───────────────────────────────────────
info "启动 Python 后端（端口 $PORT）..."
cd "$ROOT"
"$PYTHON" "$PYTHON_SERVER" --port "$PORT" &
echo $! > "$PID_FILE"
ok "Python 后端已启动（PID $(cat "$PID_FILE")）"

# ── 等待 Python 后端就绪 ───────────────────────────────────
info "等待模型加载..."
for i in $(seq 1 60); do
    STATUS=$(curl -s "http://127.0.0.1:$PORT/status" 2>/dev/null || echo '{}')
    if echo "$STATUS" | grep -qE '"loaded"\s*:\s*true'; then
        ok "模型加载完成"
        break
    fi
    if echo "$STATUS" | grep -qE '"error"\s*:\s*"[^n]'; then
        ERROR=$(echo "$STATUS" | grep -oE '"error"\s*:\s*"[^"]*"')
        error "模型加载失败：$ERROR"
        exit 1
    fi
    sleep 3
done

# ── 清理残留的 Vite 进程（端口 1420）──────────────────────
VITE_PORT=1420
if lsof -iTCP:"$VITE_PORT" -sTCP:LISTEN -t &>/dev/null; then
    OLD_PID=$(lsof -iTCP:"$VITE_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
    OLD_CMD=$(ps -p "$OLD_PID" -o comm= 2>/dev/null || echo "unknown")
    warn "端口 $VITE_PORT 被占用（PID $OLD_PID / $OLD_CMD），正在终止..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

# ── 启动 Tauri ─────────────────────────────────────────────
info "启动 Tauri 应用..."
cd "$ROOT"
SINGVC_PORT=$PORT npm run tauri dev
