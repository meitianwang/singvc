# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

SingVC 是基于 Seed-VC 的歌声音色转换桌面应用。用户上传源音频和参考音频，后端通过扩散模型将源音频音色转换为参考音色，保留原始旋律和歌词。

## 技术栈

- **桌面壳**: Tauri 2 (Rust)
- **前端**: React 18 + TypeScript + Vite 5
- **后端**: Python 3.10+ FastAPI + uvicorn
- **ML 推理**: PyTorch, Seed-VC DiT, Whisper-small, BigVGAN, CAMPPlus, RMVPE
- **音频分离**: audio-separator (UVR-MDX-NET, BS-Roformer)

## 常用命令

```bash
# 一键启动（推荐，自动处理依赖、端口检测、模型加载）
./start.sh

# 单独启动 Python 后端
python python/server.py --port 18888

# 单独启动前端开发服务器（端口 1420）
npm run dev

# Tauri 开发模式（需先启动 Python 后端）
SINGVC_PORT=18888 npm run tauri dev

# 构建
npm run build          # 前端静态文件
npm run tauri build    # Tauri 打包

# Python 依赖
pip install -r requirements-mac.txt      # Mac Apple Silicon
pip install -r requirements.txt          # Windows / Linux
```

## 架构

### 进程通信

```
Tauri (Rust) → 启动 Python 子进程 (python/server.py --port PORT)
             → CloseRequested 时 kill Python 进程
             → Tauri command: get_server_port 暴露端口给前端

React 前端 → invoke("get_server_port") 获取端口
           → HTTP 直连 Python FastAPI (localhost:PORT)
           → 所有耗时操作通过 SSE 流式返回
```

端口通过环境变量 `SINGVC_PORT` 传入 Rust，前端不经过 Rust 中转直接请求 Python 后端。

### Python 后端 API (`python/server.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/status` | GET | 模型加载状态、设备类型、采样率 |
| `/models/status` | GET | 各模型文件是否存在 |
| `/models/download` | POST | 从 R2 下载缺失模型（SSE 流） |
| `/convert` | POST | 音色转换（SSE 流：MP3 块 + 最终 WAV base64） |
| `/separation_models` | GET | 可用音频分离模型列表 |
| `/separate` | POST | 音频人声分离（SSE 流：各音轨 base64） |

### 前端状态机 (`App.tsx`)

`loading → ready → converting → done / error`

转换流程：若启用人声分离先调 `/separate` 提取人声（带缓存），再 POST `/convert`，逐块接收 MP3 实时播放，收到 `done` 后保存 WAV 并追加到历史记录。

### 模型加载策略

1. 启动时后台线程加载，`/status` 立即可响应（`loaded: false`）
2. 优先 `checkpoints/` 本地文件（R2 下载）
3. 回退 HuggingFace Hub（缓存至 `checkpoints/hf_cache/`）
4. 可通过 `configs/model_source.json` 或 `MODEL_BASE_URL` 环境变量配置自定义下载源

## 关键配置

- Vite 开发服务器固定端口 **1420**（`strictPort: true`）
- Python 后端默认端口 **18888**，`start.sh` 自动检测冲突递增
- 设备自动检测：CUDA → MPS (Apple Silicon) → CPU
- MPS 默认关闭 FP16（数值稳定性），前端可通过 `use_fp16` 参数覆盖
- HuggingFace 镜像：`start.sh` 默认设置 `HF_ENDPOINT=https://hf-mirror.com`
- `checkpoints/`、`dist/`、`src-tauri/target/` 已 gitignore
