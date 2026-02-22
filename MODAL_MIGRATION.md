# Modal 迁移记录

## 已完成

### 1. 创建 Modal 后端 (`modal_app.py`)
- 定义 GPU 容器镜像（Python 3.10 + CUDA 12.1 + PyTorch 2.4）
- `download_models` 函数：一次性下载所有模型到 Modal Volume
- `Inference` 类：A10G GPU，`@modal.enter()` 加载模型，`@modal.asgi_app()` 暴露 FastAPI
- 完整搬运 `voice_conversion` 推理管线和 `/separate` 音频分离端点
- SSE 流式输出格式与原版完全一致
- `X-API-Key` 认证中间件

### 2. 前端适配 (`src/App.tsx`)
- 移除 `@tauri-apps/api/core` 的 `invoke("get_server_port")` 调用
- 所有 API 地址改为 `VITE_BACKEND_URL` 环境变量
- `extractVocals` 和 `/convert` 请求附带 `X-API-Key` header
- `/status` 轮询间隔改为 3 秒（适配 Modal 冷启动）

### 3. Tauri 精简 (`src-tauri/src/lib.rs`)
- 移除全部 Python 进程管理代码（149 行 → 7 行）
- Tauri 变为纯 WebView 壳

### 4. 环境配置
- `.env`：配置 `VITE_BACKEND_URL` 和 `VITE_API_KEY`（已 gitignore）
- `src/vite-env.d.ts`：Vite 环境变量类型声明
- Modal Secret `singvc-api-key`：API 认证密钥

### 5. 部署
- Modal token 已配置
- 模型已下载到 Volume `singvc-models`
- 服务已部署：`https://meitianwang1--singvc-inference-web.modal.run`

## 待完成

### 网络问题
- [ ] 公司网络拦截 `*.modal.run` 域名，需申请加白后才能从本地访问

### 端到端验证
- [ ] 域名放通后重启 `npm run dev`，验证 `/status` 轮询 → READY
- [ ] 上传音频 → 人声分离 → 音色转换 → SSE 实时播放 → WAV 下载
- [ ] `npm run tauri dev` 验证 Tauri 壳正常工作

### 可选优化
- [ ] 冷启动优化：考虑 `keep_warm=1` 保持常驻容器（有成本）
- [ ] 音频分离模型预下载到 Volume（当前首次调用时 `audio-separator` 自动下载）
- [ ] 前端添加冷启动提示 UI（当前只显示 LOADING）
- [ ] 清理不再需要的文件：`start.sh`、`python/server.py`、`requirements*.txt`、`scripts/model_manager.py`
