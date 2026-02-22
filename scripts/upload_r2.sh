#!/usr/bin/env bash
# 上传模型到 Cloudflare R2
# 前置条件: 安装 rclone 并配置好 r2 remote
#   rclone config  -> 选 s3 -> endpoint: https://<ACCOUNT_ID>.r2.cloudflarestorage.com
#
# 用法: BUCKET=your-bucket-name bash scripts/upload_r2.sh

set -e

BUCKET="${BUCKET:?请设置 BUCKET 环境变量，例如: BUCKET=singvc-models bash scripts/upload_r2.sh}"
REMOTE="${R2_REMOTE:-r2}"
SRC="dist/models"

if [ ! -f "$SRC/manifest.json" ]; then
  echo "❌ 先运行 python3.10 scripts/prepare_r2_upload.py 生成 dist/models/"
  exit 1
fi

echo "上传到 ${REMOTE}:${BUCKET}/models/ ..."
rclone sync "$SRC/" "${REMOTE}:${BUCKET}/models/" \
  --progress \
  --transfers 4 \
  --s3-chunk-size 64M \
  --s3-upload-concurrency 4

echo "✅ 上传完成"
echo "公开访问地址示例: https://<your-custom-domain>/models/manifest.json"
