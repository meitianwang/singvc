"""
模型下载管理器。
从 R2 (或任意 HTTP 源) 下载推理所需的模型文件。
支持断点续传、SHA256 校验、进度显示。

用法:
  # 作为脚本直接运行，下载全部模型
  python3.10 scripts/model_manager.py

  # 在代码中使用
  from scripts.model_manager import ensure_models
  ensure_models()  # 检查并下载缺失的模型
"""
import os
import sys
import json
import hashlib
import urllib.request
import shutil
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────

# R2 公开访问基础 URL（需要用户配置）
# 优先读环境变量，否则用配置文件
_DEFAULT_BASE_URL = ""

def _load_base_url():
    url = os.environ.get("MODEL_BASE_URL", "").rstrip("/")
    if url:
        return url
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_source.json")
    cfg_path = os.path.normpath(cfg_path)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            data = json.load(f)
            return data.get("base_url", "").rstrip("/")
    return _DEFAULT_BASE_URL


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# 模型文件在本地的存放位置映射
# R2 path -> local path (relative to PROJECT_ROOT)
LOCAL_PATHS = {
    # Seed-VC DiT
    "seed-vc/": "checkpoints/seed-vc/",
    # CAMPPlus
    "campplus/": "checkpoints/campplus/",
    # BigVGAN - 需要放到 HF cache 兼容的位置，或者直接放 checkpoints
    "bigvgan/": "checkpoints/bigvgan/",
    # Whisper-small
    "whisper-small/": "checkpoints/whisper-small/",
    # RMVPE
    "rmvpe/": "checkpoints/rmvpe/",
    # UVR5 separation
    "uvr5/": "checkpoints/uvr5_models/",
}


def _local_path(r2_rel_path: str) -> str:
    """Map an R2 relative path to a local file path."""
    for prefix, local_prefix in LOCAL_PATHS.items():
        if r2_rel_path.startswith(prefix):
            suffix = r2_rel_path[len(prefix):]
            return os.path.join(PROJECT_ROOT, local_prefix, suffix)
    return os.path.join(PROJECT_ROOT, "checkpoints", r2_rel_path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: str, expected_size: int = 0):
    """Download a file with progress display and resume support."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"

    # Check if partial download exists
    existing_size = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        raise RuntimeError(f"下载失败: {url}\n{e}")

    # If server doesn't support range, start over
    if resp.status == 200 and existing_size > 0:
        existing_size = 0

    total = expected_size or int(resp.headers.get("Content-Length", 0)) + existing_size
    downloaded = existing_size
    name = os.path.basename(dest)

    mode = "ab" if existing_size > 0 else "wb"
    with open(tmp, mode) as f:
        while True:
            chunk = resp.read(1 << 20)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                bar_len = 30
                filled = int(bar_len * downloaded / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                size_mb = downloaded / 1e6
                total_mb = total / 1e6
                sys.stdout.write(f"\r  {name}: {bar} {pct:5.1f}% ({size_mb:.0f}/{total_mb:.0f} MB)")
                sys.stdout.flush()

    sys.stdout.write("\n")
    shutil.move(tmp, dest)


def check_and_download(manifest: dict, base_url: str, force: bool = False) -> list:
    """Check which files are missing or corrupted, download them.
    Returns list of downloaded file paths.
    """
    downloaded = []
    files = manifest.get("files", [])
    total = len(files)

    for i, entry in enumerate(files, 1):
        r2_path = entry["path"]
        expected_sha = entry.get("sha256", "")
        expected_size = entry.get("size", 0)
        local = _local_path(r2_path)

        # Check if file exists and is valid
        if not force and os.path.exists(local):
            if expected_size and os.path.getsize(local) == expected_size:
                # Size matches, skip SHA check for speed (large files)
                continue
            # Size mismatch, re-download

        url = f"{base_url}/{r2_path}"
        print(f"[{i}/{total}] 下载 {r2_path}")
        download_file(url, local, expected_size)

        # Verify
        if expected_sha:
            actual = sha256_file(local)
            if actual != expected_sha:
                os.remove(local)
                raise RuntimeError(f"SHA256 校验失败: {r2_path}\n  期望: {expected_sha}\n  实际: {actual}")

        downloaded.append(local)

    return downloaded


def ensure_models(force: bool = False) -> bool:
    """Ensure all models are downloaded. Returns True if all OK."""
    base_url = _load_base_url()
    if not base_url:
        print("⚠ 未配置模型下载地址 (MODEL_BASE_URL 或 configs/model_source.json)")
        print("  跳过自动下载，将使用 HuggingFace 默认下载")
        return False

    manifest_url = f"{base_url}/manifest.json"
    print(f"检查模型清单: {manifest_url}")

    try:
        resp = urllib.request.urlopen(manifest_url, timeout=10)
        manifest = json.loads(resp.read())
    except Exception as e:
        print(f"⚠ 无法获取模型清单: {e}")
        return False

    downloaded = check_and_download(manifest, base_url, force=force)
    if downloaded:
        print(f"✅ 下载了 {len(downloaded)} 个文件")
    else:
        print("✅ 所有模型已就绪")
    return True


