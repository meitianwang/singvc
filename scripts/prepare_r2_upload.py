"""
将推理所需的模型文件整理到 dist/models/ 目录，准备上传到 R2。
运行: python3.10 scripts/prepare_r2_upload.py
上传: 用 rclone 或 wrangler 把 dist/models/ 目录上传到 R2 bucket

模型清单 (manifest.json) 会一并生成，下载脚本依赖它来校验完整性。
"""
import os
import shutil
import json
import hashlib

DIST = "dist/models"
HF_CACHE = "checkpoints/hf_cache"
UVR5_DIR = "checkpoints/uvr5_models"

# R2 上的目录结构：
# models/
#   manifest.json
#   seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
#   seed-vc/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
#   campplus/campplus_cn_common.bin
#   bigvgan/config.json
#   bigvgan/bigvgan_generator.pt
#   bigvgan/*.py  (code files needed by from_pretrained)
#   whisper-small/<all needed files>
#   rmvpe/rmvpe.pt
#   uvr5/<separation model files>


def resolve_hf(repo_slug, filename):
    """Resolve a file from HF cache blob structure."""
    repo_dir = os.path.join(HF_CACHE, f"models--{repo_slug}")
    snap_dir = os.path.join(repo_dir, "snapshots")
    if not os.path.exists(snap_dir):
        return None
    revs = os.listdir(snap_dir)
    if not revs:
        return None
    snap = os.path.join(snap_dir, revs[0], filename)
    return os.path.realpath(snap) if os.path.exists(snap) else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src, dst_rel):
    dst = os.path.join(DIST, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print(f"  skip {dst_rel} (exists)")
    else:
        print(f"  copy {dst_rel} ({os.path.getsize(src) / 1e6:.1f} MB)")
        shutil.copy2(src, dst)
    return {
        "path": dst_rel,
        "size": os.path.getsize(dst),
        "sha256": sha256_file(dst),
    }


manifest = {"version": 1, "files": []}

print("=== Seed-VC DiT ===")
for fn in [
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
]:
    src = resolve_hf("Plachta--Seed-VC", fn)
    manifest["files"].append(copy(src, f"seed-vc/{fn}"))

print("=== CAMPPlus ===")
src = resolve_hf("funasr--campplus", "campplus_cn_common.bin")
manifest["files"].append(copy(src, "campplus/campplus_cn_common.bin"))

print("=== BigVGAN (inference only) ===")
bigvgan_needed = [
    "config.json", "bigvgan_generator.pt",
    "bigvgan.py", "activations.py", "env.py", "meldataset.py", "utils.py",
]
for fn in bigvgan_needed:
    src = resolve_hf("nvidia--bigvgan_v2_44khz_128band_512x", fn)
    if src:
        manifest["files"].append(copy(src, f"bigvgan/{fn}"))

# alias_free_activation subdirectories
bigvgan_snap = os.path.join(
    HF_CACHE, "models--nvidia--bigvgan_v2_44khz_128band_512x", "snapshots"
)
rev = os.listdir(bigvgan_snap)[0]
afa_base = os.path.join(bigvgan_snap, rev, "alias_free_activation")
for root, dirs, files in os.walk(afa_base):
    for fn in files:
        full = os.path.realpath(os.path.join(root, fn))
        rel = os.path.relpath(os.path.join(root, fn), os.path.join(bigvgan_snap, rev))
        manifest["files"].append(copy(full, f"bigvgan/{rel}"))

print("=== Whisper-small ===")
whisper_needed = [
    "config.json", "pytorch_model.bin", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "normalizer.json",
    "generation_config.json",
]
for fn in whisper_needed:
    src = resolve_hf("openai--whisper-small", fn)
    if src:
        manifest["files"].append(copy(src, f"whisper-small/{fn}"))

print("=== RMVPE ===")
src = resolve_hf("lj1995--VoiceConversionWebUI", "rmvpe.pt")
manifest["files"].append(copy(src, "rmvpe/rmvpe.pt"))

print("=== UVR5 separation models ===")
uvr5_files = [
    "UVR-MDX-NET-Inst_HQ_3.onnx",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "model_bs_roformer_ep_368_sdr_12.9628.yaml",
    "UVR-DeNoise.pth",
    "UVR-De-Echo-Aggressive.pth",
    "download_checks.json",
    "vr_model_data.json",
    "mdx_model_data.json",
]
for fn in uvr5_files:
    src = os.path.join(UVR5_DIR, fn)
    if os.path.exists(src):
        manifest["files"].append(copy(src, f"uvr5/{fn}"))

# Write manifest
manifest_path = os.path.join(DIST, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n✅ manifest.json written with {len(manifest['files'])} files")

total_size = sum(f["size"] for f in manifest["files"])
print(f"Total size: {total_size / 1e9:.2f} GB")
print(f"\nFiles ready in: {DIST}/")
print("Upload to R2 with:")
print(f"  rclone sync {DIST}/ r2:YOUR_BUCKET/models/")
