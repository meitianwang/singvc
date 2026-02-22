"""
下载 Seed-VC 项目所需的全部模型。

优先从 R2 下载（如果配置了 MODEL_BASE_URL 或 configs/model_source.json），
否则从 HuggingFace 下载。

用法:
  # 从 R2 下载（需要先配置 configs/model_source.json 中的 base_url）
  python3.10 download_models.py

  # 从 HuggingFace 镜像下载（回退方式）
  HF_ENDPOINT=https://hf-mirror.com python3.10 download_models.py --hf
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))


def download_from_r2():
    from model_manager import ensure_models
    return ensure_models()


def download_from_hf():
    os.makedirs("./checkpoints/hf_cache", exist_ok=True)
    from huggingface_hub import hf_hub_download, snapshot_download

    CACHE_DIR = "./checkpoints/hf_cache"

    def dl(repo_id, filename, config_filename=None):
        print(f"  ↓ {repo_id} / {filename}")
        p = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=CACHE_DIR)
        if config_filename:
            print(f"  ↓ {repo_id} / {config_filename}")
            hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=CACHE_DIR)
        return p

    print("=" * 60)
    print("1/5  Seed-VC DiT 模型")
    print("=" * 60)
    dl("Plachta/Seed-VC",
       "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
       "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")

    print("\n" + "=" * 60)
    print("2/5  CAMPPlus 说话人编码器")
    print("=" * 60)
    dl("funasr/campplus", "campplus_cn_common.bin")

    print("\n" + "=" * 60)
    print("3/5  BigVGAN vocoder")
    print("=" * 60)
    snapshot_download(repo_id="nvidia/bigvgan_v2_44khz_128band_512x", cache_dir=CACHE_DIR)

    print("\n" + "=" * 60)
    print("4/5  OpenAI Whisper-small")
    print("=" * 60)
    snapshot_download(repo_id="openai/whisper-small", cache_dir=CACHE_DIR)

    print("\n" + "=" * 60)
    print("5/5  RMVPE F0 提取器")
    print("=" * 60)
    dl("lj1995/VoiceConversionWebUI", "rmvpe.pt")

    print("\n" + "=" * 60)
    print("6/8  音频分离模型")
    print("=" * 60)
    from audio_separator.separator import Separator
    SEP_DIR = os.path.abspath("./checkpoints/uvr5_models")
    os.makedirs(SEP_DIR, exist_ok=True)

    for model_name in [
        "UVR-MDX-NET-Inst_HQ_3.onnx",
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "UVR-DeNoise.pth",
        "UVR-De-Echo-Aggressive.pth",
    ]:
        print(f"  ↓ {model_name}")
        s = Separator(log_level=30, model_file_dir=SEP_DIR)
        s.load_model(model_filename=model_name)

    print("\n✅ 所有模型下载完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", action="store_true", help="强制从 HuggingFace 下载")
    args = parser.parse_args()

    if args.hf:
        download_from_hf()
    else:
        ok = download_from_r2()
        if not ok:
            print("R2 未配置，回退到 HuggingFace 下载...")
            download_from_hf()
