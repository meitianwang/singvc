import os
from huggingface_hub import hf_hub_download

# R2 下载的本地模型路径 (优先使用)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_R2_MODEL_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")

# R2 本地路径映射: (repo_id, filename) -> local relative path
_R2_LOCAL_MAP = {
    ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"):
        "seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"):
        "seed-vc/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
    ("funasr/campplus", "campplus_cn_common.bin"):
        "campplus/campplus_cn_common.bin",
    ("lj1995/VoiceConversionWebUI", "rmvpe.pt"):
        "rmvpe/rmvpe.pt",
    ("FunAudioLLM/CosyVoice-300M", "hift.pt"):
        None,  # not used in current config
}


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    # 优先检查 R2 下载的本地文件
    local_key = (repo_id, model_filename)
    if local_key in _R2_LOCAL_MAP and _R2_LOCAL_MAP[local_key]:
        local_path = os.path.join(_R2_MODEL_DIR, _R2_LOCAL_MAP[local_key])
        if os.path.exists(local_path):
            if config_filename is None:
                return local_path
            config_key = (repo_id, config_filename)
            if config_key in _R2_LOCAL_MAP and _R2_LOCAL_MAP[config_key]:
                config_local = os.path.join(_R2_MODEL_DIR, _R2_LOCAL_MAP[config_key])
                if os.path.exists(config_local):
                    return local_path, config_local

    # 回退到 HuggingFace 下载
    os.makedirs("./checkpoints/hf_cache", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints/hf_cache")
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="./checkpoints/hf_cache")
    return model_path, config_path
