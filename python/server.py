import os
import sys
import base64
import json
import tempfile
import shutil
import argparse
import asyncio
from typing import Optional

os.environ['HF_HUB_CACHE'] = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'hf_cache')

# Add project root to path so modules can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# ─── Global state ────────────────────────────────────────────────────────────

device = None
fp16 = True
sr = None
hop_length = None
overlap_frame_len = 16
overlap_wave_len = None
max_context_window = None

model_f0 = None
semantic_fn = None
vocoder_fn = None
campplus_model = None
to_mel_f0 = None
mel_fn_args = None
f0_fn = None

model_loaded = False
model_error = None

# ─── Inference helpers (from app_svc.py, unchanged) ──────────────────────────

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust,
                     inference_cfg_rate, auto_f0_adjust, pitch_shift,
                     use_fp16: Optional[bool] = None):
    import torchaudio
    import librosa
    from pydub import AudioSegment

    bitrate = "320k"
    inference_module = model_f0
    mel_fn = to_mel_f0
    requested_amp = fp16 if use_fp16 is None else bool(use_fp16)
    use_amp = requested_amp and device.type in {"cuda", "mps"}

    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k, num_mel_bins=80,
                                               dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    F0_ori = f0_fn(ref_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    if device.type == "mps":
        F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
    else:
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]

    log_f0_alt = torch.log(F0_alt + 1e-5)
    voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
    voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
    median_log_f0_ori = torch.median(voiced_log_f0_ori)
    median_log_f0_alt = torch.median(voiced_log_f0_alt)

    shifted_log_f0_alt = log_f0_alt.clone()
    if auto_f0_adjust:
        shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
    shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    if pitch_shift != 0:
        shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

    cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)
    interpolated_shifted_f0_alt = torch.nn.functional.interpolate(
        shifted_f0_alt.unsqueeze(1), size=cond.size(1), mode='nearest').squeeze(1)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        chunk_f0 = interpolated_shifted_f0_alt[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, mel2.size(-1):]
        else:
            vc_target = inference_module.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze().cpu()
        if vc_wave.ndim == 1:
            vc_wave = vc_wave.unsqueeze(0)

        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=bitrate).read()
                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave_int16.tobytes(), frame_rate=sr,
                sample_width=output_wave_int16.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, None
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave_int16.tobytes(), frame_rate=sr,
                sample_width=output_wave_int16.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave_int16.tobytes(), frame_rate=sr,
                sample_width=output_wave_int16.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, None


# ─── Model loading ────────────────────────────────────────────────────────────

def load_models(checkpoint=None, config=None):
    global model_f0, semantic_fn, vocoder_fn, campplus_model, to_mel_f0
    global mel_fn_args, f0_fn, sr, hop_length, overlap_wave_len, max_context_window
    global model_loaded, model_error

    try:
        import yaml
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from hf_utils import load_custom_model_from_hf

        if checkpoint is None or checkpoint == "":
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        else:
            dit_checkpoint_path = checkpoint
            dit_config_path = config

        config_data = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config_data["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        hop_length = config_data["preprocess_params"]["spect_params"]["hop_length"]
        sr = config_data["preprocess_params"]["sr"]

        model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        from modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model.eval()
        campplus_model.to(device)

        vocoder_type = model_params.vocoder.type
        if vocoder_type == 'bigvgan':
            from modules.bigvgan import bigvgan
            # 优先使用 R2 下载的本地模型
            r2_bigvgan = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'bigvgan')
            if os.path.exists(os.path.join(r2_bigvgan, 'config.json')):
                bigvgan_name = os.path.abspath(r2_bigvgan)
            else:
                bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            bigvgan_model.remove_weight_norm()
            vocoder_fn = bigvgan_model.eval().to(device)
        elif vocoder_type == 'hifigan':
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            vocoder_fn = hift_gen.eval().to(device)
        else:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")

        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        # 优先使用 R2 下载的本地模型
        r2_whisper = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'whisper-small')
        if os.path.exists(os.path.join(r2_whisper, 'config.json')):
            whisper_local = os.path.abspath(r2_whisper)
        else:
            whisper_local = whisper_name
        whisper_model = WhisperModel.from_pretrained(whisper_local, torch_dtype=torch.float16, use_safetensors=False).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_local)

        def semantic_fn_inner(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                    return_tensors="pt", return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None, output_attentions=False,
                    output_hidden_states=False, return_dict=True)
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori

        semantic_fn = semantic_fn_inner

        mel_fn_args = {
            "n_fft": config_data['preprocess_params']['spect_params']['n_fft'],
            "win_size": config_data['preprocess_params']['spect_params']['win_length'],
            "hop_size": config_data['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config_data['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": config_data['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if config_data['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        from modules.audio import mel_spectrogram
        to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args)

        from modules.rmvpe import RMVPE
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        rmvpe = RMVPE(model_path, is_half=False, device=device)
        f0_fn = rmvpe.infer_from_audio

        model_f0 = model
        max_context_window = sr // hop_length * 30
        overlap_wave_len = overlap_frame_len * hop_length
        model_loaded = True
        print(f"Models loaded successfully. device={device}, sr={sr}")
    except Exception as e:
        model_error = str(e)
        print(f"Failed to load models: {e}")
        raise


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def status():
    return {
        "loaded": model_loaded,
        "error": model_error,
        "device": str(device) if device else None,
        "sr": sr,
        "fp16": fp16,
    }


@app.get("/models/status")
def models_status():
    """检查各模型文件是否存在。"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    checks = {
        "seed_vc": os.path.exists(os.path.join(project_root, "checkpoints/seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")),
        "campplus": os.path.exists(os.path.join(project_root, "checkpoints/campplus/campplus_cn_common.bin")),
        "bigvgan": os.path.exists(os.path.join(project_root, "checkpoints/bigvgan/bigvgan_generator.pt")),
        "whisper": os.path.exists(os.path.join(project_root, "checkpoints/whisper-small/pytorch_model.bin")),
        "rmvpe": os.path.exists(os.path.join(project_root, "checkpoints/rmvpe/rmvpe.pt")),
        "uvr5_mdx": os.path.exists(os.path.join(project_root, "checkpoints/uvr5_models/UVR-MDX-NET-Inst_HQ_3.onnx")),
        "uvr5_roformer": os.path.exists(os.path.join(project_root, "checkpoints/uvr5_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt")),
        "uvr5_denoise": os.path.exists(os.path.join(project_root, "checkpoints/uvr5_models/UVR-DeNoise.pth")),
        "uvr5_deecho": os.path.exists(os.path.join(project_root, "checkpoints/uvr5_models/UVR-De-Echo-Aggressive.pth")),
    }
    # Also check HF cache as fallback
    hf_cache = os.path.join(project_root, "checkpoints/hf_cache")
    hf_fallbacks = {
        "seed_vc": os.path.exists(os.path.join(hf_cache, "models--Plachta--Seed-VC")),
        "campplus": os.path.exists(os.path.join(hf_cache, "models--funasr--campplus")),
        "bigvgan": os.path.exists(os.path.join(hf_cache, "models--nvidia--bigvgan_v2_44khz_128band_512x")),
        "whisper": os.path.exists(os.path.join(hf_cache, "models--openai--whisper-small")),
        "rmvpe": os.path.exists(os.path.join(hf_cache, "models--lj1995--VoiceConversionWebUI")),
    }
    combined = {}
    for k, v in checks.items():
        combined[k] = v or hf_fallbacks.get(k, False)
    all_vc_ready = all(combined.get(k, False) for k in ["seed_vc", "campplus", "bigvgan", "whisper", "rmvpe"])
    return {
        "models": combined,
        "vc_ready": all_vc_ready,
        "all_ready": all(combined.values()),
    }


@app.post("/models/download")
async def download_models_endpoint():
    """触发从 R2 下载缺失的模型。"""
    async def stream():
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
            from model_manager import ensure_models
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(None, ensure_models)
            if ok:
                yield f"data: {json.dumps({'type': 'done', 'message': '所有模型已就绪'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': '未配置模型下载地址'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/convert")
async def convert(
    source_file: UploadFile = File(...),
    target_file: UploadFile = File(...),
    diffusion_steps: int = Form(40),
    length_adjust: float = Form(1.0),
    inference_cfg_rate: float = Form(0.7),
    auto_f0_adjust: bool = Form(False),
    pitch_shift: int = Form(0),
    use_fp16: Optional[bool] = Form(None),
):
    if not model_loaded:
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Models not loaded yet'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    tmp_dir = tempfile.mkdtemp()
    try:
        source_ext = os.path.splitext(source_file.filename)[1] or ".wav"
        target_ext = os.path.splitext(target_file.filename)[1] or ".wav"
        source_path = os.path.join(tmp_dir, f"source{source_ext}")
        target_path = os.path.join(tmp_dir, f"target{target_ext}")

        with open(source_path, "wb") as f:
            f.write(await source_file.read())
        with open(target_path, "wb") as f:
            f.write(await target_file.read())

        async def stream():
            try:
                loop = asyncio.get_event_loop()
                gen = voice_conversion(
                    source_path, target_path,
                    diffusion_steps, length_adjust, inference_cfg_rate,
                    auto_f0_adjust, pitch_shift, use_fp16
                )
                for mp3_bytes, final in await loop.run_in_executor(None, lambda: list(gen)):
                    audio_b64 = base64.b64encode(mp3_bytes).decode()
                    if final is not None:
                        sample_rate, waveform = final
                        wav_path = os.path.join(tmp_dir, "output.wav")
                        sf.write(wav_path, waveform, sample_rate)
                        with open(wav_path, "rb") as wf:
                            wav_b64 = base64.b64encode(wf.read()).decode()
                        yield f"data: {json.dumps({'type': 'chunk', 'audio': audio_b64})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'audio': wav_b64})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'chunk', 'audio': audio_b64})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")


# ─── Audio Separation ─────────────────────────────────────────────────────────

# Models cache dir (separate from SVC model checkpoints)
SEP_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'uvr5_models'))

SEPARATION_MODELS = {
    "UVR-MDX-NET-Inst_HQ_3.onnx": "MDX-Net 人声分离 (快速)",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt": "BS-Roformer 人声分离 (高质量)",
    "htdemucs_ft.yaml": "Demucs 4-轨分离 (人声/鼓/贝斯/其他)",
}

# Post-process models applied to the vocals stem after primary separation
POSTPROCESS_MODELS = {
    "denoise": "UVR-DeNoise.pth",
    "deecho":  "UVR-De-Echo-Aggressive.pth",
}

# Keywords that identify the *residual* (noise/echo) stem from post-processors
_RESIDUAL_KEYWORDS = {"noise", "echo", "reverb", "other"}


def _is_corrupt_model_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = [
        "pytorchstreamreader failed reading zip archive",
        "failed finding central directory",
        "corrupt",
        "incomplete",
    ]
    return any(marker in text for marker in markers)


def _safe_remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _stem_display_name(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    if "_(" in name:
        return name.rsplit("_(", 1)[-1].rstrip(")")
    parts = name.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else name


def _pick_clean_stem(files: list[str]) -> str:
    """From post-processor output files, pick the clean (non-residual) stem.

    UVR-DeNoise outputs '(No Noise)' (clean) and '(Noise)' (residual).
    UVR-De-Echo outputs '(No Echo)' (clean) and '(Echo)' (residual).
    A keyword like 'noise' preceded by 'no ' means clean, not residual.
    """
    for f in files:
        name_lower = os.path.basename(f).lower()
        # A stem is residual only when the keyword appears WITHOUT a "no " prefix
        is_residual = any(
            kw in name_lower and f"no {kw}" not in name_lower
            for kw in _RESIDUAL_KEYWORDS
        )
        if not is_residual:
            return f
    # fallback: largest file is usually the clean signal
    return max(files, key=lambda f: os.path.getsize(f))


def _resolve_paths(files: list[str], output_dir: str) -> list[str]:
    """Resolve separator output paths (may be relative) to absolute paths."""
    return [f if os.path.isabs(f) else os.path.join(output_dir, os.path.basename(f)) for f in files]


def _run_postprocess(vocals_path: str, pp_type: str, pp_dir: str, output_format: str) -> str:
    """
    Apply denoise / de-echo (or both) to a vocals file.
    Returns the absolute path of the cleaned vocals file.
    """
    from audio_separator.separator import Separator

    current = vocals_path  # already an absolute path
    steps = []
    if pp_type in ("denoise", "both"):
        steps.append("denoise")
    if pp_type in ("deecho", "both"):
        steps.append("deecho")

    for step in steps:
        step_dir = os.path.join(pp_dir, step)
        os.makedirs(step_dir, exist_ok=True)
        sep = Separator(
            log_level=30,
            model_file_dir=SEP_MODEL_DIR,
            output_dir=step_dir,
            output_format=output_format,
        )
        sep.load_model(model_filename=POSTPROCESS_MODELS[step])
        out_files = sep.separate(current)
        resolved = _resolve_paths(out_files, step_dir)
        current = _pick_clean_stem(resolved)

    return current


@app.get("/separation_models")
def separation_models():
    return {"models": [{"value": k, "label": v} for k, v in SEPARATION_MODELS.items()]}


@app.post("/separate")
async def separate(
    audio_file: UploadFile = File(...),
    model: str = Form("UVR-MDX-NET-Inst_HQ_3.onnx"),
    output_format: str = Form("wav"),
    postprocess: str = Form(""),  # "", "denoise", "deecho", "both"
    single_stem: str = Form(""),  # e.g. "Vocals" to get only one stem
):
    tmp_dir = tempfile.mkdtemp()
    stems_dir = os.path.join(tmp_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    os.makedirs(SEP_MODEL_DIR, exist_ok=True)

    audio_ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    audio_path = os.path.join(tmp_dir, f"input{audio_ext}")

    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    async def stream():
        try:
            from audio_separator.separator import Separator
            loop = asyncio.get_event_loop()

            def run_separation():
                sep_kwargs = dict(
                    log_level=30,
                    model_file_dir=SEP_MODEL_DIR,
                    output_dir=stems_dir,
                    output_format=output_format,
                )
                if single_stem:
                    sep_kwargs["output_single_stem"] = single_stem
                model_path = os.path.join(SEP_MODEL_DIR, model)

                def separate_once():
                    separator = Separator(**sep_kwargs)
                    separator.load_model(model_filename=model)
                    return separator.separate(audio_path)

                try:
                    return separate_once()
                except BaseException as first_exc:
                    # audio-separator may call sys.exit(1) on model load failure,
                    # which appears here as SystemExit and would otherwise kill the stream.
                    should_retry = isinstance(first_exc, SystemExit) or _is_corrupt_model_error(first_exc)
                    if not should_retry:
                        raise RuntimeError(f"Separation failed: {first_exc}") from first_exc

                    _safe_remove_file(model_path)
                    try:
                        return separate_once()
                    except BaseException as retry_exc:
                        raise RuntimeError(
                            f"Failed to load separator model '{model}'. "
                            f"The local model file may be corrupted and auto re-download retry failed: {retry_exc}"
                        ) from retry_exc

            output_files = await loop.run_in_executor(None, run_separation)

            # Resolve to absolute paths immediately (separator may return relative paths)
            output_files = _resolve_paths(output_files, stems_dir)

            # Identify vocals file for optional post-processing
            vocals_file = None
            other_files = []
            for fpath in output_files:
                if "vocal" in os.path.basename(fpath).lower():
                    vocals_file = fpath
                else:
                    other_files.append(fpath)

            if postprocess and vocals_file:
                pp_label = {"denoise": "去噪", "deecho": "去混响", "both": "去噪 + 去混响"}.get(postprocess, "后处理")
                yield f"data: {json.dumps({'type': 'progress', 'message': f'正在对人声进行{pp_label}…（首次运行将自动下载后处理模型）'})}\n\n"
                pp_dir = os.path.join(tmp_dir, "pp")
                vocals_file = await loop.run_in_executor(
                    None, _run_postprocess, vocals_file, postprocess, pp_dir, output_format
                )
                final_files = other_files + [vocals_file]
            else:
                final_files = output_files

            # Separator outputs can still include extra stems in some models;
            # when single_stem is requested, filter again before streaming.
            if single_stem:
                key = single_stem.lower()
                filtered = [
                    f for f in final_files
                    if key in _stem_display_name(f).lower() or key in os.path.basename(f).lower()
                ]
                if filtered:
                    final_files = filtered

            for fpath in final_files:
                stem_name = _stem_display_name(fpath)
                fname = os.path.basename(fpath)
                with open(fpath, "rb") as sf_:
                    audio_b64 = base64.b64encode(sf_.read()).decode()
                yield f"data: {json.dumps({'type': 'stem', 'name': stem_name, 'filename': fname, 'audio': audio_b64})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except BaseException as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fp16", type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    fp16 = args.fp16
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # MPS fp16 can be numerically unstable for singing VC; prefer fp32 unless overridden per request.
    if device.type == "mps" and fp16:
        print("MPS detected, disabling global fp16 by default for quality (can override via use_fp16=true).")
        fp16 = False

    # 尝试从 R2 下载缺失的模型（如果配置了下载地址）
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from model_manager import ensure_models
        ensure_models()
    except Exception as e:
        print(f"Model download check skipped: {e}")

    # Load models in background so /status can be polled immediately
    import threading
    threading.Thread(target=load_models, args=(args.checkpoint, args.config), daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
