"""
SingVC Modal 部署 — 将推理后端部署到 Modal 云端 GPU。

用法:
  modal run modal_app.py::download_models   # 首次：下载模型到 Volume
  modal deploy modal_app.py                 # 部署服务
"""

import modal

app = modal.App("singvc")

# 持久化模型存储
volume = modal.Volume.from_name("singvc-models", create_if_missing=True)
MODEL_DIR = "/models"

# API Key 认证
api_key_secret = modal.Secret.from_name("singvc-api-key", required_keys=["SINGVC_API_KEY"])

# GPU 容器镜像
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.3",
        "huggingface-hub>=0.28.1",
        "accelerate",
        "librosa==0.10.2",
        "scipy==1.13.1",
        "numpy==1.26.4",
        "soundfile==0.12.1",
        "pydub==0.25.1",
        "einops==0.8.0",
        "munch==4.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "python-multipart>=0.0.9",
        "hydra-core==1.3.2",
        "pyyaml",
        "python-dotenv",
        "descript-audio-codec==1.0.0",
    )
    .pip_install(
        "onnxruntime-gpu>=1.18.0",
        "protobuf>=3.20,<5",
    )
    .pip_install(
        "audio-separator[gpu]>=0.19.0",
    )
    .run_commands("python -c 'from audio_separator.separator import Separator; print(\"audio-separator OK\")'")
    .add_local_dir("modules", "/app/modules")
    .add_local_file("hf_utils.py", "/app/hf_utils.py")
    .add_local_dir("configs", "/app/configs")
)


# ─── 模型下载（一次性） ──────────────────────────────────────────────────────

@app.function(image=image, volumes={MODEL_DIR: volume}, timeout=1800)
def download_models():
    """下载所有模型到 Volume，后续启动直接挂载。"""
    from huggingface_hub import hf_hub_download, snapshot_download
    import os

    def _dl(repo, filename, subdir):
        dest = os.path.join(MODEL_DIR, subdir)
        os.makedirs(dest, exist_ok=True)
        hf_hub_download(repo, filename, local_dir=dest)
        print(f"  ✓ {repo}/{filename} → {dest}")

    print("=== 下载 Seed-VC DiT ===")
    _dl("Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "seed-vc")
    _dl("Plachta/Seed-VC",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        "seed-vc")

    print("=== 下载 CAMPPlus ===")
    _dl("funasr/campplus", "campplus_cn_common.bin", "campplus")

    print("=== 下载 BigVGAN ===")
    dest = os.path.join(MODEL_DIR, "bigvgan")
    snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x", local_dir=dest)
    print(f"  ✓ bigvgan → {dest}")

    print("=== 下载 Whisper-small ===")
    dest = os.path.join(MODEL_DIR, "whisper-small")
    snapshot_download("openai/whisper-small", local_dir=dest)
    print(f"  ✓ whisper-small → {dest}")

    print("=== 下载 RMVPE ===")
    _dl("lj1995/VoiceConversionWebUI", "rmvpe.pt", "rmvpe")

    volume.commit()
    print("\n所有模型已下载到 Volume。")


# ─── 推理辅助函数 ─────────────────────────────────────────────────────────────

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    import numpy as np
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


# ─── 推理服务 ─────────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="A10G",
    volumes={MODEL_DIR: volume},
    secrets=[api_key_secret],
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
class Inference:

    @modal.enter()
    def load_models(self):
        """容器启动时加载所有模型到 GPU。"""
        import sys
        import os
        import yaml
        import torch

        sys.path.insert(0, "/app")
        os.chdir("/app")

        self.device = torch.device("cuda")
        self.fp16 = True
        self.overlap_frame_len = 16

        # --- Seed-VC DiT ---
        from modules.commons import build_model, load_checkpoint, recursive_munch

        dit_ckpt = os.path.join(MODEL_DIR, "seed-vc",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
        dit_config = os.path.join(MODEL_DIR, "seed-vc",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")

        config_data = yaml.safe_load(open(dit_config, "r"))
        model_params = recursive_munch(config_data["model_params"])
        model_params.dit_type = "DiT"
        model = build_model(model_params, stage="DiT")

        self.hop_length = config_data["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config_data["preprocess_params"]["sr"]

        model, _, _, _ = load_checkpoint(
            model, None, dit_ckpt,
            load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.model_f0 = model

        # --- CAMPPlus ---
        from modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt = os.path.join(MODEL_DIR, "campplus", "campplus_cn_common.bin")
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
        self.campplus_model.eval().to(self.device)

        # --- BigVGAN ---
        from modules.bigvgan import bigvgan

        bigvgan_path = os.path.join(MODEL_DIR, "bigvgan")
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        self.vocoder_fn = bigvgan_model.eval().to(self.device)

        # --- Whisper-small ---
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_path = os.path.join(MODEL_DIR, "whisper-small")
        whisper_model = WhisperModel.from_pretrained(
            whisper_path, torch_dtype=torch.float16, use_safetensors=False
        ).to(self.device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt", return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(self.device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None, output_attentions=False,
                    output_hidden_states=False, return_dict=True)
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori

        self.semantic_fn = semantic_fn

        # --- Mel spectrogram ---
        from modules.audio import mel_spectrogram

        self.mel_fn_args = {
            "n_fft": config_data["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config_data["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config_data["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config_data["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": config_data["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None if config_data["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **self.mel_fn_args)

        # --- RMVPE ---
        from modules.rmvpe import RMVPE

        rmvpe_path = os.path.join(MODEL_DIR, "rmvpe", "rmvpe.pt")
        rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)
        self.f0_fn = rmvpe.infer_from_audio

        # 计算窗口参数
        self.max_context_window = self.sr // self.hop_length * 30
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length

        print(f"所有模型加载完成。device={self.device}, sr={self.sr}")

    # ─── 音色转换推理 ─────────────────────────────────────────────────────────

    def voice_conversion(self, source, target, diffusion_steps, length_adjust,
                         inference_cfg_rate, auto_f0_adjust, pitch_shift,
                         use_fp16=None):
        import torch
        import torchaudio
        import librosa
        import numpy as np
        from pydub import AudioSegment

        # 替代 @torch.no_grad() + @torch.inference_mode() 装饰器
        torch.set_grad_enabled(False)

        bitrate = "320k"
        inference_module = self.model_f0
        mel_fn = self.to_mel
        requested_amp = self.fp16 if use_fp16 is None else bool(use_fp16)
        use_amp = requested_amp and self.device.type == "cuda"

        source_audio = librosa.load(source, sr=self.sr)[0]
        ref_audio = librosa.load(target, sr=self.sr)[0]

        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_audio[:self.sr * 25]).unsqueeze(0).float().to(self.device)

        ref_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)

        # 语义特征提取（长音频分块）
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self.semantic_fn(converted_waves_16k)
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
                S_alt = self.semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
        S_ori = self.semantic_fn(ori_waves_16k)

        mel = mel_fn(source_audio.to(self.device).float())
        mel2 = mel_fn(ref_audio.to(self.device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        F0_ori = self.f0_fn(ref_waves_16k[0], thred=0.03)
        F0_alt = self.f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

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

        max_source_window = self.max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
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

            vc_wave = self.vocoder_fn(vc_target.float()).squeeze().cpu()
            if vc_wave.ndim == 1:
                vc_wave = vc_wave.unsqueeze(0)

            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                    mp3_bytes = AudioSegment(
                        output_wave_int16.tobytes(), frame_rate=self.sr,
                        sample_width=output_wave_int16.dtype.itemsize, channels=1
                    ).export(format="mp3", bitrate=bitrate).read()
                    yield mp3_bytes, (self.sr, np.concatenate(generated_wave_chunks))
                    break
                output_wave = vc_wave[0, :-self.overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=self.sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=bitrate).read()
                yield mp3_bytes, None
            elif is_last_chunk:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), self.overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=self.sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=bitrate).read()
                yield mp3_bytes, (self.sr, np.concatenate(generated_wave_chunks))
                break
            else:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-self.overlap_wave_len].cpu().numpy(), self.overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -self.overlap_wave_len:]
                processed_frames += vc_target.size(2) - self.overlap_frame_len
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave_int16.tobytes(), frame_rate=self.sr,
                    sample_width=output_wave_int16.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=bitrate).read()
                yield mp3_bytes, None

    # ─── FastAPI Web 端点 ─────────────────────────────────────────────────────

    @modal.asgi_app()
    def web(self):
        import os
        import json
        import base64
        import asyncio
        import tempfile
        import shutil
        from typing import Optional

        from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        import soundfile as sf

        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 认证中间件
        @api.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/status", "/docs", "/openapi.json") or request.method == "OPTIONS":
                return await call_next(request)
            key = request.headers.get("X-API-Key")
            expected = os.environ.get("SINGVC_API_KEY", "")
            if expected and key != expected:
                raise HTTPException(status_code=401, detail="Unauthorized")
            return await call_next(request)

        @api.get("/status")
        def status():
            return {
                "loaded": True,
                "error": None,
                "device": "cuda",
                "sr": self.sr,
                "fp16": self.fp16,
            }

        @api.post("/convert")
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
                        gen = self.voice_conversion(
                            source_path, target_path,
                            diffusion_steps, length_adjust, inference_cfg_rate,
                            auto_f0_adjust, pitch_shift, use_fp16)
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

        # ─── 音频分离 ─────────────────────────────────────────────────────────

        SEP_MODEL_DIR = os.path.join(MODEL_DIR, "uvr5_models")

        SEPARATION_MODELS = {
            "UVR-MDX-NET-Inst_HQ_3.onnx": "MDX-Net 人声分离 (快速)",
            "model_bs_roformer_ep_368_sdr_12.9628.ckpt": "BS-Roformer 人声分离 (高质量)",
            "htdemucs_ft.yaml": "Demucs 4-轨分离 (人声/鼓/贝斯/其他)",
        }

        POSTPROCESS_MODELS = {
            "denoise": "UVR-DeNoise.pth",
            "deecho": "UVR-De-Echo-Aggressive.pth",
        }

        _RESIDUAL_KEYWORDS = {"noise", "echo", "reverb", "other"}

        def _is_corrupt_model_error(exc):
            text = str(exc).lower()
            markers = ["pytorchstreamreader failed reading zip archive",
                       "failed finding central directory", "corrupt", "incomplete"]
            return any(m in text for m in markers)

        def _safe_remove_file(path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        def _stem_display_name(filename):
            name = os.path.splitext(os.path.basename(filename))[0]
            if "_(" in name:
                return name.rsplit("_(", 1)[-1].rstrip(")")
            parts = name.rsplit("_", 1)
            return parts[-1] if len(parts) > 1 else name

        def _pick_clean_stem(files):
            for f in files:
                name_lower = os.path.basename(f).lower()
                is_residual = any(
                    kw in name_lower and f"no {kw}" not in name_lower
                    for kw in _RESIDUAL_KEYWORDS)
                if not is_residual:
                    return f
            return max(files, key=lambda f: os.path.getsize(f))

        def _resolve_paths(files, output_dir):
            return [f if os.path.isabs(f) else os.path.join(output_dir, os.path.basename(f)) for f in files]

        def _run_postprocess(vocals_path, pp_type, pp_dir, output_format):
            from audio_separator.separator import Separator
            current = vocals_path
            steps = []
            if pp_type in ("denoise", "both"):
                steps.append("denoise")
            if pp_type in ("deecho", "both"):
                steps.append("deecho")
            for step in steps:
                step_dir = os.path.join(pp_dir, step)
                os.makedirs(step_dir, exist_ok=True)
                sep = Separator(log_level=30, model_file_dir=SEP_MODEL_DIR,
                                output_dir=step_dir, output_format=output_format)
                sep.load_model(model_filename=POSTPROCESS_MODELS[step])
                out_files = sep.separate(current)
                resolved = _resolve_paths(out_files, step_dir)
                current = _pick_clean_stem(resolved)
            return current

        @api.get("/separation_models")
        def separation_models():
            return {"models": [{"value": k, "label": v} for k, v in SEPARATION_MODELS.items()]}

        @api.post("/separate")
        async def separate(
            audio_file: UploadFile = File(...),
            model: str = Form("UVR-MDX-NET-Inst_HQ_3.onnx"),
            output_format: str = Form("wav"),
            postprocess: str = Form(""),
            single_stem: str = Form(""),
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
                            log_level=30, model_file_dir=SEP_MODEL_DIR,
                            output_dir=stems_dir, output_format=output_format)
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
                            should_retry = isinstance(first_exc, SystemExit) or _is_corrupt_model_error(first_exc)
                            if not should_retry:
                                raise RuntimeError(f"Separation failed: {first_exc}") from first_exc
                            _safe_remove_file(model_path)
                            try:
                                return separate_once()
                            except BaseException as retry_exc:
                                raise RuntimeError(
                                    f"Failed to load separator model '{model}'. "
                                    f"Auto re-download retry failed: {retry_exc}"
                                ) from retry_exc

                    output_files = await loop.run_in_executor(None, run_separation)
                    output_files = _resolve_paths(output_files, stems_dir)

                    vocals_file = None
                    other_files = []
                    for fpath in output_files:
                        if "vocal" in os.path.basename(fpath).lower():
                            vocals_file = fpath
                        else:
                            other_files.append(fpath)

                    if postprocess and vocals_file:
                        pp_label = {"denoise": "去噪", "deecho": "去混响", "both": "去噪 + 去混响"}.get(postprocess, "后处理")
                        yield f"data: {json.dumps({'type': 'progress', 'message': f'正在对人声进行{pp_label}…'})}\n\n"
                        pp_dir = os.path.join(tmp_dir, "pp")
                        vocals_file = await loop.run_in_executor(
                            None, _run_postprocess, vocals_file, postprocess, pp_dir, output_format)
                        final_files = other_files + [vocals_file]
                    else:
                        final_files = output_files

                    if single_stem:
                        key = single_stem.lower()
                        filtered = [f for f in final_files
                                    if key in _stem_display_name(f).lower() or key in os.path.basename(f).lower()]
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
                    import traceback
                    tb_str = traceback.format_exc()
                    err_msg = str(e) or repr(e)
                    print(f"[SEPARATE ERROR] {tb_str}", flush=True)
                    yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {err_msg}', 'traceback': tb_str})}\n\n"
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            return StreamingResponse(stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        return api
