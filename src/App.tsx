import { useEffect, useState, useRef, useCallback } from "react";
import AudioUploadWithSep from "./components/AudioUploadWithSep";
import ParamSliders, { type SepSettings } from "./components/ParamSliders";
import AudioPlayer from "./components/AudioPlayer";
import HistoryPanel, { HistoryItem } from "./components/HistoryPanel";
import "./App.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:18888";
const API_KEY = import.meta.env.VITE_API_KEY || "";

export interface Params {
  diffusion_steps: number;
  length_adjust: number;
  inference_cfg_rate: number;
  auto_f0_adjust: boolean;
  pitch_shift: number;
  use_fp16: boolean;
}

type Status = "loading" | "ready" | "converting" | "done" | "error";

/** Call /separate with single_stem=Vocals, return the extracted vocals as a File + base64 audio. */
async function extractVocals(
  file: File, model: string, postprocess: string
): Promise<{ file: File; audioB64: string }> {
  const formData = new FormData();
  formData.append("audio_file", file);
  formData.append("model", model);
  formData.append("output_format", "wav");
  formData.append("postprocess", postprocess);
  formData.append("single_stem", "Vocals");

  const headers: Record<string, string> = {};
  if (API_KEY) headers["X-API-Key"] = API_KEY;

  const res = await fetch(`${BACKEND_URL}/separate`, {
    method: "POST",
    body: formData,
    headers,
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const stems: Array<{ name?: string; filename?: string; audio: string }> = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.type === "stem") {
        stems.push(payload);
      } else if (payload.type === "error") {
        throw new Error(payload.message);
      }
    }
  }

  if (stems.length === 0) throw new Error("人声提取失败：未收到音频数据");
  const vocalsStem =
    stems.find((s) => {
      const name = (s.name ?? "").toLowerCase();
      const filename = (s.filename ?? "").toLowerCase();
      return (
        name.includes("vocal") ||
        filename.includes("vocal") ||
        name.includes("人声") ||
        filename.includes("人声")
      );
    }) ?? stems[0];
  const vocalsB64 = vocalsStem.audio;
  const vocalsFilename = vocalsStem.filename ?? "vocals.wav";

  const bytes = Uint8Array.from(atob(vocalsB64), (c) => c.charCodeAt(0));
  return {
    file: new File([bytes], vocalsFilename, { type: "audio/wav" }),
    audioB64: vocalsB64,
  };
}

export default function App() {
  const [serverReady, setServerReady] = useState(false);

  // Audio files
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);

  // Per-file separation options (unified model/postprocess)
  const [sep, setSep] = useState<SepSettings>({
    sourceEnabled: true,
    targetEnabled: false,
    model: "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    postprocess: "both",
  });

  // Intermediate separation results (base64 WAV)
  const [sourceVocalsB64, setSourceVocalsB64] = useState<string | null>(null);
  const [targetVocalsB64, setTargetVocalsB64] = useState<string | null>(null);

  // Cache: remember which file+model+pp combo produced the cached vocals
  const sourceSepCache = useRef<{ key: string; file: File; audioB64: string } | null>(null);
  const targetSepCache = useRef<{ key: string; file: File; audioB64: string } | null>(null);

  // Conversion params
  const [params, setParams] = useState<Params>({
    diffusion_steps: 50,
    length_adjust: 1.0,
    inference_cfg_rate: 0.7,
    auto_f0_adjust: false,
    pitch_shift: 0,
    use_fp16: true,
  });

  // Status
  const [status, setStatus] = useState<Status>("loading");
  const [progressMsg, setProgressMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [mp3Chunks, setMp3Chunks] = useState<string[]>([]);
  const [finalWav, setFinalWav] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // History
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const handleClearHistory = useCallback(() => setHistory([]), []);

  useEffect(() => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/status`);
        const data = await res.json();
        if (data.loaded) {
          setServerReady(true);
          setStatus("ready");
          clearInterval(pollRef.current!);
        } else if (data.error) {
          setStatus("error");
          setErrorMsg(data.error);
          clearInterval(pollRef.current!);
        }
      } catch {
        // Modal 冷启动中，继续轮询
      }
    }, 3000);
    return () => clearInterval(pollRef.current!);
  }, []);

  async function handleConvert() {
    if (!sourceFile || !targetFile || !serverReady) return;
    setStatus("converting");
    setMp3Chunks([]);
    setFinalWav(null);
    setErrorMsg("");

    try {
      let actualSource = sourceFile;
      let actualTarget = targetFile;

      if (sep.sourceEnabled) {
        const cacheKey = `${sourceFile.name}|${sourceFile.size}|${sep.model}|${sep.postprocess}`;
        if (sourceSepCache.current?.key === cacheKey) {
          actualSource = sourceSepCache.current.file;
          setSourceVocalsB64(sourceSepCache.current.audioB64);
        } else {
          setProgressMsg("正在提取原始音轨人声…");
          const result = await extractVocals(sourceFile, sep.model, sep.postprocess);
          actualSource = result.file;
          setSourceVocalsB64(result.audioB64);
          sourceSepCache.current = { key: cacheKey, file: result.file, audioB64: result.audioB64 };
        }
      } else {
        setSourceVocalsB64(null);
      }

      if (sep.targetEnabled) {
        const cacheKey = `${targetFile.name}|${targetFile.size}|${sep.model}|${sep.postprocess}`;
        if (targetSepCache.current?.key === cacheKey) {
          actualTarget = targetSepCache.current.file;
          setTargetVocalsB64(targetSepCache.current.audioB64);
        } else {
          setProgressMsg("正在提取目标音色人声…");
          const result = await extractVocals(targetFile, sep.model, sep.postprocess);
          actualTarget = result.file;
          setTargetVocalsB64(result.audioB64);
          targetSepCache.current = { key: cacheKey, file: result.file, audioB64: result.audioB64 };
        }
      } else {
        setTargetVocalsB64(null);
      }

      setProgressMsg("调音处理中…");

      const formData = new FormData();
      formData.append("source_file", actualSource);
      formData.append("target_file", actualTarget);
      formData.append("diffusion_steps", String(params.diffusion_steps));
      formData.append("length_adjust", String(params.length_adjust));
      formData.append("inference_cfg_rate", String(params.inference_cfg_rate));
      formData.append("auto_f0_adjust", String(params.auto_f0_adjust));
      formData.append("pitch_shift", String(params.pitch_shift));
      formData.append("use_fp16", String(params.use_fp16));

      const headers: Record<string, string> = {};
      if (API_KEY) headers["X-API-Key"] = API_KEY;

      const res = await fetch(`${BACKEND_URL}/convert`, {
        method: "POST",
        body: formData,
        headers,
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.type === "chunk") {
            setMp3Chunks((prev) => [...prev, payload.audio]);
          } else if (payload.type === "done") {
            setFinalWav(payload.audio);
            setStatus("done");
            setProgressMsg("");
            setHistory((prev) => [
              {
                id: Date.now().toString(36),
                timestamp: Date.now(),
                sourceName: sourceFile.name,
                targetName: targetFile.name,
                params: {
                  diffusion_steps: params.diffusion_steps,
                  pitch_shift: params.pitch_shift,
                  length_adjust: params.length_adjust,
                  inference_cfg_rate: params.inference_cfg_rate,
                },
                wavB64: payload.audio,
              },
              ...prev,
            ]);
          } else if (payload.type === "error") {
            setStatus("error");
            setErrorMsg(payload.message);
            setProgressMsg("");
          }
        }
      }
    } catch (e: unknown) {
      setStatus("error");
      setErrorMsg(String(e));
      setProgressMsg("");
    }
  }

  const busy = status === "converting";
  const canConvert = serverReady && sourceFile && targetFile && !busy;
  const showIntermediates = sourceVocalsB64 || targetVocalsB64;

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">S</div>
          <span className="app-logo-text">SingVC</span>
          <span className="app-logo-tag">Tuner</span>
        </div>
        <div className={`status-badge status-${status}`}>
          {status === "loading" && "LOADING"}
          {status === "ready" && "READY"}
          {status === "converting" && "PROCESSING"}
          {status === "done" && "DONE"}
          {status === "error" && "ERROR"}
        </div>
      </header>

      <main className="app-main">
        <div className="upload-row">
          <AudioUploadWithSep
            label="原始音轨"
            file={targetFile}
            onFile={setTargetFile}
          />
          <AudioUploadWithSep
            label="目标音色"
            file={sourceFile}
            onFile={setSourceFile}
          />
        </div>

        <ParamSliders params={params} onChange={setParams}
          sep={sep} onSepChange={setSep} disabled={busy} />

        {errorMsg && <div className="error-box">{errorMsg}</div>}

        {busy && progressMsg && (
          <div className="progress-msg">{progressMsg}</div>
        )}

        <button className="convert-btn" onClick={handleConvert} disabled={!canConvert}>
          {busy ? "调音中…" : "开始调音"}
        </button>

        {showIntermediates && (
          <div className="intermediates">
            <div className="intermediates-title">人声提取预览</div>
            <div className="intermediates-grid">
              {sourceVocalsB64 && (
                <div className="intermediate-card">
                  <div className="intermediate-label">原始音轨 — 人声</div>
                  <audio
                    controls
                    className="intermediate-audio"
                    src={`data:audio/wav;base64,${sourceVocalsB64}`}
                  />
                </div>
              )}
              {targetVocalsB64 && (
                <div className="intermediate-card">
                  <div className="intermediate-label">目标音色 — 人声</div>
                  <audio
                    controls
                    className="intermediate-audio"
                    src={`data:audio/wav;base64,${targetVocalsB64}`}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {(mp3Chunks.length > 0 || finalWav) && (
          <AudioPlayer mp3Chunks={mp3Chunks} finalWav={finalWav} />
        )}

        <HistoryPanel items={history} onClear={handleClearHistory} />
      </main>
    </div>
  );
}
