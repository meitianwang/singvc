import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import AudioUploadWithSep from "./components/AudioUploadWithSep";
import ParamSliders from "./components/ParamSliders";
import AudioPlayer from "./components/AudioPlayer";
import "./App.css";

export interface Params {
  diffusion_steps: number;
  length_adjust: number;
  inference_cfg_rate: number;
  auto_f0_adjust: boolean;
  pitch_shift: number;
}

type Status = "loading" | "ready" | "converting" | "done" | "error";

/** Call /separate with single_stem=Vocals, return the extracted vocals as a File. */
async function extractVocals(
  file: File, model: string, postprocess: string, port: number
): Promise<File> {
  const formData = new FormData();
  formData.append("audio_file", file);
  formData.append("model", model);
  formData.append("output_format", "wav");
  formData.append("postprocess", postprocess);
  formData.append("single_stem", "Vocals");

  const res = await fetch(`http://127.0.0.1:${port}/separate`, {
    method: "POST",
    body: formData,
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let vocalsB64: string | null = null;
  let vocalsFilename = "vocals.wav";

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
        vocalsB64 = payload.audio;
        vocalsFilename = payload.filename;
      } else if (payload.type === "error") {
        throw new Error(payload.message);
      }
    }
  }

  if (!vocalsB64) throw new Error("人声提取失败：未收到音频数据");
  const bytes = Uint8Array.from(atob(vocalsB64), (c) => c.charCodeAt(0));
  return new File([bytes], vocalsFilename, { type: "audio/wav" });
}

export default function App() {
  const [serverPort, setServerPort] = useState<number>(18888);
  const [serverReady, setServerReady] = useState(false);

  // Audio files
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);

  // Per-file separation options
  const [sourceSepEnabled, setSourceSepEnabled] = useState(false);
  const [sourceSepModel, setSourceSepModel] = useState("UVR-MDX-NET-Inst_HQ_3.onnx");
  const [sourceSepPP, setSourceSepPP] = useState("");
  const [targetSepEnabled, setTargetSepEnabled] = useState(false);
  const [targetSepModel, setTargetSepModel] = useState("UVR-MDX-NET-Inst_HQ_3.onnx");
  const [targetSepPP, setTargetSepPP] = useState("");

  // Conversion params
  const [params, setParams] = useState<Params>({
    diffusion_steps: 10,
    length_adjust: 1.0,
    inference_cfg_rate: 0.7,
    auto_f0_adjust: true,
    pitch_shift: 0,
  });

  // Status
  const [status, setStatus] = useState<Status>("loading");
  const [progressMsg, setProgressMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [mp3Chunks, setMp3Chunks] = useState<string[]>([]);
  const [finalWav, setFinalWav] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    invoke<number>("get_server_port")
      .then(setServerPort)
      .catch(() => {});
  }, []);

  useEffect(() => {
    const baseUrl = `http://127.0.0.1:${serverPort}`;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${baseUrl}/status`);
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
        // server not up yet
      }
    }, 1500);
    return () => clearInterval(pollRef.current!);
  }, [serverPort]);

  async function handleConvert() {
    if (!sourceFile || !targetFile || !serverReady) return;
    setStatus("converting");
    setMp3Chunks([]);
    setFinalWav(null);
    setErrorMsg("");

    try {
      let actualSource = sourceFile;
      let actualTarget = targetFile;

      if (sourceSepEnabled) {
        setProgressMsg("正在提取源音频人声…（首次运行将自动下载分离模型）");
        actualSource = await extractVocals(sourceFile, sourceSepModel, sourceSepPP, serverPort);
      }
      if (targetSepEnabled) {
        setProgressMsg("正在提取参考音频人声…");
        actualTarget = await extractVocals(targetFile, targetSepModel, targetSepPP, serverPort);
      }

      setProgressMsg("语音转换中…");

      const formData = new FormData();
      formData.append("source_file", actualSource);
      formData.append("target_file", actualTarget);
      formData.append("diffusion_steps", String(params.diffusion_steps));
      formData.append("length_adjust", String(params.length_adjust));
      formData.append("inference_cfg_rate", String(params.inference_cfg_rate));
      formData.append("auto_f0_adjust", String(params.auto_f0_adjust));
      formData.append("pitch_shift", String(params.pitch_shift));

      const res = await fetch(`http://127.0.0.1:${serverPort}/convert`, {
        method: "POST",
        body: formData,
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

  return (
    <div className="app">
      <header className="app-header">
        <h1>SingVC</h1>
        <div className={`status-badge status-${status}`}>
          {status === "loading" && "⏳ 模型加载中…"}
          {status === "ready" && "✓ 就绪"}
          {status === "converting" && "⚙ 处理中…"}
          {status === "done" && "✓ 完成"}
          {status === "error" && "✗ 错误"}
        </div>
      </header>

      <main className="app-main">
        <div className="upload-row">
          <AudioUploadWithSep
            label="源音频（待转换）"
            file={sourceFile}
            onFile={setSourceFile}
            sepEnabled={sourceSepEnabled}
            onSepEnabled={setSourceSepEnabled}
            sepModel={sourceSepModel}
            onSepModel={setSourceSepModel}
            postprocess={sourceSepPP}
            onPostprocess={setSourceSepPP}
            disabled={busy}
          />
          <AudioUploadWithSep
            label="参考音频（目标音色）"
            file={targetFile}
            onFile={setTargetFile}
            sepEnabled={targetSepEnabled}
            onSepEnabled={setTargetSepEnabled}
            sepModel={targetSepModel}
            onSepModel={setTargetSepModel}
            postprocess={targetSepPP}
            onPostprocess={setTargetSepPP}
            disabled={busy}
          />
        </div>

        <ParamSliders params={params} onChange={setParams} />

        {errorMsg && <div className="error-box">{errorMsg}</div>}

        {busy && progressMsg && (
          <div className="progress-msg">⚙ {progressMsg}</div>
        )}

        <button className="convert-btn" onClick={handleConvert} disabled={!canConvert}>
          {busy ? "处理中…" : "开始转换"}
        </button>

        {(mp3Chunks.length > 0 || finalWav) && (
          <AudioPlayer mp3Chunks={mp3Chunks} finalWav={finalWav} />
        )}
      </main>
    </div>
  );
}
