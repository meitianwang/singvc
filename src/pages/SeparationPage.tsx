import { useRef, useState } from "react";
import AudioUpload from "../components/AudioUpload";
import "./SeparationPage.css";

interface Props {
  serverPort: number;
}

interface StemResult {
  name: string;
  filename: string;
  audio: string; // base64
}

type SepStatus = "idle" | "separating" | "done" | "error";

const MODELS = [
  { value: "UVR-MDX-NET-Inst_HQ_3.onnx", label: "MDX-Net 人声分离（快速）" },
  { value: "model_bs_roformer_ep_368_sdr_12.9628.ckpt", label: "BS-Roformer 人声分离（高质量）" },
  { value: "htdemucs_ft.yaml", label: "Demucs 4-轨分离（人声 / 鼓 / 贝斯 / 其他）" },
];

const POSTPROCESS_OPTIONS = [
  { value: "", label: "无" },
  { value: "denoise", label: "去噪（UVR-DeNoise）" },
  { value: "deecho", label: "去混响（UVR-De-Echo）" },
  { value: "both", label: "去噪 + 去混响（最干净）" },
];

function StemCard({ stem }: { stem: StemResult }) {
  const audioRef = useRef<HTMLAudioElement>(null);

  const blobUrl = (() => {
    const bytes = Uint8Array.from(atob(stem.audio), (c) => c.charCodeAt(0));
    const ext = stem.filename.split(".").pop()?.toLowerCase() ?? "wav";
    const mime = ext === "mp3" ? "audio/mpeg" : ext === "flac" ? "audio/flac" : "audio/wav";
    return URL.createObjectURL(new Blob([bytes], { type: mime }));
  })();

  function handleDownload() {
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = stem.filename;
    a.click();
  }

  return (
    <div className="stem-card">
      <div className="stem-name">{stem.name}</div>
      <audio ref={audioRef} src={blobUrl} controls className="stem-audio" />
      <button className="stem-download-btn" onClick={handleDownload}>
        下载
      </button>
    </div>
  );
}

export default function SeparationPage({ serverPort }: Props) {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [model, setModel] = useState(MODELS[0].value);
  const [postprocess, setPostprocess] = useState("");
  const [status, setStatus] = useState<SepStatus>("idle");
  const [progressMsg, setProgressMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [stems, setStems] = useState<StemResult[]>([]);

  const isDemucs = model === "htdemucs_ft.yaml";

  async function handleSeparate() {
    if (!audioFile) return;
    setStatus("separating");
    setStems([]);
    setErrorMsg("");
    setProgressMsg("正在分离音频，请稍候…（首次运行将自动下载模型）");

    const formData = new FormData();
    formData.append("audio_file", audioFile);
    formData.append("model", model);
    formData.append("output_format", "wav");
    formData.append("postprocess", isDemucs ? "" : postprocess);

    try {
      const res = await fetch(`http://127.0.0.1:${serverPort}/separate`, {
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
          if (payload.type === "stem") {
            setStems((prev) => [
              ...prev,
              { name: payload.name, filename: payload.filename, audio: payload.audio },
            ]);
          } else if (payload.type === "progress") {
            setProgressMsg(payload.message);
          } else if (payload.type === "done") {
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

  const canSeparate = audioFile && status !== "separating";

  return (
    <div className="sep-page">
      <AudioUpload label="待分离音频" file={audioFile} onFile={setAudioFile} />

      <div className="model-row">
        <label className="model-label">分离模型</label>
        <select
          className="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={status === "separating"}
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <div className={`model-row${isDemucs ? " row-disabled" : ""}`}>
        <label className="model-label">人声后处理</label>
        <select
          className="model-select"
          value={postprocess}
          onChange={(e) => setPostprocess(e.target.value)}
          disabled={status === "separating" || isDemucs}
        >
          {POSTPROCESS_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      {errorMsg && <div className="error-box">{errorMsg}</div>}

      <button
        className="sep-btn"
        onClick={handleSeparate}
        disabled={!canSeparate}
      >
        {status === "separating" ? "分离中…" : "开始分离"}
      </button>

      {status === "separating" && progressMsg && (
        <div className="sep-waiting">⚙ {progressMsg}</div>
      )}

      {stems.length > 0 && (
        <div className="stems-section">
          <div className="stems-title">
            分离结果
            {status === "separating" && (
              <span className="separating-dot"> ⚙</span>
            )}
          </div>
          <div className="stems-grid">
            {stems.map((stem) => (
              <StemCard key={stem.filename} stem={stem} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
