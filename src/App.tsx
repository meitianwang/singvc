import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import AudioUpload from "./components/AudioUpload";
import ParamSliders from "./components/ParamSliders";
import AudioPlayer from "./components/AudioPlayer";
import SeparationPage from "./pages/SeparationPage";
import "./App.css";

export interface Params {
  diffusion_steps: number;
  length_adjust: number;
  inference_cfg_rate: number;
  auto_f0_adjust: boolean;
  pitch_shift: number;
}

type Status = "loading" | "ready" | "converting" | "done" | "error";
type Tab = "vc" | "sep";

export default function App() {
  const [serverPort, setServerPort] = useState<number>(18888);
  const [serverReady, setServerReady] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("vc");

  // Voice conversion state
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [params, setParams] = useState<Params>({
    diffusion_steps: 10,
    length_adjust: 1.0,
    inference_cfg_rate: 0.7,
    auto_f0_adjust: true,
    pitch_shift: 0,
  });
  const [status, setStatus] = useState<Status>("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [mp3Chunks, setMp3Chunks] = useState<string[]>([]);
  const [finalWav, setFinalWav] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Get server port from Tauri
  useEffect(() => {
    invoke<number>("get_server_port")
      .then(setServerPort)
      .catch(() => {});
  }, []);

  // Poll /status until models are loaded
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
        // server not up yet, keep polling
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

    const formData = new FormData();
    formData.append("source_file", sourceFile);
    formData.append("target_file", targetFile);
    formData.append("diffusion_steps", String(params.diffusion_steps));
    formData.append("length_adjust", String(params.length_adjust));
    formData.append("inference_cfg_rate", String(params.inference_cfg_rate));
    formData.append("auto_f0_adjust", String(params.auto_f0_adjust));
    formData.append("pitch_shift", String(params.pitch_shift));

    try {
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
          } else if (payload.type === "error") {
            setStatus("error");
            setErrorMsg(payload.message);
          }
        }
      }
    } catch (e: unknown) {
      setStatus("error");
      setErrorMsg(String(e));
    }
  }

  const canConvert = serverReady && sourceFile && targetFile && status !== "converting";

  return (
    <div className="app">
      <header className="app-header">
        <h1>SingVC</h1>
        <div className={`status-badge status-${status}`}>
          {status === "loading" && "⏳ 模型加载中…"}
          {status === "ready" && "✓ 就绪"}
          {status === "converting" && "⚙ 转换中…"}
          {status === "done" && "✓ 完成"}
          {status === "error" && "✗ 错误"}
        </div>
      </header>

      <nav className="tab-nav">
        <button
          className={`tab${activeTab === "vc" ? " active" : ""}`}
          onClick={() => setActiveTab("vc")}
        >
          歌声转换
        </button>
        <button
          className={`tab${activeTab === "sep" ? " active" : ""}`}
          onClick={() => setActiveTab("sep")}
        >
          音频分离
        </button>
      </nav>

      {activeTab === "vc" ? (
        <main className="app-main">
          <div className="upload-row">
            <AudioUpload label="源音频（待转换）" file={sourceFile} onFile={setSourceFile} />
            <AudioUpload label="参考音频（目标音色）" file={targetFile} onFile={setTargetFile} />
          </div>

          <ParamSliders params={params} onChange={setParams} />

          {errorMsg && <div className="error-box">{errorMsg}</div>}

          <button
            className="convert-btn"
            onClick={handleConvert}
            disabled={!canConvert}
          >
            {status === "converting" ? "转换中…" : "开始转换"}
          </button>

          {(mp3Chunks.length > 0 || finalWav) && (
            <AudioPlayer mp3Chunks={mp3Chunks} finalWav={finalWav} />
          )}
        </main>
      ) : (
        <SeparationPage serverPort={serverPort} />
      )}
    </div>
  );
}
