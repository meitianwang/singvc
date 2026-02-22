import { useState } from "react";
import "./HistoryPanel.css";

export interface HistoryItem {
  id: string;
  timestamp: number;
  sourceName: string;
  targetName: string;
  params: {
    diffusion_steps: number;
    pitch_shift: number;
    length_adjust: number;
    inference_cfg_rate: number;
  };
  wavB64: string;
}

interface Props {
  items: HistoryItem[];
  onClear: () => void;
}

function fmtTime(ts: number) {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function download(wavB64: string, filename: string) {
  const bytes = Uint8Array.from(atob(wavB64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function HistoryPanel({ items, onClear }: Props) {
  const [expanded, setExpanded] = useState(true);

  if (items.length === 0) return null;

  return (
    <div className="history-panel">
      <div className="history-header" onClick={() => setExpanded((v) => !v)}>
        <span className="history-title">历史记录（{items.length}）</span>
        <div className="history-actions">
          <button
            className="history-clear-btn"
            onClick={(e) => { e.stopPropagation(); onClear(); }}
          >
            清空
          </button>
          <span className="history-toggle">{expanded ? "▾" : "▸"}</span>
        </div>
      </div>

      {expanded && (
        <div className="history-list">
          {items.map((item) => (
            <div key={item.id} className="history-item">
              <div className="history-meta">
                <span className="history-time">{fmtTime(item.timestamp)}</span>
                <span className="history-files" title={`${item.sourceName} → ${item.targetName}`}>
                  {item.sourceName} → {item.targetName}
                </span>
                <span className="history-params">
                  步数{item.params.diffusion_steps}
                  {item.params.pitch_shift !== 0 && ` / 移调${item.params.pitch_shift > 0 ? "+" : ""}${item.params.pitch_shift}`}
                </span>
              </div>
              <div className="history-controls">
                <audio
                  controls
                  className="history-audio"
                  src={`data:audio/wav;base64,${item.wavB64}`}
                />
                <button
                  className="history-dl-btn"
                  onClick={() => download(item.wavB64, `singvc_${item.id}.wav`)}
                  title="下载"
                >
                  ⬇
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
