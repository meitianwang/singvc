import { useState, useEffect } from "react";
import { loadWav } from "../historyDB";
import { downloadWav } from "../downloadWav";
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

function HistoryItemRow({ item }: { item: HistoryItem }) {
  const [wav, setWav] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadWav(item.id).then((data) => {
      setWav(data ?? null);
      setLoading(false);
    });
  }, [item.id]);

  return (
    <div className="history-item">
      <div className="history-meta">
        <span className="history-time">{fmtTime(item.timestamp)}</span>
        <span className="history-files" title={`${item.sourceName} → ${item.targetName}`}>
          {item.sourceName} → {item.targetName}
        </span>
      </div>
      <div className="history-controls">
        {loading ? (
          <span className="history-loading">加载中…</span>
        ) : wav ? (
          <>
            <audio
              controls
              className="history-audio"
              src={`data:audio/wav;base64,${wav}`}
            />
            <button
              className="history-dl-btn"
              onClick={() => downloadWav(wav, `singvc_${item.id}.wav`)}
              title="下载"
            >
              ⬇
            </button>
          </>
        ) : (
          <span className="history-no-audio">音频数据已清除</span>
        )}
      </div>
    </div>
  );
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
            <HistoryItemRow key={item.id} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}
