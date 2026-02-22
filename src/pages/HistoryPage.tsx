import { useEffect, useState, useCallback, useRef } from "react";
import HistoryPanel, { type HistoryItem } from "../components/HistoryPanel";
import { loadHistory, clearAll } from "../historyDB";

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const loaded = useRef(false);

  useEffect(() => {
    loadHistory().then((items) => {
      setHistory(items);
      loaded.current = true;
    });
  }, []);

  const handleClear = useCallback(() => {
    setHistory([]);
    clearAll();
  }, []);

  return (
    <main className="app-main">
      {history.length > 0 ? (
        <HistoryPanel items={history} onClear={handleClear} />
      ) : loaded.current ? (
        <div className="history-empty">暂无历史记录</div>
      ) : null}
    </main>
  );
}
