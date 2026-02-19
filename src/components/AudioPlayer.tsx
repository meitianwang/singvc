import { useEffect, useRef } from "react";
import "./AudioPlayer.css";

interface Props {
  mp3Chunks: string[];   // base64 encoded MP3 chunks (streamed)
  finalWav: string | null; // base64 encoded full WAV
}

export default function AudioPlayer({ mp3Chunks, finalWav }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);

  // When final WAV arrives, set it as the audio src
  useEffect(() => {
    if (finalWav && audioRef.current) {
      const wavBytes = Uint8Array.from(atob(finalWav), (c) => c.charCodeAt(0));
      const blob = new Blob([wavBytes], { type: "audio/wav" });
      audioRef.current.src = URL.createObjectURL(blob);
      audioRef.current.load();
    }
  }, [finalWav]);

  function handleDownload() {
    if (!finalWav) return;
    const wavBytes = Uint8Array.from(atob(finalWav), (c) => c.charCodeAt(0));
    const blob = new Blob([wavBytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "singvc_output.wav";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="player-panel">
      <div className="player-title">输出音频</div>

      <div className="chunk-progress">
        已接收 {mp3Chunks.length} 个音频块
        {!finalWav && mp3Chunks.length > 0 && (
          <span className="converting-dot"> ⚙</span>
        )}
      </div>

      {finalWav ? (
        <>
          <audio ref={audioRef} controls className="audio-element" />
          <button className="download-btn" onClick={handleDownload}>
            下载 WAV
          </button>
        </>
      ) : (
        <div className="player-waiting">转换完成后可播放和下载</div>
      )}
    </div>
  );
}
