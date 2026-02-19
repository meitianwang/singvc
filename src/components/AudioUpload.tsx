import { useRef, useState } from "react";
import "./AudioUpload.css";

interface Props {
  label: string;
  file: File | null;
  onFile: (f: File) => void;
}

export default function AudioUpload({ label, file, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) onFile(f);
  }

  return (
    <div
      className={`upload-zone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        style={{ display: "none" }}
        onChange={handleChange}
      />
      <div className="upload-icon">{file ? "🎵" : "🎤"}</div>
      <div className="upload-label">{label}</div>
      {file ? (
        <div className="upload-filename">{file.name}</div>
      ) : (
        <div className="upload-hint">点击或拖拽音频文件</div>
      )}
    </div>
  );
}
