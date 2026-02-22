import { invoke } from "@tauri-apps/api/core";

export async function downloadWav(wavB64: string, defaultFilename: string) {
  try {
    await invoke("save_wav", { wavB64: wavB64, defaultName: defaultFilename });
  } catch {
    // Fallback for non-Tauri (browser dev mode)
    const bytes = Uint8Array.from(atob(wavB64), (c) => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = defaultFilename;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
