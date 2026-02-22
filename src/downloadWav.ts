import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";

export async function downloadWav(wavB64: string, defaultFilename: string) {
  const bytes = Uint8Array.from(atob(wavB64), (c) => c.charCodeAt(0));

  try {
    const filePath = await save({
      defaultPath: defaultFilename,
      filters: [{ name: "WAV Audio", extensions: ["wav"] }],
    });
    if (!filePath) return; // user cancelled
    await writeFile(filePath, bytes);
  } catch {
    // Fallback for non-Tauri (browser dev mode)
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
