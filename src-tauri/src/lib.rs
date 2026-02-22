use base64::Engine;

#[tauri::command]
async fn save_wav(wav_b64: String, default_name: String) -> Result<bool, String> {
    let data = base64::engine::general_purpose::STANDARD
        .decode(&wav_b64)
        .map_err(|e| e.to_string())?;

    let handle = rfd::AsyncFileDialog::new()
        .set_file_name(&default_name)
        .add_filter("WAV Audio", &["wav"])
        .save_file()
        .await;

    match handle {
        Some(h) => {
            std::fs::write(h.path(), &data).map_err(|e| e.to_string())?;
            Ok(true)
        }
        None => Ok(false),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![save_wav])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
