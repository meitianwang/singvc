use std::env;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Manager, State};

const DEFAULT_PORT: u16 = 18888;

pub struct ServerPort(pub u16);
pub struct PythonProcess(pub Mutex<Option<Child>>);

/// 读取 SINGVC_PORT 环境变量，回退到 DEFAULT_PORT
fn resolve_port() -> u16 {
    env::var("SINGVC_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

#[tauri::command]
fn get_server_port(port: State<ServerPort>) -> u16 {
    port.0
}

fn is_server_running(port: u16) -> bool {
    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

fn command_available(command: &str) -> bool {
    Command::new(command).arg("--version").output().is_ok()
}

fn resolve_python_bin() -> Option<String> {
    if let Ok(custom_python) = env::var("SINGVC_PYTHON") {
        if command_available(&custom_python) {
            return Some(custom_python);
        }
        eprintln!("SINGVC_PYTHON is set but not executable: {}", custom_python);
    }

    for candidate in ["python3.11", "python3.12", "python3.10", "python3", "python"] {
        if command_available(candidate) {
            return Some(candidate.to_string());
        }
    }

    None
}

fn resolve_server_script(app: &AppHandle) -> Option<PathBuf> {
    let mut candidates = Vec::<PathBuf>::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("python").join("server.py"));
    }

    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("python").join("server.py"));
        candidates.push(cwd.join("..").join("python").join("server.py"));
        candidates.push(cwd.join("..").join("..").join("python").join("server.py"));
    }

    if cfg!(debug_assertions) {
        candidates.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("python")
                .join("server.py"),
        );
    }

    candidates.into_iter().find(|path| path.exists())
}

fn start_python_server(app: &AppHandle, port: u16) {
    if is_server_running(port) {
        println!("Detected running python server on port {port}, reusing it");
        return;
    }

    let Some(server_script) = resolve_server_script(app) else {
        eprintln!("Python server script not found (expected python/server.py)");
        return;
    };

    let Some(python_bin) = resolve_python_bin() else {
        eprintln!("Python executable not found (tried python3*/python)");
        return;
    };

    let working_dir = server_script
        .parent()
        .map(PathBuf::from)
        .or_else(|| env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    let child = match Command::new(&python_bin)
        .arg(&server_script)
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&working_dir)
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            eprintln!("Failed to start python server: {err}");
            return;
        }
    };

    let state: State<PythonProcess> = app.state();
    *state.0.lock().unwrap() = Some(child);

    println!(
        "Python server started with {} {} on port {}",
        python_bin,
        server_script.display(),
        port
    );
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = resolve_port();
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(ServerPort(port))
        .manage(PythonProcess(Mutex::new(None)))
        .setup(move |app| {
            start_python_server(app.handle(), port);
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state: State<PythonProcess> = window.state();
                if let Some(mut child) = state.0.lock().unwrap().take() {
                    let _ = child.kill();
                    println!("Python server stopped");
                };
            }
        })
        .invoke_handler(tauri::generate_handler![get_server_port])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
