pub mod app;
pub mod inference;
pub mod prefilter;
pub mod runtime;

pub use inference::{Ep, InferenceEngine, TileParams};

/// Tauri application entry: manages a lazily-initialized [`InferenceEngine`]
/// and exposes `preview_image` / `spectrum_image` / `run_descreen` commands.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(app::AppState(std::sync::Mutex::new(None)))
        .invoke_handler(app::handler())
        .run(tauri::generate_context!())
        .expect("failed to run descreen-studio app");
}
