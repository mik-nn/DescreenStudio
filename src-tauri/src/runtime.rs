use std::error::Error;
use std::path::PathBuf;

const SYSROOT_DLLS: [&str; 2] = ["onnxruntime.dll", "DirectML.dll"];

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Copy onnxruntime.dll / DirectML.dll (kept in `deps/`, gitignored) next to
/// the executable when newer, then point ORT_DYLIB_PATH at an existing copy.
/// See `scripts/fetch_ort_dll.ps1` to refresh the files from the Python wheel.
pub fn prepare_runtime() -> Result<(), Box<dyn Error>> {
    let manifest = manifest_dir();
    let exe_dir = std::env::current_exe()?
        .parent()
        .ok_or("no exe dir")?
        .to_path_buf();
    let mut dylib: Option<PathBuf> = None;
    for dll in SYSROOT_DLLS {
        let src = manifest.join("deps").join(dll);
        let dst = exe_dir.join(dll);
        if src.exists()
            && (!dst.exists()
                || src.metadata()?.modified()? > dst.metadata()?.modified()?)
        {
            std::fs::copy(&src, &dst)?;
        }
        if dll == "onnxruntime.dll" {
            if dst.exists() {
                dylib = Some(dst);
            } else if src.exists() {
                dylib = Some(src);
            }
        }
    }
    if let Some(p) = dylib {
        std::env::set_var("ORT_DYLIB_PATH", &p);
    }
    Ok(())
}

/// Locate `model.opt.onnx`: bundled resource dir -> exe dir -> manifest dir
/// (dev fallback, the file lives in `src-tauri/` and is gitignored).
pub fn resolve_model(extra_base: Option<PathBuf>) -> Result<PathBuf, Box<dyn Error>> {
    let mut cands = Vec::new();
    if let Some(b) = extra_base {
        cands.push(b.join("model.opt.onnx"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            cands.push(d.join("model.opt.onnx"));
        }
    }
    cands.push(manifest_dir().join("model.opt.onnx"));
    cands
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| "model.opt.onnx not found (resource / exe / manifest dirs)".into())
}
