//! Обёртка вокруг nvcc для сборки Candle под драйвер CUDA 13.2.
//!
//! Проблема: установлен тулкит CUDA 13.4 (нужен для совместимости с MSVC 14.50,
//! см. STL1002), но драйвер NVIDIA поддерживает только CUDA 13.2
//! (cuDriverGetVersion = 13020). Candle встраивает PTX-текст ядер и передаёт
//! его драйверу для JIT-компиляции; nvcc 13.4 помечает PTX как `.version 9.4`,
//! и драйвер отклоняет его с CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
//!
//! Решение: после успешной работы nvcc понижаем директиву `.version` в
//! сгенерированных .ptx до 9.0 — PTX обратно совместим, драйвер 13.2
//! принимает ISA 9.0, а используемые ядрами инструкции (bf16/fp16/fp8)
//! требуют ISA не выше 9.0.
//!
//! Сборка обёртки:
//!   rustc -O nvcc_wrapper.rs -o nvcc_wrapper.exe
//!
//! Использование: переменная NVCC в .cargo/config.toml указывает на
//! nvcc_wrapper.exe; путь к настоящему nvcc передаётся через REAL_NVCC.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

/// Целевая версия PTX ISA: поддерживается драйвером CUDA 13.2.
const TARGET_PTX_VERSION: &str = "9.0";

fn find_ptx_outputs(args: &[String]) -> Vec<PathBuf> {
    // nvcc -ptx пишет результат в --output-directory (или cwd) с именем
    // исходного .cu-файла и расширением .ptx.
    let mut out_dirs: Vec<PathBuf> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if let Some(rest) = arg.strip_prefix("--output-directory") {
            let dir = if let Some(d) = rest.strip_prefix('=') {
                d.to_string()
            } else if let Some(d) = iter.next() {
                d.clone()
            } else {
                continue;
            };
            out_dirs.push(PathBuf::from(dir));
        } else if let Some(d) = arg.strip_prefix("-o") {
            // -o<path> или -o <path> — точный путь результата (для .ptx)
            let p = if d.len() > 1 {
                d[1..].trim_start_matches('=').to_string()
            } else if let Some(d) = iter.next() {
                d.clone()
            } else {
                continue;
            };
            if p.ends_with(".ptx") {
                sources.push(p);
            }
        } else if arg.ends_with(".cu") || arg.ends_with(".cuh") {
            // имя .ptx совпадает с именем .cu
            sources.push(arg.clone());
        }
    }
    if out_dirs.is_empty() {
        out_dirs.push(env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    }
    let mut result = Vec::new();
    for s in sources {
        let p = PathBuf::from(&s);
        let base = p
            .file_stem()
            .map(|b| b.to_string_lossy().into_owned())
            .unwrap_or_default();
        if s.ends_with(".ptx") {
            result.push(p);
        } else if !base.is_empty() {
            for d in &out_dirs {
                result.push(d.join(format!("{base}.ptx")));
            }
        }
    }
    result
}

fn patch_ptx(path: &Path) -> std::io::Result<bool> {
    let text = fs::read_to_string(path)?;
    let patched = regexless_patch(&text);
    if patched != text {
        fs::write(path, patched)?;
        return Ok(true);
    }
    Ok(false)
}

/// Заменяет `.version 9.4` (и любой другой X.Y) на целевую версию.
/// Без regex-зависимостей: ручной разбор префикса `.version `.
fn regexless_patch(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match rest.find(".version ") {
            Some(idx) => {
                out.push_str(&rest[..idx]);
                let after = &rest[idx + ".version ".len()..];
                // после номера версии идёт комментарий или перевод строки
                let ver_end = after
                    .find(|c: char| c == '\n' || c == ' ' || c == ';')
                    .unwrap_or(after.len());
                let (ver, tail) = after.split_at(ver_end);
                let ver = ver.trim();
                if ver.chars().all(|c| c.is_ascii_digit() || c == '.')
                    && ver.contains('.')
                    && ver != TARGET_PTX_VERSION
                {
                    out.push_str(".version ");
                    out.push_str(TARGET_PTX_VERSION);
                } else {
                    out.push_str(".version ");
                    out.push_str(ver);
                }
                rest = tail;
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

fn main() -> ExitCode {
    let real_nvcc = env::var("REAL_NVCC").unwrap_or_else(|_| {
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe".to_string()
    });
    let args: Vec<String> = env::args().skip(1).collect();

    let status = Command::new(&real_nvcc)
        .args(&args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|e| {
            eprintln!("nvcc_wrapper: failed to spawn {real_nvcc}: {e}");
            std::process::exit(111);
        });

    if !status.success() {
        return ExitCode::from(status.code().unwrap_or(1) as u8);
    }

    // nvcc отработал — понижаем .version в сгенерированных .ptx
    for ptx in find_ptx_outputs(&args) {
        if ptx.exists() {
            match patch_ptx(&ptx) {
                Ok(true) => eprintln!(
                    "nvcc_wrapper: patched PTX ISA -> {TARGET_PTX_VERSION} in {}",
                    ptx.display()
                ),
                Ok(false) => {}
                Err(e) => eprintln!("nvcc_wrapper: failed to patch {}: {e}", ptx.display()),
            }
        }
    }
    ExitCode::SUCCESS
}
