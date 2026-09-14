use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array3, Array4, s};
use ndarray::prelude::*;

use descreen_studio::{Ep, InferenceEngine, TileParams};

const SYSROOT_DLLS: [&str; 2] = ["onnxruntime.dll", "DirectML.dll"];

/// Copy onnxruntime.dll / DirectML.dll (kept in `deps/`, gitignored) next to the
/// executable so `ort`'s load-dynamic runtime can find them, then read the binaries.
/// See `scripts/fetch_ort_dll.ps1` to refresh the files from the Python wheel.
fn prepare_runtime() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::current_exe()?;
    let exe_dir = exe.parent().ok_or("no exe dir")?;
    for dll in SYSROOT_DLLS {
        let src = manifest.join("deps").join(dll);
        let dst = exe_dir.join(dll);
        if !src.exists() {
            panic!(
                "missing {}", src.display()
            );
        }
        if !dst.exists() || src.metadata()?.modified()? > dst.metadata()?.modified()? {
            std::fs::copy(&src, &dst)?;
        }
    }
    // Also make ORT able to find them during session creation.
    std::env::set_var("ORT_DYLIB_PATH", manifest.join("deps").join("onnxruntime.dll"));
    Ok(())
}

/// Deterministic synthetic scan (gradient + low-noise), [0, 1], channels-first.
fn synthetic_scan(h: usize, w: usize) -> Array3<f32> {
    let mut img = Array3::<f32>::zeros((3, h, w));
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let n = (seed & 0xFFFF) as f32 / 65535.0;
                let grad = 0.25 + 0.5 * ((x as f32) / w as f32 + (y as f32) / h as f32 * 0.5);
                img[[c, y, x]] = (grad + 0.02 * (n - 0.5)).clamp(0.0, 1.0);
            }
        }
    }
    img
}

fn main() -> Result<(), Box<dyn Error>> {
    prepare_runtime()?;

    let mut model = String::from("model.opt.onnx");
    let mut h_px = 2480;
    let mut w_px = 3508;
    let mut ep = Ep::DirectML;
    let mut tile = 512usize;
    let mut overlap = 64usize;
    let mut verify = false;
    let mut probe = 0usize;
    let mut dump_dir: Option<String> = None;
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model = args[i].clone(); }
            "--h" => { i += 1; h_px = args[i].parse()?; }
            "--w" => { i += 1; w_px = args[i].parse()?; }
            "--tile" => { i += 1; tile = args[i].parse()?; }
            "--overlap" => { i += 1; overlap = args[i].parse()?; }
            "--verify" => verify = true,
            "--probe" => { i += 1; probe = args[i].parse()?; }
            "--dump" => { i += 1; dump_dir = Some(args[i].clone()); }
            "--ep" => {
                i += 1;
                ep = match args[i].as_str() {
                    "dml" | "directml" => Ep::DirectML,
                    "cpu" => Ep::Cpu,
                    other => return Err(format!("unknown ep {other}").into()),
                };
            }
            "--cpu" => ep = Ep::Cpu,
            other => return Err(format!("unknown arg {other}").into()),
        }
        i += 1;
    }

    let mut engine = InferenceEngine::new(&model, ep)?;
    println!("model: {model}");
    println!("ep requested: {:?}", engine.ep());

    let tp = TileParams { tile, overlap };
    let img = synthetic_scan(h_px, w_px);
    let dump = dump_dir.as_deref().map(std::path::Path::new);

    if let Some(dir) = dump {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("input_c0_3hw.raw"), as_bytes(img.view()))?;
        std::fs::write(dir.join("dims.txt"), format!("{h_px} {w_px}\n"))?;
    }

    // Warm-up on a single full tile.
    let warm = Array3::zeros((3, tile, tile));
    engine.descreen(&warm, tp)?;

    println!("image: {w_px}x{h_px} -> {tile}px tiles, overlap {overlap}");
    let t0 = Instant::now();
    let out = engine.descreen(&img, tp)?;
    let dt = t0.elapsed();

    if let Some(dir) = dump {
        std::fs::write(dir.join("out_c0_3hw.raw"), as_bytes(out.view()))?;
    }

    let strat: usize = (tile - overlap).max(2);
    let ty = ((h_px as isize - tile as isize) as f64 / strat as f64).ceil().max(0.0) as usize;
    let tx = ((w_px as isize - tile as isize) as f64 / strat as f64).ceil().max(0.0) as usize;
    let ph = ty * strat + tile;
    let pw = tx * strat + tile;
    let tiles_x = (pw - tile) / strat + 1;
    let tiles_y = (ph - tile) / strat + 1;
    let n = tiles_x * tiles_y;
    println!(
        "A4 @300dpi ~ {w_px}x{h_px}: total {:.2}s over {n} tiles ({:.0} ms/tile, padded {pw}x{ph})",
        dt.as_secs_f64(),
        dt.as_secs_f64() / n as f64 * 1000.0
    );

    // Seam check: two overlapping tiles share a Hann-weighted band; a step
    // between the acc-weighted mean over two close columns near a tile border
    // should be small (no hard discontinuity).
    let col_a = out.index_axis(Axis(2), 62).mean().unwrap();
    let col_b = out.index_axis(Axis(2), 70).mean().unwrap();
    println!("seam-sample delta (col 62 vs 70): {:.4}", (col_a - col_b).abs());

    if probe > 0 {
        // Diagnose nondeterminism / slice-vs-array input differences of the
        // engine on identical content.
        let full4 = Array4::from_shape_vec((1, 3, h_px, w_px), img.iter().copied().collect())?;
        let mut padded = Array4::<f32>::zeros((1, 3, h_px, w_px));
        {
            let mut dst = padded.slice_mut(s![0, .., .., ..]);
            dst.assign(&Array3::from_shape_vec((3, h_px, w_px), img.iter().copied().collect()).unwrap());
        }
        let patch = padded.slice(s![0..1, .., .., ..]).to_owned();
        let mut prev: Option<&str> = None;
        let mut prev_mean_array = Array3::<f32>::zeros((3, h_px, w_px));
        for (k, (name, arr)) in [
            ("a_direct", full4.clone()),
            ("b_sliceclone", patch.clone()),
            ("a_direct2", full4.clone()),
        ]
        .into_iter()
        .enumerate()
        {
            let t1 = Instant::now();
            let fout = engine.process_tile(&arr)?;
            let f3 = fout.index_axis(Axis(0), 0).to_owned();
            let mn = f3.index_axis(Axis(1), 0).mean().unwrap();
            let secs = t1.elapsed().as_secs_f64();
            match &prev {
                Some(pn) => {
                    let d = f3
                        .iter()
                        .zip(prev_mean_array.iter())
                        .fold(0.0f32, |m2, (a, b)| m2.max((a - b).abs()));
                    println!(
                        "probe {k} {name}: mean_c0={mn:.4} t={secs:.2}s |d vs {pn}|={d:.2e}",
                    );
                }
                None => println!("probe {k} {name}: mean_c0={mn:.4} t={secs:.2}s"),
            }
            prev = Some(name);
            prev_mean_array = f3;
        }
    }

    if verify {
        let full4 = Array4::from_shape_vec((1, 3, h_px, w_px), img.iter().copied().collect())?;
        let t1 = Instant::now();
        let fout = engine.process_tile(&full4)?;
        println!("full-image single-pass: {:.2}s", t1.elapsed().as_secs_f64());
        let f3 = fout.index_axis(Axis(0), 0);
        if let Some(dir) = dump {
            std::fs::write(dir.join("full_c0_3hw.raw"), as_bytes(f3.view()))?;
        }
        let mut max_abs: f32 = 0.0;
        let mut argmax = (0usize, 0usize, 0usize);
        let mut all_rms: f64 = 0.0;
        let mut all_n: usize = 0;
        let mut edge_rms: f64 = 0.0;
        let mut edge_n: usize = 0;
        let margin = (h_px.min(w_px) / 10).max(32);
        for i in 0..3 {
            for y in 0..h_px {
                for x in 0..w_px {
                    let d = out[[i, y, x]] - f3[[i, y, x]];
                    let ad = d.abs();
                    if ad > max_abs {
                        max_abs = ad;
                        argmax = (i, y, x);
                    }
                    all_rms += (d as f64) * (d as f64);
                    all_n += 1;
                    if y < margin || y >= h_px - margin || x < margin || x >= w_px - margin {
                        edge_rms += (d as f64) * (d as f64);
                        edge_n += 1;
                    }
                }
            }
        }
        println!(
            "tiled vs full: max_abs {:.2e} at (ch{}, y:{}, x:{}), rmse all {:.2e}, rmse edge-band {:.2e}",
            max_abs,
            argmax.0,
            argmax.1,
            argmax.2,
            (all_rms / all_n as f64).sqrt(),
            (edge_rms / edge_n as f64).sqrt()
        );
    }

    Ok(())
}

fn as_bytes(v: ndarray::ArrayView3<f32>) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for e in v.iter() {
        b.extend_from_slice(&e.to_le_bytes());
    }
    b
}