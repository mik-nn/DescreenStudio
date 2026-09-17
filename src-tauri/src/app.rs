use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use base64::Engine as _;
use image::ImageEncoder;
use rustfft::num_complex::Complex32;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::inference::{Ep, InferenceEngine, TileParams};
use crate::prefilter::{
    adaptive_notch, blend_map, build_notch_lists, detect_lf_bands, linear_tone, luma_of,
    return_background, subtract_lf, PrefilterCfg,
};
use crate::runtime;

pub struct AppState(pub Mutex<Option<InferenceEngine>>);

#[derive(Clone, serde::Serialize)]
pub(crate) struct ProgressPayload {
    done: usize,
    total: usize,
}

#[derive(serde::Serialize)]
pub(crate) struct PreviewOut {
    data_url: String,
    width: u32,
    height: u32,
}

#[derive(serde::Serialize)]
pub(crate) struct DescreenOut {
    output_path: String,
    width: u32,
    height: u32,
    elapsed_ms: u64,
}

pub struct DescreenCore {
    pub output_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub elapsed_ms: u64,
}

fn load_rgb(path: &Path) -> Result<image::RgbImage, String> {
    image::open(path)
        .map_err(|e| format!("open failed: {e}"))
        .map(|img| img.to_rgb8())
}

fn png_data_url_rgb(img: &image::RgbImage) -> Result<String, String> {
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    ))
}

fn png_data_url_gray(img: &image::GrayImage) -> Result<String, String> {
    let mut buf = Vec::new();
    image::codecs::png::PngEncoder::new(&mut buf)
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::L8,
        )
        .map_err(|e| format!("png encode failed: {e}"))?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    ))
}

pub(crate) fn preview_data_url(path: &Path, max_dim: u32) -> Result<PreviewOut, String> {
    let rgb = load_rgb(path)?;
    let (w, h) = rgb.dimensions();
    let thumb = if w.max(h) > max_dim {
        image::imageops::thumbnail(&rgb, max_dim, max_dim)
    } else {
        rgb
    };
    let (tw, th) = thumb.dimensions();
    Ok(PreviewOut {
        data_url: png_data_url_rgb(&thumb)?,
        width: tw,
        height: th,
    })
}

/// Log-magnitude 2D FFT spectrum (256x256, fftshifted) of the input image.
/// Same spectrum view as the v3.7 residual-raster detector.
pub(crate) fn spectrum_data_url(path: &Path) -> Result<PreviewOut, String> {
    const N: usize = 256;
    let gray = image::open(path)
        .map_err(|e| format!("open failed: {e}"))?
        .to_luma8();
    let g = image::imageops::resize(
        &gray,
        N as u32,
        N as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut buf: Vec<Complex32> = g
        .pixels()
        .map(|p| Complex32::new(p[0] as f32 / 255.0, 0.0))
        .collect();
    let mut tmp = vec![Complex32::new(0.0, 0.0); N];
    for r in 0..N {
        tmp.copy_from_slice(&buf[r * N..(r + 1) * N]);
        fft.process(&mut tmp);
        buf[r * N..(r + 1) * N].copy_from_slice(&tmp);
    }
    for c in 0..N {
        for r in 0..N {
            tmp[r] = buf[r * N + c];
        }
        fft.process(&mut tmp);
        for r in 0..N {
            buf[r * N + c] = tmp[r];
        }
    }
    let mut mag = vec![0.0f32; N * N];
    for y in 0..N {
        for x in 0..N {
            let sx = (x + N / 2) % N;
            let sy = (y + N / 2) % N;
            mag[y * N + x] = (buf[sy * N + sx].norm() + 1.0).ln();
        }
    }
    let max = mag.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let out = image::GrayImage::from_fn(N as u32, N as u32, |x, y| {
        image::Luma([(mag[(y as usize) * N + x as usize] / max * 255.0) as u8])
    });
    Ok(PreviewOut {
        data_url: png_data_url_gray(&out)?,
        width: N as u32,
        height: N as u32,
    })
}

/// Core descreen pipeline (Tauri-free, unit-testable):
/// load -> A0 raster measurement -> edge-adaptive notch -> LF-band subtraction
/// -> tiled inference -> AI-blend mix (anchor: prefiltered) -> paper-background
/// return -> linear tone match -> save `<stem>_descreened.png` next to input.
pub fn descreen_file(
    engine: &mut InferenceEngine,
    in_path: &Path,
    blend: f32,
    tp: TileParams,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<DescreenCore, String> {
    let rgb = load_rgb(in_path)?;
    let (w, h) = rgb.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let mut img = ndarray::Array3::<f32>::zeros((3, hu, wu));
    for (x, y, px) in rgb.enumerate_pixels() {
        img[[0, y as usize, x as usize]] = px[0] as f32 / 255.0;
        img[[1, y as usize, x as usize]] = px[1] as f32 / 255.0;
        img[[2, y as usize, x as usize]] = px[2] as f32 / 255.0;
    }
    let t0 = Instant::now();
    // Classical prefilter unloads the net: measured screen + LF banding gone,
    // the net owns detail only.
    let pcfg = PrefilterCfg::default();
    let t00 = Instant::now();
    let peaks = build_notch_lists(&img, &pcfg, 1024);
    let t_measure = t00.elapsed();
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        use crate::prefilter::{luma_of as _luma, screen_band_energy as _sbe};
        eprintln!(
            "[stage] peaks per ch: {:?}",
            peaks.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
        eprintln!("[stage] input bandE={:.4}", _sbe(&_luma(&img), 0.12, 0.18));
    }
    let t00 = Instant::now();
    let pre = adaptive_notch(&img, &peaks, &pcfg);
    let t_notch = t00.elapsed();
    let bands = detect_lf_bands(&luma_of(&pre), &pcfg);
    let t00 = Instant::now();
    let pre = subtract_lf(&pre, &bands, &pcfg);
    let t_lf = t00.elapsed();
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        use crate::prefilter::{luma_of as _luma2, screen_band_energy as _sbe2};
        eprintln!("[stage] lf bands: {}", bands.len());
        eprintln!("[stage] pre bandE={:.4}", _sbe2(&_luma2(&pre), 0.12, 0.18));
        // debug dump of the prefilter output next to the input
        let dbg_path = in_path.with_file_name(format!(
            "{}_dbg_pre.png",
            in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("scan")
        ));
        let mut dbg_img = image::RgbImage::new(w, h);
        for y in 0..hu {
            for x in 0..wu {
                let p = dbg_img.get_pixel_mut(x as u32, y as u32);
                for ch in 0..3 {
                    p[ch] = (pre[[ch, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        let _ = dbg_img.save(&dbg_path);
    }
    let out = engine
        .descreen_progress(&pre, tp, progress)
        .map_err(|e| format!("inference failed: {e}"))?;
    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let b = blend.clamp(0.0, 1.0);
    let wmap = blend_map(&img, &pcfg);
    let mut blended = ndarray::Array3::<f32>::zeros((3, hu, wu));
    for ch in 0..3 {
        for y in 0..hu {
            for x in 0..wu {
                let w = (wmap[[y, x]] * b).clamp(0.0, 1.0);
                blended[[ch, y, x]] = pre[[ch, y, x]] * (1.0 - w) + out[[ch, y, x]] * w;
            }
        }
    }
    let swapped = return_background(&img, &blended, pcfg.bg_sigma);
    let t00 = Instant::now();
    let toned = linear_tone(&swapped, &img);
    let t_tone = t00.elapsed();
    eprintln!(
        "[timing] measure={:.1}s notch={:.1}s lf={:.1}s bgswap+tone+blend={:.1}s",
        t_measure.as_secs_f64(),
        t_notch.as_secs_f64(),
        t_lf.as_secs_f64(),
        t_tone.as_secs_f64()
    );
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        use crate::prefilter::{luma_of as _luma3, screen_band_energy as _sbe3};
        eprintln!("[stage] net bandE={:.4}", _sbe3(&_luma3(&out), 0.12, 0.18));
        eprintln!("[stage] final bandE={:.4}", _sbe3(&_luma3(&toned), 0.12, 0.18));
    }
    let mut mixed = image::RgbImage::new(w, h);
    for y in 0..hu {
        for x in 0..wu {
            let p = mixed.get_pixel_mut(x as u32, y as u32);
            for ch in 0..3 {
                p[ch] = (toned[[ch, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    let stem = in_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scan");
    let out_path = in_path.with_file_name(format!("{stem}_descreened.png"));
    mixed
        .save(&out_path)
        .map_err(|e| format!("save failed: {e}"))?;
    Ok(DescreenCore {
        output_path: out_path,
        width: w,
        height: h,
        elapsed_ms,
    })
}

fn ensure_engine(state: &State<'_, AppState>, app: &AppHandle) -> Result<InferenceEngine, String> {
    let taken = state
        .0
        .lock()
        .map_err(|e| format!("state lock: {e}"))?
        .take();
    match taken {
        Some(e) => Ok(e),
        None => {
            runtime::prepare_runtime().map_err(|e| e.to_string())?;
            let res_base = app.path().resource_dir().ok();
            let model = runtime::resolve_model(res_base).map_err(|e| e.to_string())?;
            let name = model.to_string_lossy().into_owned();
            InferenceEngine::new(&name, Ep::DirectML)
                .or_else(|e| {
                    eprintln!("DirectML session failed ({e}), falling back to CPU");
                    InferenceEngine::new(&name, Ep::Cpu)
                })
                .map_err(|e| format!("session init failed: {e}"))
        }
    }
}

#[tauri::command]
async fn preview_image(path: String, max_dim: u32) -> Result<PreviewOut, String> {
    let p = PathBuf::from(path);
    tauri::async_runtime::spawn_blocking(move || preview_data_url(&p, max_dim.max(64)))
        .await
        .map_err(|e| format!("join: {e}"))?
}

#[tauri::command]
async fn spectrum_image(path: String) -> Result<PreviewOut, String> {
    let p = PathBuf::from(path);
    tauri::async_runtime::spawn_blocking(move || spectrum_data_url(&p))
        .await
        .map_err(|e| format!("join: {e}"))?
}

#[tauri::command]
async fn run_descreen(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    blend: f32,
) -> Result<DescreenOut, String> {
    let mut engine = ensure_engine(&state, &app)?;
    let in_path = PathBuf::from(path);
    let tp = TileParams::default();
    let (engine_back, result) = tauri::async_runtime::spawn_blocking(move || {
        let mut cb = |done: usize, total: usize| {
            let _ = app.emit("descreen-progress", ProgressPayload { done, total });
        };
        let r = descreen_file(&mut engine, &in_path, blend, tp, &mut cb);
        (engine, r)
    })
    .await
    .map_err(|e| format!("join: {e}"))?;
    *state
        .0
        .lock()
        .map_err(|e| format!("state lock: {e}"))? = Some(engine_back);
    let core = result?;
    Ok(DescreenOut {
        output_path: core.output_path.to_string_lossy().into_owned(),
        width: core.width,
        height: core.height,
        elapsed_ms: core.elapsed_ms,
    })
}

pub fn handler() -> impl Fn(tauri::ipc::Invoke) -> bool {
    tauri::generate_handler![preview_image, spectrum_image, run_descreen]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_png(dir: &Path) -> PathBuf {
        let (w, h) = (256u32, 256u32);
        let mut rgb = image::RgbImage::new(w, h);
        for (x, y, px) in rgb.enumerate_pixels_mut() {
            let v = ((x + y) % 256) as u8;
            *px = image::Rgb([v, v, v]);
        }
        let p = dir.join("tiny.png");
        rgb.save(&p).unwrap();
        p
    }

    #[test]
    fn notch_alone_kills_synthetic_grating() {
        use crate::prefilter::{
            adaptive_notch, build_notch_lists, luma_of, screen_band_energy, PrefilterCfg,
        };
        use ndarray::Array3;

        let (w, h) = (512u32, 512u32);
        let mut rgb = image::RgbImage::new(w, h);
        for (x, y, px) in rgb.enumerate_pixels_mut() {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let content = 0.3 + 0.35 * (fx + 0.5 * fy);
            let g1 = 0.09 * (2.0 * std::f32::consts::PI * 0.15 * fx * w as f32).cos();
            let v = ((content + g1).clamp(0.0, 1.0) * 255.0).round() as u8;
            *px = image::Rgb([v, v, v]);
        }
        let (w, h) = (w as usize, h as usize);
        let img = Array3::from_shape_fn((3, h, w), |(ch, y, x)| {
            rgb.get_pixel(x as u32, y as u32)[ch] as f32 / 255.0
        });
        let cfg = PrefilterCfg::default();
        let peaks = build_notch_lists(&img, &cfg, 1024);
        eprintln!("detected: {:?}", peaks[0]);
        let e0 = screen_band_energy(&luma_of(&img), 0.12, 0.18);
        let out = adaptive_notch(&img, &peaks, &cfg);
        let e1 = screen_band_energy(&luma_of(&out), 0.12, 0.18);
        eprintln!("notch alone band 0.12-0.18: {e0} -> {e1}");
        assert!(e1 < e0 * 0.2, "notch alone must kill the grating band");
    }

    #[test]
    fn prefilter_chain_beats_raw_inference() {
        use crate::prefilter::{luma_of, screen_band_energy};
        use ndarray::Array3;

        runtime::prepare_runtime().unwrap();
        let model = runtime::resolve_model(None).unwrap();
        let mut eng =
            InferenceEngine::new(&model.to_string_lossy(), Ep::Cpu).expect("cpu session");
        let dir = std::env::temp_dir().join("descreen_chain_test");
        std::fs::create_dir_all(&dir).unwrap();
        // synthetic AM print: gradient content + two screen gratings (r=0.15)
        let (w, h) = (512u32, 512u32);
        let mut rgb = image::RgbImage::new(w, h);
        for (x, y, px) in rgb.enumerate_pixels_mut() {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let content = 0.3 + 0.35 * (fx + 0.5 * fy);
            let g1 = 0.09 * (2.0 * std::f32::consts::PI * 0.15 * fx * w as f32).cos();
            let g2 = 0.09 * (2.0 * std::f32::consts::PI * 0.15 * (fx + fy) * w as f32 * 0.7071).cos();
            let v = ((content + g1 + g2).clamp(0.0, 1.0) * 255.0).round() as u8;
            *px = image::Rgb([v, v, v]);
        }
        let inp = dir.join("am_print.png");
        rgb.save(&inp).unwrap();
        let before = luma_of(&{
            let r = image::open(&inp).unwrap().to_rgb8();
            let (w, h) = (r.width() as usize, r.height() as usize);
            Array3::from_shape_fn((3, h, w), |(ch, y, x)| {
                r.get_pixel(x as u32, y as u32)[ch] as f32 / 255.0
            })
        });
        let r0 = screen_band_energy(&before, 0.12, 0.18);
        assert!(r0 > 0.05, "test image must carry screen-band energy, got {r0}");

        let mut prog = Vec::new();
        let core = descreen_file(
            &mut eng,
            &inp,
            1.0,
            TileParams { tile: 256, overlap: 32 },
            &mut |d, t| prog.push((d, t)),
        )
        .unwrap();
        let out_rgb = image::open(&core.output_path).unwrap().to_rgb8();
        let (w, h) = (out_rgb.width() as usize, out_rgb.height() as usize);
        let after = luma_of(&Array3::from_shape_fn((3, h, w), |(ch, y, x)| {
            out_rgb.get_pixel(x as u32, y as u32)[ch] as f32 / 255.0
        }));
        let r1 = screen_band_energy(&after, 0.12, 0.18);
        eprintln!("chain band 0.12-0.18: {r0} -> {r1}");
        assert!(r1 < r0 * 0.5, "chain must halve the screen band: {r0} -> {r1}");
        let (mut ms, mut mo) = (0.0f64, 0.0);
        for y in 0..h {
            for x in 0..w {
                ms += before[[y, x]] as f64;
                mo += after[[y, x]] as f64;
            }
        }
        ms /= (h * w) as f64;
        mo /= (h * w) as f64;
        assert!((ms - mo).abs() < 0.03, "tone must hold: {ms} vs {mo}");
    }

    #[test]
    fn stage5_pipeline_end_to_end() {
        runtime::prepare_runtime().unwrap();
        let model = runtime::resolve_model(None).unwrap();
        let mut eng =
            InferenceEngine::new(&model.to_string_lossy(), Ep::Cpu).expect("cpu session");
        let dir = std::env::temp_dir().join("descreen_stage5_test");
        std::fs::create_dir_all(&dir).unwrap();
        let inp = tiny_png(&dir);

        let pv = preview_data_url(&inp, 1600).unwrap();
        assert_eq!((pv.width, pv.height), (256, 256));
        assert!(pv.data_url.starts_with("data:image/png;base64,"));

        let sp = spectrum_data_url(&inp).unwrap();
        assert_eq!((sp.width, sp.height), (256, 256));
        assert!(sp.data_url.starts_with("data:image/png;base64,"));

        let mut prog: Vec<(usize, usize)> = Vec::new();
        let core = descreen_file(
            &mut eng,
            &inp,
            0.85,
            TileParams { tile: 256, overlap: 32 },
            &mut |d, t| prog.push((d, t)),
        )
        .unwrap();
        assert_eq!((core.width, core.height), (256, 256));
        assert!(core.output_path.exists());
        assert!(!prog.is_empty());
        let (d, t) = *prog.last().unwrap();
        assert_eq!(d, t);
        let out_rgb = image::open(&core.output_path).unwrap().to_rgb8();
        assert_eq!(out_rgb.dimensions(), (256, 256));
    }
}
