use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use base64::Engine as _;
use image::ImageEncoder;
use rustfft::num_complex::Complex32;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::candle_engine::CandleEngine;
use crate::inference::{Ep, InferenceEngine, TileParams};
use crate::prefilter::{
    adaptive_notch, blend_map, build_notch_lists, detect_lf_bands, linear_tone, luma_of,
    return_background, subtract_lf, top4_peak_sum, PrefilterCfg,
};
use crate::runtime;

/// Движок инференса: Candle (safetensors + FiLM, основной) или
/// ort/DirectML (run3-era fallback, JOURNAL 2026-09-21 «ONNX закрыт»).
pub enum Engine {
    Ort(InferenceEngine),
    Candle(CandleEngine),
}

impl Engine {
    pub fn set_cond(&mut self, cond: Option<[f32; 4]>) {
        if let Engine::Candle(e) = self {
            e.set_cond(cond);
        }
    }

    pub fn descreen_progress(
        &mut self,
        img: &ndarray::Array3<f32>,
        tp: TileParams,
        on_tile: &mut dyn FnMut(usize, usize),
    ) -> Result<ndarray::Array3<f32>, Box<dyn std::error::Error>> {
        match self {
            Engine::Ort(e) => e.descreen_progress(img, tp, on_tile),
            Engine::Candle(e) => e.descreen_progress(img, tp, on_tile),
        }
    }

    pub fn describe(&self) -> &'static str {
        match self {
            Engine::Ort(_) => "ort",
            Engine::Candle(_) => "candle",
        }
    }

    /// Прогон без прогресса (бенчи).
    pub fn descreen(
        &mut self,
        img: &ndarray::Array3<f32>,
        tp: TileParams,
    ) -> Result<ndarray::Array3<f32>, Box<dyn std::error::Error>> {
        let mut noop = |_d: usize, _t: usize| {};
        self.descreen_progress(img, tp, &mut noop)
    }
}

pub struct AppState(pub Mutex<Option<Engine>>);

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
    // TIFF от МФУ до ~10000 px: дефолтный лимит декодера (~512 MB декод-буферов)
    // валит Calendar1 (7650×12724); снимаем до 4 GB (сам декод <0.5 GB).
    let mut limits = image::io::Limits::default();
    limits.max_alloc = Some(4 * 1024 * 1024 * 1024);
    let mut reader = image::ImageReader::open(path)
        .map_err(|e| format!("open failed: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("open failed: {e}"))?;
    reader.limits(limits);
    reader
        .decode()
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
    engine: &mut Engine,
    in_path: &Path,
    blend: f32,
    tp: TileParams,
    progress: &mut dyn FnMut(usize, usize),
    nyquist: bool,
    v1_enabled: bool,
) -> Result<DescreenCore, String> {
    let rgb = load_rgb(in_path)?;
    let (w, h) = rgb.dimensions();
    let (mut wu, mut hu) = (w as usize, h as usize);
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
    // Найквист–Котельников режим (2.27→продукт): приводим период растра к
    // 2 px (r=0.5), вся цепочка работает в этом разрешении, результат
    // возвращается в исходный размер (Lanczos3).
    // `DESCREEN_NATIVE=1` — нативный режим (2.40): рабочий кадр = скан, ни
    // даунсэмпла, ни апскейла, снятие растра — целиком на сети.
    let native_mode = std::env::var("DESCREEN_NATIVE").map(|v| v != "0").unwrap_or(false);
    let nyq = if nyquist && !native_mode {
        crate::nyquist::plan_with(&img, &pcfg, &crate::nyquist::NyqCfg::from_env())
    } else {
        crate::nyquist::NyquistPlan {
            r_in: 0.0,
            conf_in: 0.0,
            scale: 1.0,
            r_out: 0.0,
            out_w: wu,
            out_h: hu,
            iters: 0,
        }
    };
    let nyq_active = nyquist && !native_mode && nyq.scale < crate::nyquist::SCALE_SKIP;
    // 2.43: замер растра по сетке окон вместо одного центрального. На
    // `Magazine (2)` (69 Мпикс) центр пришёлся на плашку без растра — A0 дал
    // 0.244 при conf 0.03, V1 не сработал, кольцевой notch вырезал не ту
    // полосу, и unsharp усилил неснятую решётку. Считается ОДИН раз: V1,
    // радиус кольца и FiLM-cond должны видеть одну и ту же несущую.
    let est_native = if v1_enabled || nyquist || native_mode {
        crate::prefilter::estimate_raster_multi(&luma_of(&img), &pcfg, 1024, 3)
    } else {
        crate::prefilter::estimate_raster(&luma_of(&img), &pcfg, 1024)
    };
    // 2.42: радиус несущей для кольцевого notch. В Найквист-режиме он задан
    // планом (r_out), в нативном — оценкой A0 на исходном кадре.
    let ring_r = if nyq_active {
        nyq.r_out
    } else if nyquist || native_mode {
        est_native.r
    } else {
        0.0
    };
    // V1 (2.32): параметрическое вычитание растра «оценить -> синтезировать ->
    // вычесть». Работает в НАТИВНОМ разрешении и поэтому идёт ДО Найквист-
    // даунсэмпла: на chain-кадре (период 2 px) демодуляция вырождена — боковые
    // полосы АМ складываются. `plan()` уже посчитан по исходному изображению,
    // так что масштаб нормализации вычитание не портит.
    let v1_cfg = crate::v1::V1Cfg::from_env();
    let v1_acted = if v1_enabled {
        let band0 = crate::prefilter::band_energy_abs(&luma_of(&img), 0.40 * est_native.r / 0.5, 0.50 * est_native.r / 0.5);
        let t1 = Instant::now();
        let rep = crate::v1::apply(&mut img, est_native.r, &v1_cfg, &pcfg, 1024);
        let band1 = crate::prefilter::band_energy_abs(&luma_of(&img), 0.40 * est_native.r / 0.5, 0.50 * est_native.r / 0.5);
        eprintln!(
            "[stage] V1: r_ref={:.4} sigma={} gate=({}, {}) carriers={:?} coh={:.2} acted={} band={:.3e}->{:.3e} ({:.1}%) {:.1}s",
            rep.r_ref,
            v1_cfg.sigma,
            v1_cfg.gate_lo,
            v1_cfg.gate_hi,
            rep.carriers
                .iter()
                .map(|v| v.iter().map(|(r, a)| format!("{r:.3}@{a:.0}")).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            rep.mean_coh,
            rep.acted,
            band0,
            band1,
            if band0 > 0.0 { 100.0 * (1.0 - band1 / band0) } else { 0.0 },
            t1.elapsed().as_secs_f64()
        );
        rep.acted
    } else {
        false
    };
    if nyq_active {
        eprintln!(
            "[stage] nyquist: r_in={:.4} conf={:.3} scale={:.4} out={}x{} r_out={:.4} iters={}",
            nyq.r_in,
            nyq.conf_in,
            nyq.scale,
            nyq.out_w,
            nyq.out_h,
            nyq.r_out,
            nyq.iters
        );
        img = crate::nyquist::resize_box_rgb(&img, nyq.scale);
        hu = nyq.out_h;
        wu = nyq.out_w;
    }
    // A0-замер на входе сети (до префильтра) — cond для FiLM-run27_hyb;
    // паритет с eval_realscan.py:140-166. Для ort/run3 cond игнорируется.
    // 2.43: сетка окон — один центральный ROI ошибается на крупных сканах.
    let est = crate::prefilter::estimate_raster_multi(&luma_of(&img), &pcfg, 1024, 3);
    let cond = crate::candle_engine::build_cond_vec(est.r, est.angle_deg, est.confidence);
    eprintln!(
        "[stage] A0: r={:.4} ang={:.1} conf={:.3} cond=[{:.3},{:.3},{:.3},{:.3}] engine={}",
        est.r,
        est.angle_deg,
        est.confidence,
        cond[0],
        cond[1],
        cond[2],
        cond[3],
        engine.describe()
    );
    engine.set_cond(Some(cond));
    let t00 = Instant::now();
    let mut peaks = build_notch_lists(&img, &pcfg, 1024);
    // 2.40: в Найквист-режиме несущая в рабочем кадре ИЗВЕСТНА по построению
    // (r_out), а A0 на этом кадре читает ложный пик (Pack: 0.4258 вместо 0.30),
    // из-за чего notch промахивался и остаток растра доживал до выхода.
    // DESCREEN_NOTCH_R фиксирует радиус несущей, сохраняя углы и гармоники.
    if let Ok(v) = std::env::var("DESCREEN_NOTCH_R") {
        if let Ok(want) = v.parse::<f32>() {
            for list in peaks.iter_mut() {
                if let Some(&(r0, _)) = list.first() {
                    if r0 > 1e-4 {
                        let k = want / r0;
                        for p in list.iter_mut() {
                            p.0 = (p.0 * k).clamp(0.02, 0.48);
                        }
                    }
                }
            }
            if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
                eprintln!("[stage] notch r override -> {want:.4}");
            }
        }
    }
    let t_measure = t00.elapsed();
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        use crate::prefilter::{luma_of as _luma, screen_band_energy as _sbe};
        eprintln!(
            "[stage] peaks per ch: {:?}",
            peaks.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
        eprintln!("[stage] input bandE={:.4}", _sbe(&_luma(&img), 0.12, 0.18));
    }
    // Классический notch остаётся как исходная стадия, но при сработавшем V1
    // он уже не нужен: растр снят вычитанием, а «прокол» спектра только
    // добавляет рингинг (замер 2.31: на Pack +34% энергии полосы). Режим:
    // DESCREEN_NOTCH=auto (по умолчанию) | off | force.
    let notch_mode = std::env::var("DESCREEN_NOTCH").unwrap_or_else(|_| "auto".into());
    let skip_notch = match notch_mode.as_str() {
        "off" => true,
        "force" => false,
        _ => v1_acted,
    };
    let t00 = Instant::now();
    let mut pre = if skip_notch {
        eprintln!("[stage] notch: skipped (v1_acted={v1_acted} mode={notch_mode})");
        img.clone()
    } else {
        adaptive_notch(&img, &peaks, &pcfg)
    };
    // 2.40: кольцевой notch по известной несущей. В Найквист-режиме частота
    // растра в рабочем кадре задана планом (r_out), и остаток V1 + его AM-
    // боковые полосы обязаны уйти, иначе они ложатся в видимую полосу и
    // усиливаются unsharp-ом (Pack: 4.28 % модуляции против 3.12 % в скане).
    if std::env::var("DESCREEN_RING").map(|v| v != "0").unwrap_or(false) && ring_r > 0.0 {
        let bw = std::env::var("DESCREEN_RING_BW")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.02);
        let stage = std::env::var("DESCREEN_RING_STAGE").unwrap_or_else(|_| "both".into());
        if stage != "post" {
            pre = crate::prefilter::ring_notch(&pre, ring_r, bw, 4);
        }
        if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
            eprintln!("[stage] ring notch: r={:.4} bw={:.3} stage={}", ring_r, bw, stage);
        }
    }
    let t_notch = t00.elapsed();
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        // intermediate dump: notch stage output (border coverage diagnostics)
        let dbg_path = in_path.with_file_name(format!(
            "{}_dbg_notch.png",
            in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("scan")
        ));
        let mut dbg_img = image::RgbImage::new(wu as u32, hu as u32);
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
    let bands = detect_lf_bands(&luma_of(&pre), &pcfg);
    // LF-стадия (осевые полосы сканера/бумаги). Замер на контонном эталоне
    // (Temp/lf_value.py): notch 18.74 дБ → notch+LF 17.42 дБ, то есть стадия
    // снимает контент вместе с полосой. Оставлена переключаемой для A/B:
    // DESCREEN_LF=off | force (по умолчанию force — прежнее поведение).
    let lf_mode = std::env::var("DESCREEN_LF").unwrap_or_else(|_| "auto".into());
    let skip_lf = lf_mode == "off";
    let t00 = Instant::now();
    let pre = if skip_lf || bands.is_empty() {
        if skip_lf {
            eprintln!("[stage] lf: skipped (mode=off)");
        }
        pre
    } else {
        let cand = subtract_lf(&pre, &bands, &pcfg);
        // Гейт: полоса настоящая, только если осевая пиковость реально упала.
        // На book1/BookCOver вычитание снижало осевую энергию на ~20%, но
        // пиковость не менялась (301.9→302.2, 2424.9→2434.1) — то есть
        // срезался контент, а не полоса (Temp/lf_real.py).
        let pk0 = crate::prefilter::axis_peakiness(&luma_of(&pre));
        let pk1 = crate::prefilter::axis_peakiness(&luma_of(&cand));
        let keep = pk1 < pk0 * crate::prefilter::LF_PEAK_DROP;
        eprintln!(
            "[stage] lf gate: peakiness {pk0:.1} -> {pk1:.1} (need <{:.1}) -> {}",
            pk0 * crate::prefilter::LF_PEAK_DROP,
            if keep { "apply" } else { "revert" }
        );
        if keep {
            cand
        } else {
            pre
        }
    };
    let t_lf = t00.elapsed();
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        use crate::prefilter::{luma_of as _luma2, screen_band_energy as _sbe2};
        eprintln!("[stage] lf bands: {}", bands.len());
        if !bands.is_empty() {
            eprintln!(
                "[stage] lf peaks: {:?}",
                bands
                    .iter()
                    .map(|b| (b.r, b.angle_deg))
                    .collect::<Vec<(f32, f32)>>()
            );
        }
        eprintln!("[stage] pre bandE={:.4}", _sbe2(&_luma2(&pre), 0.12, 0.18));
        // debug dump of the prefilter output next to the input
        let dbg_path = in_path.with_file_name(format!(
            "{}_dbg_pre.png",
            in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("scan")
        ));
        let mut dbg_img = image::RgbImage::new(wu as u32, hu as u32);
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
    // Full post-net tail (blend -> paper background -> tone), parameterized by
    // the correction scale k: net_corr = pre + k * (net - pre). Parity with
    // `ml/preproc/backoff.apply_global` net_raw before `finish`. b_use — вес
    // нейросети (адаптивный спад ниже, см. blend-fallback).
    let finish = |net: &ndarray::Array3<f32>, k: f32, b_use: f32|
     -> ndarray::Array3<f32> {
        let mut blended = ndarray::Array3::<f32>::zeros((3, hu, wu));
        for ch in 0..3 {
            for y in 0..hu {
                for x in 0..wu {
                    let w = (wmap[[y, x]] * b_use).clamp(0.0, 1.0);
                    let nv = pre[[ch, y, x]] + k * (net[[ch, y, x]] - pre[[ch, y, x]]);
                    blended[[ch, y, x]] = pre[[ch, y, x]] * (1.0 - w) + nv * w;
                }
            }
        }
        let swapped = return_background(&img, &blended, pcfg.bg_sigma);
        linear_tone(&swapped, &img)
    };
    let t00 = Instant::now();
    let mut toned = finish(&out, 1.0, b);
    // Gated scalar backoff: the net correction may re-introduce periodicity
    // (case 1821146u, top4 113 -> 182); if the finished frame's top-4
    // residual peaks exceed the prefiltered input by > gate, re-run the tail
    // with the correction scaled to backoff_w.
    let mut backoff_fired = false;
    if pcfg.backoff_enabled {
        let t4_in = top4_peak_sum(&luma_of(&pre));
        let t4_out = top4_peak_sum(&luma_of(&toned));
        let mut t4_after = t4_out;
        if t4_in > 0.0 && t4_out > t4_in * (1.0 + pcfg.backoff_gate) {
            toned = finish(&out, pcfg.backoff_w, b);
            t4_after = top4_peak_sum(&luma_of(&toned));
            backoff_fired = true;
        }
        if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
            eprintln!(
                "[stage] backoff: top4 pre={t4_in:.1} final={t4_out:.1} gate={:.2} fired={backoff_fired} after={t4_after:.1}",
                1.0 + pcfg.backoff_gate
            );
        }
    }
    let t_finish = t00.elapsed();
    // Adaptive unsharp (паритет ml/preproc/unsharp.py): feedback по
    // lapvar-резкости к цели 1.0; DESCREEN_NO_UNSHARP=1 — выкл для A/B.
    // Blend-fallback: если после потолка unsharp sharp < 0.90 — нейросеть
    // съела контент (случай Calendar1 на Найквисте); хвост пересобирается
    // на меньшем весе net (дешёво: inference уже сделан) и берётся вариант
    // с лучшей резкостью.
    let t00 = Instant::now();
    let k_final = if backoff_fired { pcfg.backoff_w } else { 1.0 };
    let no_unsharp = std::env::var_os("DESCREEN_NO_UNSHARP").is_some();
    // Потолок amount: DESCREEN_US_MAX (по умолчанию AMOUNT_MAX). Петля целится
    // в sharp=1.0 относительно скана, а в Найквист-режиме эта цель держится
    // остатком растра — без потолка она раздувает полосу 0.02–0.08 (2.39).
    let us_max = std::env::var("DESCREEN_US_MAX")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(crate::unsharp::AMOUNT_MAX);
    // Опорный кадр для петли резкости. `scan` — историческое поведение
    // (сравнение со сканом), `pre` — с пре-стейджем: в нативном режиме скан
    // содержит растр, и петля «догоняет» его ВЧ, усиливая остаток (2.40).
    let us_ref = if std::env::var("DESCREEN_US_REF").map(|v| v == "pre").unwrap_or(false) {
        &pre
    } else {
        &img
    };
    // 2.42: цель петли. После кольцевого notch резкость падает на 15–20 %,
    // поэтому при включённой зачистке цель поднимается (DESCREEN_US_TARGET).
    let us_target = std::env::var("DESCREEN_US_TARGET")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(crate::unsharp::SHARP_TARGET);
    // 2.42: σ unsharp. Меньшая σ меньше усиливает решётку (σ=1 даёт на несущей
    // 0.145 c/px gain 1+0.34·amount, σ=0.6 — 1+0.14·amount).
    let us_sigma = std::env::var("DESCREEN_US_SIGMA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(crate::unsharp::SIGMA);
    // 2.43: порог маски структуры. 0 — линейный unsharp (историческое
    // поведение), >0 — усиление пропорционально локальному контрасту, чтобы не
    // раздувать зерно и остаток решётки на плашках.
    let us_mask = std::env::var("DESCREEN_US_MASK")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.0);
    let (mut toned, mut us_amount, mut us_sharp) = if no_unsharp {
        (toned, 0.0f32, 0.0f32)
    } else {
        crate::unsharp::adaptive_unsharp_ex(
            &toned, us_ref, us_max, us_target, us_sigma, us_mask,
        )
    };
    let mut b_used = b;
    if !no_unsharp && us_sharp < 0.90 && b > 0.4 {
        let (orig_sharp, orig_amt) = (us_sharp, us_amount);
        for bf in [0.4f32, 0.15] {
            let t2 = finish(&out, k_final, bf);
            let (o2, a2, s2) = crate::unsharp::adaptive_unsharp_ex(
                &t2, &img, us_max, us_target, us_sigma, us_mask,
            );
            if s2 > us_sharp {
                toned = o2;
                us_amount = a2;
                us_sharp = s2;
                b_used = bf;
            }
            if us_sharp >= 0.90 {
                break;
            }
        }
        if std::env::var_os("DESCREEN_STAGE_DBG").is_some() || b_used != b {
            eprintln!(
                "[stage] blend-fallback: b {b:.2} -> {b_used:.2} sharp {orig_sharp:.3} -> {us_sharp:.3} (amount {orig_amt:.2} -> {us_amount:.2})"
            );
        }
    }
    let t_unsharp = t00.elapsed();
    // 2.40: финальная зачистка несущей. unsharp (σ=1) усиливает остаток на
    // r_out в ≈1.4× (Pack: пре 1.55 % -> после unsharp 1.94 %), поэтому кольцо
    // вырезается ещё раз уже в готовом кадре.
    if ring_r > 0.0
        && std::env::var("DESCREEN_RING").map(|v| v != "0").unwrap_or(false)
        && std::env::var("DESCREEN_RING_STAGE").unwrap_or_else(|_| "both".into()) != "pre"
    {
        let bw = std::env::var("DESCREEN_RING_BW")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.02);
        toned = crate::prefilter::ring_notch(&toned, ring_r, bw, 4);
        if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
            eprintln!("[stage] ring notch (post): r={:.4} bw={:.3}", ring_r, bw);
        }
    }
    eprintln!(
        "[timing] measure={:.1}s notch={:.1}s lf={:.1}s finish={:.1}s unsharp={:.1}s",
        t_measure.as_secs_f64(),
        t_notch.as_secs_f64(),
        t_lf.as_secs_f64(),
        t_finish.as_secs_f64(),
        t_unsharp.as_secs_f64()
    );
    eprintln!("[stage] unsharp: amount={us_amount:.2} sharp={us_sharp:.3}");
    if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
        use crate::prefilter::{luma_of as _luma3, screen_band_energy as _sbe3};
        eprintln!("[stage] net bandE={:.4}", _sbe3(&_luma3(&out), 0.12, 0.18));
        eprintln!("[stage] final bandE={:.4}", _sbe3(&_luma3(&toned), 0.12, 0.18));
    }
    // Найквист-режим: цепочка работала в уменьшенном разрешении — результат
    // возвращается в исходный размер (Lanczos3), чтобы UI/файл совпадали с входом.
    let toned_res = if nyq_active {
        // bench-протокол 2.27: метрики на chain-разрешении
        if std::env::var_os("DESCREEN_STAGE_DBG").is_some() {
            let dbg_path = in_path.with_file_name(format!(
                "{}_dbg_out.png",
                in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("scan")
            ));
            let mut dbg_img = image::RgbImage::new(wu as u32, hu as u32);
            for y in 0..hu {
                for x in 0..wu {
                    let p = dbg_img.get_pixel_mut(x as u32, y as u32);
                    for ch in 0..3 {
                        p[ch] = (toned[[ch, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
            let _ = dbg_img.save(&dbg_path);
        }
        crate::nyquist::upsample_rgb(&toned, w as usize, h as usize)
    } else {
        toned
    };
    let mut mixed = image::RgbImage::new(w, h);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = mixed.get_pixel_mut(x as u32, y as u32);
            for ch in 0..3 {
                p[ch] = (toned_res[[ch, y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
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

fn ensure_engine(state: &State<'_, AppState>, app: &AppHandle) -> Result<Engine, String> {
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
            // Основной путь — Candle на safetensors (run27_hyb + FiLM, CUDA);
            // DESCREEN_ENGINE=ort — форс fallback-ветки (run3 ONNX/DirectML).
            let force_ort = std::env::var("DESCREEN_ENGINE").as_deref() == Ok("ort");
            if !force_ort {
                if let Ok(model) = runtime::resolve_safetensors(res_base.clone()) {
                    let name = model.to_string_lossy().into_owned();
                    match CandleEngine::new(&name) {
                        Ok(e) => return Ok(Engine::Candle(e)),
                        Err(err) => eprintln!("candle init failed ({err}), falling back to ort"),
                    }
                }
            }
            let model = runtime::resolve_model(res_base).map_err(|e| e.to_string())?;
            let name = model.to_string_lossy().into_owned();
            InferenceEngine::new(&name, Ep::DirectML)
                .or_else(|e| {
                    eprintln!("DirectML session failed ({e}), falling back to CPU");
                    InferenceEngine::new(&name, Ep::Cpu)
                })
                .map(Engine::Ort)
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
    nyquist: Option<bool>,
    v1: Option<bool>,
) -> Result<DescreenOut, String> {
    let nyquist = nyquist.unwrap_or(false);
    let v1 = v1.unwrap_or(true);
    let mut engine = ensure_engine(&state, &app)?;
    let in_path = PathBuf::from(path);
    let tp = TileParams::default();
    let (engine_back, result) = tauri::async_runtime::spawn_blocking(move || {
        let mut cb = |done: usize, total: usize| {
            let _ = app.emit("descreen-progress", ProgressPayload { done, total });
        };
        let r = descreen_file(&mut engine, &in_path, blend, tp, &mut cb, nyquist, v1);
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

    /// Тесты гоняют цепочку без V1: проверяют классические стадии, а V1 имеет
    /// собственные юнит-тесты в `v1::tests`.
    fn v1_off_for_test() -> bool {
        false
    }

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
        let eng = InferenceEngine::new(&model.to_string_lossy(), Ep::Cpu).expect("cpu session");
        let mut eng = Engine::Ort(eng);
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
            false,
            v1_off_for_test(),
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
    fn descreen_nyquist_returns_source_size_and_drops_band() {
        use crate::prefilter::{luma_of, screen_band_energy};
        use ndarray::Array3;

        runtime::prepare_runtime().unwrap();
        let model = runtime::resolve_model(None).unwrap();
        let eng = InferenceEngine::new(&model.to_string_lossy(), Ep::Cpu).expect("cpu session");
        let mut eng = Engine::Ort(eng);
        let dir = std::env::temp_dir().join("descreen_nyquist_test");
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (512u32, 512u32);
        let mut rgb = image::RgbImage::new(w, h);
        for (x, y, px) in rgb.enumerate_pixels_mut() {
            let fx = x as f32 / w as f32;
            let content = 0.3 + 0.35 * fx;
            let g1 = 0.09 * (2.0 * std::f32::consts::PI * 0.15 * fx * w as f32).cos();
            let v = ((content + g1).clamp(0.0, 1.0) * 255.0).round() as u8;
            *px = image::Rgb([v, v, v]);
        }
        let inp = dir.join("am_nyq.png");
        rgb.save(&inp).unwrap();
        let before = luma_of(&{
            let r = image::open(&inp).unwrap().to_rgb8();
            Array3::from_shape_fn((3, h as usize, w as usize), |(ch, y, x)| {
                r.get_pixel(x as u32, y as u32)[ch] as f32 / 255.0
            })
        });
        let r0 = screen_band_energy(&before, 0.12, 0.18);
        assert!(r0 > 0.05, "grating must be present, got {r0}");

        let core = descreen_file(
            &mut eng,
            &inp,
            1.0,
            TileParams { tile: 256, overlap: 32 },
            &mut |_, _| {},
            true,
            v1_off_for_test(),
        )
        .unwrap();
        // вывод всегда в исходном размере
        assert_eq!((core.width, core.height), (w, h));
        let out_rgb = image::open(&core.output_path).unwrap().to_rgb8();
        assert_eq!(out_rgb.dimensions(), (w, h));
        let after = luma_of(&Array3::from_shape_fn((3, h as usize, w as usize), |(ch, y, x)| {
            out_rgb.get_pixel(x as u32, y as u32)[ch] as f32 / 255.0
        }));
        let r1 = screen_band_energy(&after, 0.12, 0.18);
        eprintln!("nyquist chain band 0.12-0.18: {r0} -> {r1}");
        assert!(r1 < r0 * 0.6, "nyquist chain must drop the band: {r0} -> {r1}");
    }

    #[test]
    fn stage5_pipeline_end_to_end() {
        runtime::prepare_runtime().unwrap();
        let model = runtime::resolve_model(None).unwrap();
        let eng = InferenceEngine::new(&model.to_string_lossy(), Ep::Cpu).expect("cpu session");
        let mut eng = Engine::Ort(eng);
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
            false,
            v1_off_for_test(),
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
