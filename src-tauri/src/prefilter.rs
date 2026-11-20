//! Classical prefilter chain (port of `ml/preproc`):
//! A0 raster measurement -> edge-adaptive Butterworth notch ->
//! LF-band subtraction -> (neural inference) -> paper-background return ->
//! linear tone match.
//!
//! The chain unloads the network: the stationary screen carrier and the
//! global LF banding are removed classically, the net owns only detail.

use ndarray::{Array2, Array3};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct PrefilterCfg {
    /// Detection ROI (v3.7 parity).
    pub roi_size: usize,
    /// v3.7 detector defaults.
    pub dc_radius_px: f32,
    pub tophat_win: usize,
    pub sensitivity: f32,
    pub sym_ratio: f32,
    pub max_peaks: usize,
    /// Notch radius at roi_size scale (narrower than v3.7 UI default).
    pub notch_radius_roi_px: f32,
    pub notch_order: i32,
    /// Adaptive notch grid.
    pub patch: usize,
    pub stride: usize,
    pub base_radius_roi_px: f32,
    pub radius_ref_roi: usize,
    pub gaussian_notch: bool,
    pub edge_gain: f32,
    pub lo_q: f32,
    pub hi_q: f32,
    /// LF banding.
    pub lf_lo: f32,
    pub lf_hi: f32,
    pub lf_axis_tol_deg: f32,
    pub lf_min_rois: usize,
    pub lf_roi: usize,
    pub lf_notch_r: f32,
    pub lf_notch_order: i32,
    /// Paper background split.
    pub bg_sigma: f32,
    /// Spatially-varying net weight (highlight/shadow guards).
    pub blend_hi_lo: f32,
    pub blend_hi_hi: f32,
    pub blend_hi_strength: f32,
    pub blend_sh_lo: f32,
    pub blend_sh_hi: f32,
    pub blend_sh_strength: f32,
    /// Edge guard: lower net weight on contrast transitions (patch-averaged
    /// map — a pixel-precise gradient map is poisoned by raster dots).
    pub blend_edge_guard: f32,
    pub blend_edge_patch: usize,
    pub blend_edge_stride: usize,
    pub blend_edge_lo_q: f32,
    pub blend_edge_hi_q: f32,
    /// Gated scalar backoff (JOURNAL 2026-10-10, case 1821146u): if the
    /// finished frame's top-4 residual peaks exceed the prefiltered input
    /// by more than `backoff_gate` (relative), the net correction is
    /// re-applied with weight `backoff_w`. Parity with
    /// `ml/benchmark/eval_realscan.py --backoff-mode gated`.
    pub backoff_enabled: bool,
    pub backoff_gate: f32,
    pub backoff_w: f32,
}

impl Default for PrefilterCfg {
    fn default() -> Self {
        Self {
            roi_size: 256,
            dc_radius_px: 12.0,
            tophat_win: 7,
            sensitivity: 2.5,
            sym_ratio: 0.5,
            max_peaks: 12,
            notch_radius_roi_px: 6.0,
            notch_order: 3,
            patch: 256,
            stride: 128,
            base_radius_roi_px: 6.0,
            radius_ref_roi: 1024,
            gaussian_notch: true,
            edge_gain: 1.0,
            lo_q: 0.5,
            hi_q: 0.95,
            lf_lo: 0.006,
            lf_hi: 0.05,
            lf_axis_tol_deg: 8.0,
            lf_min_rois: 3,
            lf_roi: 1024,
            lf_notch_r: 0.004,
            lf_notch_order: 4,
            bg_sigma: 32.0,
            blend_hi_lo: 0.75,
            blend_hi_hi: 0.95,
            blend_hi_strength: 1.0,
            blend_sh_lo: 0.25,
            blend_sh_hi: 0.05,
            blend_sh_strength: 0.5,
            blend_edge_guard: 0.7,
            blend_edge_patch: 128,
            blend_edge_stride: 64,
            blend_edge_lo_q: 0.5,
            blend_edge_hi_q: 0.95,
            backoff_enabled: true,
            backoff_gate: 0.25,
            backoff_w: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// 2D FFT helpers (rows then cols, fftshifted log-magnitude)
// ---------------------------------------------------------------------------

fn fft2_forward(gray: &[f32], n: usize) -> Vec<Complex32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex32> = gray.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    let mut tmp = vec![Complex32::new(0.0, 0.0); n];
    for r in 0..n {
        tmp.copy_from_slice(&buf[r * n..(r + 1) * n]);
        fft.process(&mut tmp);
        buf[r * n..(r + 1) * n].copy_from_slice(&tmp);
    }
    for c in 0..n {
        for r in 0..n {
            tmp[r] = buf[r * n + c];
        }
        fft.process(&mut tmp);
        for r in 0..n {
            buf[r * n + c] = tmp[r];
        }
    }
    buf
}

fn fft2_inverse(buf: &mut [Complex32], n: usize) {
    // NOTE: rustfft leaves both directions unnormalized, so each inverse
    // pass carries an explicit 1/n (total 1/n^2 for the 2D roundtrip).
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(n);
    let scale = 1.0 / n as f32;
    let mut tmp = vec![Complex32::new(0.0, 0.0); n];
    for r in 0..n {
        tmp.copy_from_slice(&buf[r * n..(r + 1) * n]);
        fft.process(&mut tmp);
        for v in tmp.iter_mut() {
            *v *= scale;
        }
        buf[r * n..(r + 1) * n].copy_from_slice(&tmp);
    }
    for c in 0..n {
        for r in 0..n {
            tmp[r] = buf[r * n + c];
        }
        fft.process(&mut tmp);
        for r in 0..n {
            buf[r * n + c] = tmp[r] * scale;
        }
    }
}

/// Unshifted FFT with DC at index 0 is what rustfft produces; the helpers
/// above already return that layout, masks are built on the shifted frame.

fn logmag_shifted(raw: &[Complex32], n: usize) -> Vec<f32> {
    let h = n / 2;
    let mut out = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            let sx = (x + h) % n;
            let sy = (y + h) % n;
            out[y * n + x] = (raw[sy * n + sx].norm() + 1.0).ln();
        }
    }
    out
}

fn box_blur(a: &[f32], n: usize, win: usize) -> Vec<f32> {
    let r = win / 2;
    let h = n + 2 * r;
    let mut pad = vec![0.0f32; h * h];
    for y in 0..h {
        let sy = (y as isize - r as isize).clamp(0, n as isize - 1) as usize;
        for x in 0..h {
            let sx = (x as isize - r as isize).clamp(0, n as isize - 1) as usize;
            pad[y * h + x] = a[sy * n + sx];
        }
    }
    let mut ii = vec![0.0f32; (h + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0.0f32;
        for x in 0..h {
            row += pad[y * h + x];
            ii[(y + 1) * (h + 1) + x + 1] = ii[y * (h + 1) + x + 1] + row;
        }
    }
    let mut out = vec![0.0f32; n * n];
    let norm = (win * win) as f32;
    for y in 0..n {
        for x in 0..n {
            let s = ii[(y + win) * (h + 1) + x + win] - ii[y * (h + 1) + x + win]
                - ii[(y + win) * (h + 1) + x]
                + ii[y * (h + 1) + x];
            out[y * n + x] = s / norm;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Peak detection (v3.7 parity)
// ---------------------------------------------------------------------------

pub struct Peak {
    pub r: f32,
    pub angle_deg: f32,
    pub strength: f32,
}

fn freq_at(x: usize, y: usize, n: usize) -> (f32, f32, f32) {
    let fx = (x as f32 - n as f32 / 2.0) / n as f32;
    let fy = (y as f32 - n as f32 / 2.0) / n as f32;
    (fx, fy, (fx * fx + fy * fy).sqrt())
}

/// Frequency coords on the UNSHIFTED grid (DC at index 0), as produced by
/// rustfft directly.
fn freq_at_raw(x: usize, y: usize, n: usize) -> (f32, f32, f32) {
    let ux = |v: usize| {
        if v <= n / 2 {
            v as f32 / n as f32
        } else {
            (v as f32 - n as f32) / n as f32
        }
    };
    let (fx, fy) = (ux(x), ux(y));
    (fx, fy, (fx * fx + fy * fy).sqrt())
}

pub fn detect_peaks(diff: &[f32], n: usize, cfg: &PrefilterCfg) -> (Vec<Peak>, f32) {
    let dc = cfg.dc_radius_px / n as f32;
    let mut vals = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let (_, _, r) = freq_at(x, y, n);
            if r > dc {
                vals.push(diff[y * n + x]);
            }
        }
    }
    let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len().max(1) as f32;
    let thr = mean + var.sqrt() * cfg.sensitivity;

    let mut order: Vec<usize> = (0..n * n).collect();
    order.sort_by(|&a, &b| diff[b].partial_cmp(&diff[a]).unwrap());
    let mut visited = vec![false; n * n];
    let mut peaks = Vec::new();
    for &k in &order {
        let v = diff[k];
        if v <= thr {
            break;
        }
        let (x, y) = (k % n, k / n);
        if visited[k] {
            continue;
        }
        let (_, _, r) = freq_at(x, y, n);
        if r <= dc {
            continue;
        }
        // 3x3 local max.
        let mut is_max = true;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < n as isize && ny >= 0 && ny < n as isize {
                    if diff[ny as usize * n + nx as usize] > v + 1e-9 {
                        is_max = false;
                        break;
                    }
                }
            }
            if !is_max {
                break;
            }
        }
        if !is_max {
            continue;
        }
        let sx = (n - x) % n;
        let sy = (n - y) % n;
        if diff[sy * n + sx] <= thr * cfg.sym_ratio {
            continue;
        }
        let (fx, fy, r) = freq_at(x, y, n);
        peaks.push(Peak {
            r,
            angle_deg: fy.atan2(fx).to_degrees(),
            strength: v,
        });
        for dy in -1..=1 {
            for dx in -1..=1 {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < n as isize && ny >= 0 && ny < n as isize {
                    visited[ny as usize * n + nx as usize] = true;
                }
                let mx = sx as isize + dx;
                let my = sy as isize + dy;
                if mx >= 0 && mx < n as isize && my >= 0 && my < n as isize {
                    visited[my as usize * n + mx as usize] = true;
                }
            }
        }
        if peaks.len() >= cfg.max_peaks {
            break;
        }
    }
    (peaks, thr)
}

/// Predicted notch targets of a square screen: fundamental, 2nd harmonic,
/// two sqrt(2) diagonals.
pub fn harmonic_family(r: f32, angle_deg: f32) -> Vec<(f32, f32)> {
    let s2 = 2.0f32.sqrt();
    vec![(r, angle_deg), (2.0 * r, angle_deg), (s2 * r, angle_deg + 45.0), (s2 * r, angle_deg - 45.0)]
}

/// Per-channel notch lists from a center ROI: fundamental + strong co-screens
/// (>0.4x) + predicted harmonic families.
pub fn build_notch_lists(img: &Array3<f32>, cfg: &PrefilterCfg, roi: usize) -> Vec<Vec<(f32, f32)>> {
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let s = roi.min(h).min(w);
    let y0 = h / 2 - s / 2;
    let x0 = w / 2 - s / 2;
    (0..c)
        .map(|ch| {
            let mut v = vec![0.0f32; s * s];
            for y in 0..s {
                for x in 0..s {
                    v[y * s + x] = img[[ch, y0 + y, x0 + x]];
                }
            }
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            for val in v.iter_mut() {
                *val -= mean;
            }
            let raw = fft2_forward(&v, s);
            let lm = logmag_shifted(&raw, s);
            let bg = box_blur(&lm, s, cfg.tophat_win);
            let diff: Vec<f32> = lm.iter().zip(bg.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
            let (peaks, _) = detect_peaks(&diff, s, cfg);
            let mut lst = Vec::new();
            if let Some(fund) = peaks.first() {
                lst.extend(harmonic_family(fund.r, fund.angle_deg));
                for p in peaks.iter().skip(1) {
                    if p.strength > fund.strength * 0.4 && lst.len() < cfg.max_peaks {
                        lst.extend(harmonic_family(p.r, p.angle_deg));
                    }
                }
            }
            lst.truncate(cfg.max_peaks);
            lst
        })
        .collect()
}

/// Замер A0 для FiLM-cond (порт `ml/preproc/spectrum.py::estimate_raster`):
/// один центральный ROI `roi×roi`, tophat-контраст, те же peaks, что и в
/// `build_notch_lists`; confidence — запас фундаментали над порогом.
pub struct RasterEstimate {
    pub r: f32,
    pub angle_deg: f32,
    pub confidence: f32,
}

/// Пики по одному каналу-плоскости (CPU, ROI в центре) — для итеративного
/// поиска остаточной несущей: после вычитания сильных плит часть несущих
/// становится видимой только в остатке (BookCOver: 0.140@−134 отсутствует в
/// первичном списке, но доминирует после удаления соседних).
pub fn detect_peaks_plane(
    plane: &[f32], h: usize, w: usize, roi: usize, cfg: &PrefilterCfg
) -> (Vec<Peak>, f32) {
    let s = roi.min(h).min(w);
    let y0 = h / 2 - s / 2;
    let x0 = w / 2 - s / 2;
    let mut v = vec![0.0f32; s * s];
    for y in 0..s {
        for x in 0..s {
            v[y * s + x] = plane[(y0 + y) * w + (x0 + x)];
        }
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, s);
    let lm = logmag_shifted(&raw, s);
    let bg = box_blur(&lm, s, cfg.tophat_win);
    let diff: Vec<f32> = lm.iter().zip(bg.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
    detect_peaks(&diff, s, cfg)
}

pub fn estimate_raster(gray: &Array2<f32>, cfg: &PrefilterCfg, roi: usize) -> RasterEstimate {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let s = roi.min(h).min(w);
    estimate_raster_at(gray, cfg, h / 2 - s / 2, w / 2 - s / 2, s)
}

/// Один ROI заданного положения (2.43: вынесено из `estimate_raster`, чтобы
/// замер можно было делать в нескольких окнах кадра).
fn estimate_raster_at(
    gray: &Array2<f32>, cfg: &PrefilterCfg, y0: usize, x0: usize, s: usize,
) -> RasterEstimate {
    let mut v = vec![0.0f32; s * s];
    for y in 0..s {
        for x in 0..s {
            v[y * s + x] = gray[[y0 + y, x0 + x]];
        }
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, s);
    let lm = logmag_shifted(&raw, s);
    let bg = box_blur(&lm, s, cfg.tophat_win);
    let diff: Vec<f32> = lm.iter().zip(bg.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
    let (peaks, thr) = detect_peaks(&diff, s, cfg);
    match peaks.first() {
        None => RasterEstimate { r: 0.0, angle_deg: 0.0, confidence: 0.0 },
        Some(f) => RasterEstimate {
            r: f.r,
            angle_deg: f.angle_deg,
            confidence: ((f.strength - thr) / (thr * 5.0 + 1e-12)).clamp(0.0, 1.0),
        },
    }
}

/// Замер растра по сетке `side × side` окон вместо одного центрального.
///
/// Причина (2.42): на `Magazine (2)` центр 69 Мпикс кадра пришёлся на плашку
/// без растра — A0 вернул 0.244 при confidence 0.03, V1 не сработал, кольцевой
/// notch вырезал не ту полосу, а unsharp усилил неснятую решётку. Один ROI не
/// может отличить «растра нет» от «в этом окне его не видно».
///
/// Выбирается самый уверенный замер, подтверждённый соседними окнами: растр
/// физически один на страницу, поэтому совпадение радиуса (±0.02 c/px) и угла
/// (±8° по модулю 90° — у растровых розеток квадратная симметрия) считается
/// подтверждением. Одиночный замер получает пониженную уверенность, чтобы
/// гейты (`coh`) не принимали шум за несущую.
pub fn estimate_raster_multi(
    gray: &Array2<f32>, cfg: &PrefilterCfg, roi: usize, side: usize,
) -> RasterEstimate {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let s = roi.min(h).min(w);
    let side = side.max(1);
    if side == 1 || s < 128 {
        return estimate_raster(gray, cfg, roi);
    }
    let mut ests: Vec<RasterEstimate> = Vec::with_capacity(side * side);
    for iy in 0..side {
        for ix in 0..side {
            let y0 = (h - s) * iy / (side - 1);
            let x0 = (w - s) * ix / (side - 1);
            let e = estimate_raster_at(gray, cfg, y0, x0, s);
            if e.r > 0.02 && e.r < 0.49 {
                ests.push(e);
            }
        }
    }
    if ests.is_empty() {
        return RasterEstimate { r: 0.0, angle_deg: 0.0, confidence: 0.0 };
    }
    let ang_diff = |a: f32, b: f32| {
        let mut d = (a - b).abs() % 90.0;
        if d > 45.0 {
            d = 90.0 - d;
        }
        d
    };
    // Кластеризация замеров вместо выбора одного «лучшего».
    //
    // Растр физически один на страницу, поэтому совпавшие радиус (±0.02 c/px) и
    // угол (±8° по модулю 90° — квадратная симметрия розетки) — одно и то же
    // свидетельство, и сила свидетельства = сумма уверенностей окон, а не
    // максимум. Максимум ошибается в обе стороны: на Offset2 он выбрал
    // гармонику 0.1807 = 0.1284·√2 (2.43), а правило «минимальный радиус» на
    // Magazine (2) выбрало ложный НЧ-пик 0.0735 (2026-10-30, 2.45).
    let cluster_of = |seed: &RasterEstimate| -> (usize, f32) {
        let mut n = 0usize;
        let mut wsum = 0.0f32;
        for e in ests.iter() {
            if (e.r - seed.r).abs() <= 0.02 && ang_diff(e.angle_deg, seed.angle_deg) <= 8.0 {
                n += 1;
                wsum += e.confidence.max(1e-6);
            }
        }
        (n, wsum)
    };
    let mut best_seed = &ests[0];
    let mut best_cl = cluster_of(best_seed);
    for e in ests.iter().skip(1) {
        let cl = cluster_of(e);
        // Сначала суммарная уверенность (сколько окон согласно), при равенстве —
        // меньший радиус: гармоника фундаменталом быть не может.
        let better = cl.1 > best_cl.1 * 1.001
            || ((cl.1 - best_cl.1).abs() <= best_cl.1 * 0.001 && e.r < best_seed.r);
        if better {
            best_seed = e;
            best_cl = cl;
        }
    }
    // Гармоника сильнее фундаментала в отдельном окне: если меньший радиус
    // подтверждён хотя бы двумя окнами и кратен лучшему (r√2 или 2r), берём его.
    // Требование подтверждения — именно то, чего не хватило ложному 0.0735 на
    // Magazine (2): его видело одно окно из девяти.
    let mut ref_seed = best_seed;
    let mut ref_cl = best_cl;
    for e in ests.iter() {
        if e.r >= best_seed.r * 0.9 {
            continue;
        }
        let (n, wsum) = cluster_of(e);
        let ratio = best_seed.r / e.r;
        let harmonic = (1.30..=1.50).contains(&ratio) || (1.85..=2.15).contains(&ratio);
        if n >= 2 && wsum >= 0.5 * best_cl.1 && harmonic && e.r < ref_seed.r {
            ref_seed = e;
            ref_cl = (n, wsum);
        }
    }
    let (ref_r, ref_ang) = (ref_seed.r, ref_seed.angle_deg);
    let (sup, wsum) = ref_cl;
    let mut rsum = 0.0f32;
    let mut best_conf = 0.0f32;
    for e in ests.iter() {
        if (e.r - ref_r).abs() <= 0.02 && ang_diff(e.angle_deg, ref_ang) <= 8.0 {
            let wt = e.confidence.max(1e-6);
            rsum += e.r * wt;
            if e.confidence > best_conf {
                best_conf = e.confidence;
            }
        }
    }
    let r = if wsum > 0.0 { rsum / wsum } else { ref_r };
    // Одиночное окно — слабое свидетельство: растр один на страницу, поэтому
    // подтверждение соседями обязательно.
    let confidence = if sup <= 1 {
        best_conf * 0.45
    } else {
        (best_conf * (1.0 + 0.2 * (sup - 1) as f32)).min(1.0)
    };
    RasterEstimate { r, angle_deg: ref_ang, confidence }
}

// ---------------------------------------------------------------------------
// Butterworth notch machinery (v3.7 formula, r units)
// ---------------------------------------------------------------------------

fn notch_mask(
    n: usize,
    peaks: &[(f32, f32)],
    radius_r: f32,
    order: i32,
    gaussian: bool,
) -> Vec<f32> {
    let h = n / 2;
    let mut mask = vec![1.0f32; n * n];
    // gaussian sigma matched to the same -3dB width: s = R/sqrt(2 ln 2)
    let sigma = radius_r / (2.0 * 2.0f32.ln()).sqrt();
    // peak unit vectors hoisted out of the pixel loop
    let dirs: Vec<(f32, f32)> = peaks
        .iter()
        .map(|&(_, ang)| {
            let a = ang.to_radians();
            (a.cos(), a.sin())
        })
        .collect();
    for y in 0..n {
        for x in 0..n {
            // shifted frame: DC at center
            let fx = (x as isize - h as isize) as f32 / n as f32;
            let fy = (y as isize - h as isize) as f32 / n as f32;
            // product over peaks and mirrors (v3.7 / numpy parity)
            let mut m = 1.0f32;
            for (&(r, _), &(ax, ay)) in peaks.iter().zip(dirs.iter()) {
                for sgn in [1.0f32, -1.0] {
                    let dx = fx - sgn * r * ax;
                    let dy = fy - sgn * r * ay;
                    let d = (dx * dx + dy * dy).sqrt().max(1e-9);
                    let hh = if gaussian {
                        1.0 - (-d * d / (2.0 * sigma * sigma)).exp()
                    } else {
                        let q = radius_r / d;
                        1.0 / (1.0 + q.powi(2 * order))
                    };
                    m *= hh;
                }
            }
            mask[y * n + x] = m;
        }
    }
    mask
}

/// Apply a shifted-frame mask to a square patch (mean-preserving).
/// Shift/unshift follow numpy fftshift/ifftshift conventions, exact for both
/// even and odd sizes.
fn apply_mask_patch(patch: &mut [f32], n: usize, mask: &[f32]) {
    let mean: f32 = patch.iter().sum::<f32>() / patch.len() as f32;
    for v in patch.iter_mut() {
        *v -= mean;
    }
    let mut buf = fft2_forward(patch, n);
    let h = n / 2;
    let hu = (n + 1) / 2;
    // fftshift
    let mut sh = vec![Complex32::new(0.0, 0.0); n * n];
    for y in 0..n {
        for x in 0..n {
            sh[y * n + x] = buf[((y + h) % n) * n + (x + h) % n];
        }
    }
    for (b, m) in sh.iter_mut().zip(mask.iter()) {
        *b *= *m;
    }
    // ifftshift (inverse of the above; differs for odd n)
    for y in 0..n {
        for x in 0..n {
            buf[y * n + x] = sh[((y + hu) % n) * n + (x + hu) % n];
        }
    }
    fft2_inverse(&mut buf, n);
    for (v, b) in patch.iter_mut().zip(buf.iter()) {
        *v = b.re + mean;
    }
}

/// Rectangular (h×w) forward FFT: rows (length w) then columns (length h).
fn fft2_forward_rect(gray: &[f32], h: usize, w: usize) -> Vec<Complex32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft_w = planner.plan_fft_forward(w);
    let fft_h = planner.plan_fft_forward(h);
    let mut buf: Vec<Complex32> = gray.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    for r in 0..h {
        fft_w.process(&mut buf[r * w..(r + 1) * w]);
    }
    let mut col = vec![Complex32::new(0.0, 0.0); h];
    for c in 0..w {
        for r in 0..h {
            col[r] = buf[r * w + c];
        }
        fft_h.process(&mut col);
        for r in 0..h {
            buf[r * w + c] = col[r];
        }
    }
    buf
}

fn fft2_inverse_rect(buf: &mut [Complex32], h: usize, w: usize) {
    let mut planner = FftPlanner::<f32>::new();
    let fft_w = planner.plan_fft_inverse(w);
    let fft_h = planner.plan_fft_inverse(h);
    let sw = 1.0 / w as f32;
    let sh = 1.0 / h as f32;
    for r in 0..h {
        let row = &mut buf[r * w..(r + 1) * w];
        fft_w.process(row);
        for v in row.iter_mut() {
            *v *= sw;
        }
    }
    let mut col = vec![Complex32::new(0.0, 0.0); h];
    for c in 0..w {
        for r in 0..h {
            col[r] = buf[r * w + c];
        }
        fft_h.process(&mut col);
        for r in 0..h {
            buf[r * w + c] = col[r] * sh;
        }
    }
}

/// Mask on the true h×w shifted spectrum (numpy `notch_mask(h, w, ...)`).
fn notch_mask_rect(
    h: usize,
    w: usize,
    peaks: &[(f32, f32)],
    radius_r: f32,
    order: i32,
    gaussian: bool,
) -> Vec<f32> {
    let (hh, hw) = (h as f32 / 2.0, w as f32 / 2.0);
    let sigma = radius_r / (2.0 * 2.0f32.ln()).sqrt();
    let dirs: Vec<(f32, f32)> = peaks
        .iter()
        .map(|&(_, ang)| {
            let a = ang.to_radians();
            (a.cos(), a.sin())
        })
        .collect();
    let mut mask = vec![1.0f32; h * w];
    for y in 0..h {
        let fy = (y as f32 - hh) / h as f32;
        for x in 0..w {
            let fx = (x as f32 - hw) / w as f32;
            let mut m = 1.0f32;
            for (&(r, _), &(ax, ay)) in peaks.iter().zip(dirs.iter()) {
                for sgn in [1.0f32, -1.0] {
                    let dx = fx - sgn * r * ax;
                    let dy = fy - sgn * r * ay;
                    let d = (dx * dx + dy * dy).sqrt().max(1e-9);
                    let hh_ = if gaussian {
                        1.0 - (-d * d / (2.0 * sigma * sigma)).exp()
                    } else {
                        let q = radius_r / d;
                        1.0 / (1.0 + q.powi(2 * order))
                    };
                    m *= hh_;
                }
            }
            mask[y * w + x] = m;
        }
    }
    mask
}

fn apply_mask_rect(plane: &mut [f32], h: usize, w: usize, mask: &[f32]) {
    let mean: f32 = plane.iter().sum::<f32>() / plane.len() as f32;
    for v in plane.iter_mut() {
        *v -= mean;
    }
    let mut buf = fft2_forward_rect(plane, h, w);
    let (ch, cw) = (h / 2, w / 2);
    let mut sh = vec![Complex32::new(0.0, 0.0); h * w];
    for y in 0..h {
        for x in 0..w {
            sh[y * w + x] = buf[((y + ch) % h) * w + (x + cw) % w];
        }
    }
    for (b, m) in sh.iter_mut().zip(mask.iter()) {
        *b *= *m;
    }
    let (uh, uw) = ((h + 1) / 2, (w + 1) / 2);
    for y in 0..h {
        for x in 0..w {
            buf[y * w + x] = sh[((y + uh) % h) * w + (x + uw) % w];
        }
    }
    fft2_inverse_rect(&mut buf, h, w);
    for (v, b) in plane.iter_mut().zip(buf.iter()) {
        *v = b.re + mean;
    }
}

/// Кольцевой notch по всей плоскости: полоса |f| ∈ r ± bw глушится
/// Butterworth-ом порядка `order`.
///
/// Зачем отдельно от `adaptive_notch`: в Найквист-режиме несущая в рабочем
/// кадре известна по построению (`r_out = r_in / scale`), а точечный notch по
/// пикам A0 её не берёт — A0 читает диагональную гармонику (Pack: 0.4258 =
/// √2·0.30, спектр рабочего кадра: 0.425 @ 2330× против 0.301 @ 2048×), а
/// AM-боковые полосы вообще не ложатся в точки. Замер 2.40: точечный notch
/// оставлял 4.13 % модуляции (неотличимо от «пропущено»), кольцо 0.30±0.02 —
/// 1.55 % при потере 5 % резкости.
pub fn ring_notch(img: &Array3<f32>, r: f32, bw: f32, order: i32) -> Array3<f32> {
    ring_notch_radii(img, &[(r, bw)], order)
}

/// Кольцевой notch сразу по семейству радиусов (2.46).
///
/// Зачем: видимая решётка — не только фундаментал. На `Pack` сильнейшие пики в
/// ГОТОВОМ кадре стоят на 0.2048 = 0.1451·√2 (диагонали, ±30°/−120°), а кольцо
/// резало только 0.1451 — точки оставались на месте, хотя «остаток в полосе
/// несущей» падал на 75 %. Маска считается один раз (объединение колец
/// произведением пропусканий), поэтому цена та же, что у одного кольца.
///
/// Ширина задаётся для каждого кольца отдельно (`(radius, bw)`): у гармоники
/// пик уже, чем у фундаментала (AM-боковые полосы шире у более сильной
/// составляющей), и кольцо 0.02 на 0.205 срезает слишком много контента —
/// резкость 51 % → 46 % скана при том же остатке.
pub fn ring_notch_radii(
    img: &Array3<f32>, rings: &[(f32, f32)], order: i32,
) -> Array3<f32> {
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let (hc, wc) = (h as f32 / 2.0, w as f32 / 2.0);
    let mut mask = vec![1.0f32; h * w];
    for y in 0..h {
        let fy = (y as f32 - hc) / h as f32;
        for x in 0..w {
            let fx = (x as f32 - wc) / w as f32;
            let d = (fx * fx + fy * fy).sqrt();
            let mut keep = 1.0f32;
            for &(r, bw) in rings {
                let t = (d - r).abs() / bw.max(1e-4);
                let gain = 1.0 / (1.0 + t.powi(2 * order));
                keep *= 1.0 - gain;
            }
            mask[y * w + x] = keep;
        }
    }
    let mut out = Array3::<f32>::zeros((c, h, w));
    for ci in 0..c {
        let mut plane: Vec<f32> = (0..h * w).map(|i| img[[ci, i / w, i % w]]).collect();
        apply_mask_rect(&mut plane, h, w, &mask);
        for i in 0..h * w {
            out[[ci, i / w, i % w]] = plane[i];
        }
    }
    out
}

fn hann_1d(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos()))
        .collect()
}

fn grid_1d(len: usize, p: usize, s: usize) -> Vec<usize> {
    if len <= p {
        return vec![0];
    }
    let mut v: Vec<usize> = (0..=(len - p)).step_by(s).collect();
    if *v.last().unwrap() + p < len {
        v.push(len - p);
    }
    v
}

fn quantile(mut v: Vec<f32>, q: f32) -> f32 {    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (v.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f32)
}

/// Edge-adaptive notch: radius grows with local contrast.
/// Returns the filtered image; global peak lists, local radii.
pub fn adaptive_notch(img: &Array3<f32>, peaks_per_ch: &[Vec<(f32, f32)>], cfg: &PrefilterCfg) -> Array3<f32> {
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let (p, s) = (cfg.patch, cfg.stride);
    // local contrast map (mean gradient magnitude per patch), luma
    let ys = grid_1d(h, p, s);
    let xs = grid_1d(w, p, s);
    let gx = |yy: usize, xx: usize| -> f32 {
        let xm = xx.saturating_sub(1);
        let xp = (xx + 1).min(w - 1);
        let ym = yy.saturating_sub(1);
        let yp = (yy + 1).min(h - 1);
        let mut g = 0.0f32;
        for ch in 0..c {
            let dx = img[[ch, yy, xp]] - img[[ch, yy, xm]];
            let dy = img[[ch, yp, xx]] - img[[ch, ym, xx]];
            g += (dx * dx + dy * dy).sqrt();
        }
        g / c as f32
    };
    let mut cmap = Array2::<f32>::zeros((ys.len(), xs.len()));
    for (j, &y) in ys.iter().enumerate() {
        for (i, &x) in xs.iter().enumerate() {
            let y1 = (y + p).min(h);
            let x1 = (x + p).min(w);
            let mut acc = 0.0f32;
            let mut n = 0usize;
            for yy in (y..y1).step_by(4) {
                for xx in (x..x1).step_by(4) {
                    acc += gx(yy, xx);
                    n += 1;
                }
            }
            cmap[[j, i]] = acc / n.max(1) as f32;
        }
    }
    let flat: Vec<f32> = cmap.iter().cloned().collect();
    let (lo, hi) = (quantile(flat.clone(), cfg.lo_q), quantile(flat, cfg.hi_q));
    let c01: Array2<f32> = cmap.mapv(|v| ((v - lo) / (hi - lo + 1e-9)).clamp(0.0, 1.0));
    let gain_map = blur_grid(&c01, 1.0);
    let win1 = hann_1d(p);
    let win2: Vec<f32> = win1.iter().flat_map(|&a| win1.iter().map(move |&b| a * b)).collect();

    let mut acc = Array3::<f32>::zeros((c, h, w));
    let mut weight = Array2::<f32>::zeros((h, w));
    // mask cache: gain quantized to 8 levels — neighboring patches share masks
    let mut cache: std::collections::HashMap<(usize, u8), Vec<f32>> =
        std::collections::HashMap::new();
    for (j, &y) in ys.iter().enumerate() {
        for (i, &x) in xs.iter().enumerate() {
            let cn = ((gain_map[[j, i]]).clamp(0.0, 1.0) * 8.0).floor() as u8;
            let gain = 1.0 + cfg.edge_gain * cn as f32 / 8.0;
            // patch is always full p×p via edge-clamped sampling below.
            let radius_r = cfg.base_radius_roi_px * gain / cfg.radius_ref_roi as f32;
            for ch in 0..c {
                let mut patch = vec![0.0f32; p * p];
                for yy in 0..p {
                    for xx in 0..p {
                        let sy = (y + yy).min(h - 1);
                        let sx = (x + xx).min(w - 1);
                        patch[yy * p + xx] = img[[ch, sy, sx]];
                    }
                }
                let mask = cache.entry((ch, cn)).or_insert_with(|| {
                    notch_mask(p, &peaks_per_ch[ch], radius_r, cfg.notch_order, cfg.gaussian_notch)
                });
                apply_mask_patch(&mut patch, p, mask);
                for yy in 0..p {
                    for xx in 0..p {
                        let sy = (y + yy).min(h - 1);
                        let sx = (x + xx).min(w - 1);
                        // Hann weight only inside the real frame region
                        let inside = y + yy < h && x + xx < w;
                        let wv = if inside { win2[yy * p + xx] } else { 0.0 };
                        acc[[ch, sy, sx]] += patch[yy * p + xx] * wv;
                        if ch == 0 {
                            weight[[sy, sx]] += wv;
                        }
                    }
                }
            }
        }
    }
    let mut out = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                // Окно Ханна равно нулю на краю патча, поэтому крайние пиксели
                // кадра могут не набрать веса вообще. Деление на 1e-9 давало
                // чёрную рамку (замер: Magazine (2) row0 = col0 = 0.0 против
                // 0.78/0.48 в глубине). Там, где веса нет, оставляем исходный
                // пиксель: notch к нему всё равно не применился.
                let wsum = weight[[y, x]];
                out[[ch, y, x]] = if wsum > 1e-3 {
                    acc[[ch, y, x]] / wsum
                } else {
                    img[[ch, y, x]]
                };
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// LF banding: detect global axial bands, subtract with a narrow notch
// ---------------------------------------------------------------------------

pub struct LFBand {
    pub r: f32,
    pub angle_deg: f32,
}

fn rois_1024(h: usize, w: usize, s: usize) -> Vec<(usize, usize)> {
    let s = s.min(h).min(w);
    let mut v = vec![(0, 0), (0, w - s), (h - s, 0), (h - s, w - s), (h / 2 - s / 2, w / 2 - s / 2)];
    v.sort();
    v.dedup();
    v
}

pub fn detect_lf_bands(gray: &Array2<f32>, cfg: &PrefilterCfg) -> Vec<LFBand> {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let s = cfg.lf_roi.min(h).min(w);
    let mut votes: Vec<(f32, char)> = Vec::new();
    for (y0, x0) in rois_1024(h, w, s) {
        let mut v = vec![0.0f32; s * s];
        for y in 0..s {
            for x in 0..s {
                v[y * s + x] = gray[[y0 + y, x0 + x]];
            }
        }
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        for val in v.iter_mut() {
            *val -= mean;
        }
        let raw = fft2_forward(&v, s);
        let lm = logmag_shifted(&raw, s);
        let mut best = (0usize, f32::NEG_INFINITY);
        for y in 0..s {
            for x in 0..s {
                let (fx, fy, r) = freq_at(x, y, s);
                if r < cfg.lf_lo || r >= cfg.lf_hi {
                    continue;
                }
                let ang = fy.atan2(fx).to_degrees();
                let d_axis = ((ang + 90.0) % 180.0 - 90.0).abs().min(((ang + 180.0) % 180.0 - 90.0).abs());
                if d_axis > cfg.lf_axis_tol_deg {
                    continue;
                }
                let k = y * s + x;
                if lm[k] > best.1 {
                    best = (k, lm[k]);
                }
            }
        }
        if best.1 == f32::NEG_INFINITY {
            continue;
        }
        // gate: must stand out of the LF floor (median in band)
        let mut band_vals = Vec::new();
        for y in 0..s {
            for x in 0..s {
                let (fx, fy, r) = freq_at(x, y, s);
                if r < cfg.lf_lo || r >= cfg.lf_hi {
                    continue;
                }
                let ang = fy.atan2(fx).to_degrees();
                let d_axis = ((ang + 90.0) % 180.0 - 90.0).abs().min(((ang + 180.0) % 180.0 - 90.0).abs());
                if d_axis <= cfg.lf_axis_tol_deg {
                    band_vals.push(lm[y * s + x]);
                }
            }
        }
        band_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = band_vals[band_vals.len() / 2];
        if best.1 < med + 1.0 {
            continue;
        }
        let (bx, by) = (best.0 % s, best.0 / s);
        let (fx, fy, r) = freq_at(bx, by, s);
        let ang = fy.atan2(fx).to_degrees();
        let axis = if (ang.abs() - 90.0).abs() < 45.0 { 'v' } else { 'h' };
        votes.push((r, axis));
    }
    // tolerance clustering
    let mut bands = Vec::new();
    let mut unused = votes;
    while let Some((sr, sax)) = unused.first().cloned() {
        unused.remove(0);
        let mut group = vec![(sr, sax)];
        let mut rest = Vec::new();
        for v in unused.drain(..) {
            if v.1 == sax && (v.0 - sr).abs() < 0.0025 {
                group.push(v);
            } else {
                rest.push(v);
            }
        }
        unused = rest;
        if group.len() >= cfg.lf_min_rois {
            let mr: f32 = group.iter().map(|g| g.0).sum::<f32>() / group.len() as f32;
            bands.push(LFBand { r: mr, angle_deg: if sax == 'v' { 90.0 } else { 0.0 } });
        }
    }
    bands
}

pub fn subtract_lf(img: &Array3<f32>, bands: &[LFBand], cfg: &PrefilterCfg) -> Array3<f32> {
    if bands.is_empty() {
        return img.clone();
    }
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let peaks: Vec<(f32, f32)> = bands.iter().map(|b| (b.r, b.angle_deg)).collect();
    // NOTE: notch on the true h×w spectrum. An earlier port padded to a square
    // canvas with replicate edges; the padding seam produced a wrap-around lobe
    // at the left/right frame borders (Pack parity vs Python, JOURNAL 2.28).
    let mask = notch_mask_rect(h, w, &peaks, cfg.lf_notch_r, cfg.lf_notch_order, false);
    let mut out = img.clone();
    for ch in 0..c {
        let mut plane = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = img[[ch, y, x]];
            }
        }
        apply_mask_rect(&mut plane, h, w, &mask);
        for y in 0..h {
            for x in 0..w {
                out[[ch, y, x]] = plane[y * w + x];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Paper background return + linear tone
// ---------------------------------------------------------------------------

/// Гауссов НЧ-фильтр (σ в пикселях) — реализация на FFT.
///
/// ВАЖНО: FFT считает ЦИКЛИЧЕСКУЮ свёртку. Без паддинга НЧ-оценка на краю
/// кадра подмешивает содержимое ПРОТИВОПОЛОЖНОГО края: при σ=32 это первые и
/// последние ~100 px. Именно из-за этого в 2.48 по левому и правому краю
/// кадра пошли полосы (на 1821146u выход у левой кромки 0.16 против 0.22 в
/// глубине, на Magazine — 0.94 против 0.88). Поэтому кадр предварительно
/// растягивается replicate-заливкой на `3σ` и обрезается обратно.
fn gaussian_lowpass_channel(ch: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let pad = (3.0 * sigma).ceil() as usize;
    if pad == 0 {
        return gaussian_lowpass_core(ch, h, w, sigma);
    }
    let (hp, wp) = (h + 2 * pad, w + 2 * pad);
    let mut padded = vec![0.0f32; hp * wp];
    for y in 0..hp {
        let sy = y.clamp(pad, h + pad - 1) - pad;
        for x in 0..wp {
            let sx = x.clamp(pad, w + pad - 1) - pad;
            padded[y * wp + x] = ch[sy * w + sx];
        }
    }
    let blurred = gaussian_lowpass_core(&padded, hp, wp, sigma);
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        out[y * w..(y + 1) * w].copy_from_slice(
            &blurred[(y + pad) * wp + pad..(y + pad) * wp + pad + w],
        );
    }
    out
}

fn gaussian_lowpass_core(ch: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let mut buf: Vec<Complex32> = ch.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    // separable rows (w) then cols (h); gaussian factors on unshifted grids
    let mut pr = FftPlanner::<f32>::new();
    let fr = pr.plan_fft_forward(w);
    let fc = pr.plan_fft_forward(h);
    let mut tmp = vec![Complex32::new(0.0, 0.0); h.max(w)];
    for y in 0..h {
        tmp[..w].copy_from_slice(&buf[y * w..(y + 1) * w]);
        fr.process(&mut tmp[..w]);
        // gaussian row factor (unshifted frequencies k/w)
        for x in 0..w {
            let fx = if x <= w / 2 { x as f32 / w as f32 } else { (x as f32 - w as f32) / w as f32 };
            let g = (-0.5 * (fx * 2.0 * std::f32::consts::PI * sigma).powi(2)).exp();
            tmp[x] *= g;
        }
        buf[y * w..(y + 1) * w].copy_from_slice(&tmp[..w]);
    }
    let mut pi = FftPlanner::<f32>::new();
    let ir = pi.plan_fft_inverse(w);
    let ic = pi.plan_fft_inverse(h);
    for y in 0..h {
        tmp[..w].copy_from_slice(&buf[y * w..(y + 1) * w]);
        ir.process(&mut tmp[..w]);
        let s = 1.0 / w as f32;
        for v in tmp[..w].iter_mut() {
            *v *= s;
        }
        buf[y * w..(y + 1) * w].copy_from_slice(&tmp[..w]);
    }
    for x in 0..w {
        for y in 0..h {
            tmp[y] = buf[y * w + x];
        }
        fc.process(&mut tmp[..h]);
        for y in 0..h {
            let fy = if y <= h / 2 { y as f32 / h as f32 } else { (y as f32 - h as f32) / h as f32 };
            let g = (-0.5 * (fy * 2.0 * std::f32::consts::PI * sigma).powi(2)).exp();
            tmp[y] *= g;
        }
        ic.process(&mut tmp[..h]);
        let s = 1.0 / h as f32;
        for y in 0..h {
            buf[y * w + x] = tmp[y] * s;
        }
    }
    buf.iter().map(|b| b.re).collect()
}

pub fn return_background(src: &Array3<f32>, net: &Array3<f32>, sigma: f32) -> Array3<f32> {
    let (c, h, w) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut out = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        let s: Vec<f32> = (0..h * w).map(|k| src[[ch, k / w, k % w]]).collect();
        let n: Vec<f32> = (0..h * w).map(|k| net[[ch, k / w, k % w]]).collect();
        let bg_s = gaussian_lowpass_channel(&s, h, w, sigma);
        let bg_n = gaussian_lowpass_channel(&n, h, w, sigma);
        for k in 0..h * w {
            out[[ch, k / w, k % w]] = (bg_s[k] + (n[k] - bg_n[k])).clamp(0.0, 1.0);
        }
    }
    out
}

/// НЧ-фильтр Баттерворта с резкой границей полосы (2.50).
///
/// `t_cut` — период среза в пикселях, `order` — порядок (8 при срезе по
/// суперячейке растра). В отличие от гаусса, который режет полого, здесь
/// полоса до `~t_cut` проходит почти целиком, а выше — падает на десятки дБ:
/// при `t_cut = 2.4/r`, `order = 8` усиление на несущей (1/r) меньше 1e-6,
/// на суперячейке (2/r) ≈ 0.08, на самом срезе 0.5.
///
/// Края закрываются replicate-паддингом на `2·t_cut`: без него свёртка идёт
/// по кругу и на кромке кадра подмешивается противоположная сторона (та же
/// ошибка, что чинилась в 2.49 для гауссова пути).
fn butter_lowpass_channel(
    ch: &[f32],
    h: usize,
    w: usize,
    t_cut: f32,
    order: f32,
) -> Vec<f32> {
    let pad = (2.0 * t_cut).ceil().max(16.0) as usize;
    let (hp, wp) = (h + 2 * pad, w + 2 * pad);
    let mut plane = vec![0.0f32; hp * wp];
    for y in 0..hp {
        let sy = y.saturating_sub(pad).min(h - 1);
        for x in 0..wp {
            let sx = x.saturating_sub(pad).min(w - 1);
            plane[y * wp + x] = ch[sy * w + sx];
        }
    }
    let mut buf = fft2_forward_rect(&plane, hp, wp);
    for y in 0..hp {
        let fy = if y <= hp / 2 {
            y as f32 / hp as f32
        } else {
            (y as f32 - hp as f32) / hp as f32
        };
        for x in 0..wp {
            let fx = if x <= wp / 2 {
                x as f32 / wp as f32
            } else {
                (x as f32 - wp as f32) / wp as f32
            };
            let f = (fy * fy + fx * fx).sqrt() * t_cut;
            if f > 0.0 {
                buf[y * wp + x] *= 1.0 / (1.0 + f.powf(2.0 * order));
            }
        }
    }
    fft2_inverse_rect(&mut buf, hp, wp);
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = buf[(y + pad) * wp + x + pad].re;
        }
    }
    out
}

/// Локальная тоновая страховка для нативного режима (2.50).
///
/// Что лечит (замер на 11 реальных сканах): в областях без структуры — белая
/// бумага, плашки, блики — сеть уводит средний уровень на 2–6 уровней и
/// рисует пятна с периодом 16–64 px. На белой плашке `Calendar1` (320²)
/// энергия полосы 16–32 px у выхода 2.48/2.49 в 2.1 раза выше, чем у скана
/// (10.2 против 4.8), 32–64 px — в 1.9 раза (7.6 против 4.0). Глазами это и
/// есть «регулярная пятнистость» на белых костюмах и плашках.
///
/// Операция: `out = net + LP(src - net)`, `LP` — Баттерворт: НЧ-составляющая
/// (тон, освещённость, реальная фактура бумаги) берётся из СКАНА, всё, что
/// выше среза, — из выхода сети. ВЧ сети (включая снятый растр) не трогается.
///
/// Почему не как в 2.49 (гаусс σ=3 + маска структурности):
/// * гаусс σ=3 пропускал 35 % полосы 13 px — суперячейки растра (2 периода
///   точек), и возвращал в кадр саму решётку: на белом 1821146u оставался
///   пик 13.3 px амплитудой 1.5 уровня;
/// * маска считалась по ВЧ скана, а у скана ВЧ — это растр: на плашке
///   `Calendar1` активность 0.026 при пороге 0.020, то есть маска = 1 ровно
///   на самых белых и самых растровых местах, и страховка там не работала
///   вовсе (guard 16–32 px = 11.3 против raw 11.6).
///
/// Срез `t_cut` задаётся над суперячейкой растра (2/r), поэтому решётка в
/// кадр не возвращается: T=12 даёт видимую сетку на полосе `Pack`, T=16–18
/// (r≈0.145) — уже нет.
///
/// 2.53: цвет отдельно от тона. Растр четырёхкрасочный, и его след живёт не
/// только в яркости: между растровыми точками разных красок остаётся
/// цветная «розетка» с периодом 8–32 px. Поканальная страховка возвращала её
/// из скана целиком, потому что НЧ цветности проходили ниже `t_cut`
/// одинаково с тоном. Замер на `Pack` (полоса 8–32 px, окно 1472,2400,
/// амплитуда как 2·СКО в уровнях): у скана решётка в u/v читается отчётливо,
/// у 2.52 в `v` остаётся та же регулярная сетка, у 2.53 — ровный шум без
/// периода. Тон/освещённость — величина общая для каналов, поэтому из скана
/// берётся только яркостная НЧ-поправка со срезом `t_cut`; цветность
/// возвращается лишь на самой низкой частоте (смещение баланса, окно
/// `t_chroma`, по умолчанию 4·t_cut), а всё, что мельче, — из сети.
pub fn local_tone_guard(
    net: &Array3<f32>,
    src: &Array3<f32>,
    t_cut: f32,
    order: f32,
) -> Array3<f32> {
    let t_chroma = std::env::var("DESCREEN_TG_TCH")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(4.0 * t_cut)
        .clamp(32.0, 160.0);
    let (c, h, w) = (net.shape()[0], net.shape()[1], net.shape()[2]);
    let mut d: Vec<Vec<f32>> = Vec::with_capacity(c);
    for ch in 0..c {
        d.push(
            (0..h * w)
                .map(|k| src[[ch, k / w, k % w]] - net[[ch, k / w, k % w]])
                .collect(),
        );
    }
    // Общая (яркостная) часть поправки и поканальный остаток — цветность.
    let mut m = vec![0.0f32; h * w];
    for k in 0..h * w {
        m[k] = if c >= 3 {
            0.299 * d[0][k] + 0.587 * d[1][k] + 0.114 * d[2][k]
        } else {
            d[0][k]
        };
    }
    let lp_m = butter_lowpass_channel(&m, h, w, t_cut, order);
    // Цветность: срез на порядок ниже по частоте, box-фильтра достаточно
    // (префиксные суммы, цена линейна) — Баттерворт стоил бы ещё 3 полных FFT.
    // Два прохода box ≈ треугольное ядро: у полосы 8–32 px (цветная «розетка»)
    // остаётся <1 % против 10–20 % у одного прохода, при той же цене.
    let cwin = (t_chroma as usize).max(3) | 1;
    let mut lp_c: Vec<Vec<f32>> = Vec::with_capacity(c);
    for ch in 0..c {
        let q: Vec<f32> = (0..h * w).map(|k| d[ch][k] - m[k]).collect();
        let b1 = box_blur_rect(&q, h, w, cwin);
        lp_c.push(box_blur_rect(&b1, h, w, cwin));
    }
    let mut out = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        for k in 0..h * w {
            let n = net[[ch, k / w, k % w]];
            out[[ch, k / w, k % w]] = (n + lp_m[k] + lp_c[ch][k]).clamp(0.0, 1.0);
        }
    }
    out
}

/// Раздельный box-фильтр с железным краем (clamp): окно нечётное.
///
/// Нужен для оценки медленно меняющихся амплитуд решётки. Префиксные суммы по
/// строке/столбцу, поэтому цена линейна по числу пикселей и не зависит от окна.
fn box_blur_rect(a: &[f32], h: usize, w: usize, win: usize) -> Vec<f32> {
    let r = (win.max(1) / 2) as isize;
    let mut tmp = vec![0.0f32; h * w];
    let mut pre = vec![0.0f64; w.max(h) + 1];
    for y in 0..h {
        let base = y * w;
        pre[0] = 0.0;
        for x in 0..w {
            pre[x + 1] = pre[x] + a[base + x] as f64;
        }
        for x in 0..w {
            let lo = (x as isize - r).clamp(0, w as isize - 1) as usize;
            let hi = (x as isize + r).clamp(0, w as isize - 1) as usize;
            tmp[base + x] = ((pre[hi + 1] - pre[lo]) / (hi - lo + 1) as f64) as f32;
        }
    }
    let mut out = vec![0.0f32; h * w];
    for x in 0..w {
        pre[0] = 0.0;
        for y in 0..h {
            pre[y + 1] = pre[y] + tmp[y * w + x] as f64;
        }
        for y in 0..h {
            let lo = (y as isize - r).clamp(0, h as isize - 1) as usize;
            let hi = (y as isize + r).clamp(0, h as isize - 1) as usize;
            out[y * w + x] = ((pre[hi + 1] - pre[lo]) / (hi - lo + 1) as f64) as f32;
        }
    }
    out
}

/// Подавление регулярной решётки сети (2.51).
///
/// Что лечит (замер на идеально плоском входе 1024², `DESCREEN_NET_ONLY=1`):
/// на постоянном поле выход сети не постоянен — на нём стоят пики 4.00 px по
/// осям (амплитуда 1.1–2.6 уровня), 2.83 px по диагонали (0.3–0.9) и 2.00 px
/// (1.0–2.6). Все они лежат на узлах решётки с периодом 4 px, то есть это
/// checkerboard апсемплинга сети, а не растр: на реальных сканах те же пики
/// видны поверх гладких белых костюмов/плашек и читаются как «регулярная
/// крапчатость».
///
/// Операция — разложение на 16 фаз (y%4, x%4). Каждая фаза — своя подрешётка
/// с шагом 4 px; её локальное среднее (box с окном `win` в полных пикселях —
/// это ровно амплитуда соответствующей гармоники решётки) вычитается из
/// кадра, а вместо неё возвращается общее локальное среднее с тем же
/// эффективным окном:
///
/// ```text
/// out = ch - mean_phase[i][j] + mean_full      (окна согласованы: 4·win/4)
/// ```
///
/// Всё, что не периодично с периодом 4 px, из кадра не уходит: в фазовых
/// средних оно сокращается. Согласование окон обязательно — при окне общего
/// среднего вчетверо меньше полосы фазовых средних разность начинает содержать
/// полосу 4·pw..pw, и scrub не убирает пятна, а удваивает их (проверено).
///
/// Замер после стадии (кропы 512², 2.50 → 2.51): пики 4/2.83/2 px падают с
/// 0.78–2.31 до ≤0.01 уровня, sd и острота не меняются (Calendar1 3.48→3.48,
/// 2.23→2.21; Pack 3.60→3.57), средний уровень сохраняется.
pub fn grid_scrub(img: &Array3<f32>, win: usize) -> Array3<f32> {
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let win = win.max(4) | 1; // окно в полных пикселях, нечётное
    let pw = (win / 4).max(3) | 1; // окно в фазовой подрешётке
    let win_full = 4 * pw - 1; // то же эффективное окно (span 4·pw-1 ≈ 4·pw), нечётное
    let mut res = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        // Канал в плоский буфер: индексация ndarray с делением k/w в горячем
        // цикле стоила на 18 МП больше 10 с.
        let mut src = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = img[[ch, y, x]];
            }
        }
        let lp_full = box_blur_rect(&src, h, w, win_full);
        let mut dst = src.clone();
        for i in 0..4.min(h) {
            let hs = (h - i + 3) / 4;
            for j in 0..4.min(w) {
                let ws = (w - j + 3) / 4;
                if hs == 0 || ws == 0 {
                    continue;
                }
                let mut sub = vec![0.0f32; hs * ws];
                for m in 0..hs {
                    let base = (i + 4 * m) * w + j;
                    for n in 0..ws {
                        sub[m * ws + n] = src[base + 4 * n];
                    }
                }
                let lp = box_blur_rect(&sub, hs, ws, pw);
                for m in 0..hs {
                    let base = (i + 4 * m) * w + j;
                    for n in 0..ws {
                        let k = base + 4 * n;
                        // Общее локальное среднее берём в тех же координатах:
                        // сдвиг меньше пикселя не критичен (окно ≥30 px), а
                        // согласованность окон убирает утечку полосы 4·pw..pw.
                        dst[k] = (src[k] - lp[m * ws + n] + lp_full[k]).clamp(0.0, 1.0);
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                res[[ch, y, x]] = dst[y * w + x];
            }
        }
    }
    res
}

/// Тоновое выравнивание выхода к скану.
///
/// 2.52: масштаб контраста считается по **НЧ** обеих картинок, а не по всему
/// кадру. Причина: после снятия растра СКО выхода законно ниже СКО скана —
/// растр давал в него основной вклад (замер `1821146u`: 65.78 против 68.60, то
/// есть a = 1.043). Подгонка «в лоб» возвращала этот разброс умножением всего
/// кадра, и на плашках бумажная фактура с полосами 16–128 px становилась на
/// 4–5 % контрастнее скана (замер октав: 1.05× с этим умножением против 1.00×
/// без него) — то есть стадия сама подкрашивала ровно те пятна, которые убирает
/// страховка тона. Тон и освещённость живут в НЧ, поэтому масштаб берётся по
/// box-НЧ (окно 33–129 px), а среднее — по всему кадру (оно должно совпасть
/// точно).
pub fn linear_tone(out: &Array3<f32>, src: &Array3<f32>) -> Array3<f32> {
    let (c, h, w) = (out.shape()[0], out.shape()[1], out.shape()[2]);
    let win = ((h.min(w) / 4).max(33)).min(129) | 1;
    let n = (h * w) as f64;
    let mut res = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        let mut o = vec![0.0f32; h * w];
        let mut s = vec![0.0f32; h * w];
        let (mut mo, mut ms) = (0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let vo = out[[ch, y, x]];
                let vs = src[[ch, y, x]];
                o[y * w + x] = vo;
                s[y * w + x] = vs;
                mo += vo as f64;
                ms += vs as f64;
            }
        }
        mo /= n;
        ms /= n;
        let lo = box_blur_rect(&o, h, w, win);
        let ls = box_blur_rect(&s, h, w, win);
        let (mlo, mls) = {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for k in 0..h * w {
                a += lo[k] as f64;
                b += ls[k] as f64;
            }
            (a / n, b / n)
        };
        let (mut vo, mut vs) = (0.0f64, 0.0f64);
        for k in 0..h * w {
            vo += (lo[k] as f64 - mlo).powi(2);
            vs += (ls[k] as f64 - mls).powi(2);
        }
        let (sdo, sds) = ((vo / n).sqrt(), (vs / n).sqrt());
        // Страховка: если НЧ выхода пусто (плоская плашка), масштаб не трогаем.
        let a = if sdo > 1e-6 {
            (sds / sdo).clamp(0.9, 1.1) as f32
        } else {
            1.0
        };
        for k in 0..h * w {
            res[[ch, k / w, k % w]] = ((o[k] - mo as f32) * a + ms as f32).clamp(0.0, 1.0);
        }
    }
    res
}

/// Residual screen energy: top symmetric peak (xmean) above the raster band.
/// No-GT metric for real scans; higher = more halftone left.
pub fn residual_peak_xmean(gray: &Array2<f32>) -> f32 {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let n = h.min(w);
    let y0 = h / 2 - n / 2;
    let x0 = w / 2 - n / 2;
    let mut v = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            v[y * n + x] = gray[[y0 + y, x0 + x]];
        }
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, n);
    let mut best = 0.0f32;
    let mut msum = 0.0f32;
    for k in 0..n * n {
        let m = raw[k].norm();
        msum += m;
        let (x, y) = (k % n, k / n);
        let (_, _, r) = freq_at_raw(x, y, n);
        if r > 0.05 && m > best {
            best = m;
        }
    }
    best / (msum / (n * n) as f32 + 1e-9)
}

/// Residual top-k metric params (parity with `ml/preproc/metrics.py`).
pub const TOP4_ROI: usize = 1024;
pub const TOP4_R_MIN: f32 = 0.05;
pub const TOP4_EXCL: f32 = 0.02;

/// Top-4 residual peak sum (xmean normalization, disk+mirror exclusion) —
/// Rust port of `metrics.residual_peaks_multi(k=4)`. Higher = more halftone
/// left; the gated backoff compares this metric on the finished frame vs the
/// prefiltered input.
pub fn top4_peak_sum(gray: &Array2<f32>) -> f32 {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let s = h.min(w).min(TOP4_ROI);
    let y0 = h / 2 - s / 2;
    let x0 = w / 2 - s / 2;
    // numpy slice span floor(s/2)*2 — keeps the ROI square for odd sizes.
    let n = s - (s & 1);
    let mut v = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            v[y * n + x] = gray[[y0 + y, x0 + x]];
        }
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, n);
    let half = n / 2;
    let mut mag = vec![0.0f32; n * n];
    let mut msum = 0.0f32;
    for y in 0..n {
        for x in 0..n {
            // fftshift parity: out[i] = in[(i - n/2) mod n]
            let m = raw[((y + n - half) % n) * n + ((x + n - half) % n)].norm();
            mag[y * n + x] = m;
            msum += m;
        }
    }
    let mean_mag = msum / (n * n) as f32 + 1e-9;
    for y in 0..n {
        for x in 0..n {
            let (_, _, r) = freq_at(x, y, n);
            if r < TOP4_R_MIN {
                mag[y * n + x] = 0.0;
            }
        }
    }
    let excl2 = TOP4_EXCL * TOP4_EXCL;
    let mut sum = 0.0f32;
    for _ in 0..4 {
        let (mut bj, mut bi) = (0usize, 0usize);
        let mut bv = 0.0f32;
        for y in 0..n {
            for x in 0..n {
                if mag[y * n + x] > bv {
                    bv = mag[y * n + x];
                    bj = y;
                    bi = x;
                }
            }
        }
        if bv <= 0.0 {
            break;
        }
        sum += bv / mean_mag;
        let (fx0, fy0, _) = freq_at(bi, bj, n);
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, _) = freq_at(x, y, n);
                let d2 = (fx - fx0) * (fx - fx0) + (fy - fy0) * (fy - fy0);
                let d2m = (fx + fx0) * (fx + fx0) + (fy + fy0) * (fy + fy0);
                if d2 < excl2 || d2m < excl2 {
                    mag[y * n + x] = 0.0;
                }
            }
        }
    }
    sum
}

/// Screen-band energy: |F|^2 inside an annulus, relative to total.
/// Targeted residual metric (unlike the global top peak, blind to content).
pub fn screen_band_energy(gray: &Array2<f32>, r_lo: f32, r_hi: f32) -> f32 {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let n = h.min(w);
    let y0 = h / 2 - n / 2;
    let x0 = w / 2 - n / 2;
    let mut v = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            v[y * n + x] = gray[[y0 + y, x0 + x]];
        }
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, n);
    let mut band = 0.0f32;
    let mut tot = 0.0f32;
    for k in 0..n * n {
        let e = raw[k].norm_sqr();
        tot += e;
        let (x, y) = (k % n, k / n);
        let (_, _, r) = freq_at_raw(x, y, n);
        if r >= r_lo && r < r_hi {
            band += e;
        }
    }
    band / (tot + 1e-9)
}

/// АБСОЛЮТНАЯ энергия полосы (сумма |F|² в кольце), без нормировки на текущую
/// энергию кадра.
///
/// Отличие от [`screen_band_energy`] принципиально: доля от текущей энергии
/// бессмысленна после удаления растра — знаменатель сжимается вместе с
/// сигналом, и «доля» может даже вырасти, хотя абсолютная энергия упала.
/// Для оценки «сколько полосы снято» нужна абсолютная величина до и после
/// на одном и том же ROI.
pub fn band_energy_abs(gray: &Array2<f32>, r_lo: f32, r_hi: f32) -> f32 {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let n = h.min(w).min(1024);
    let y0 = h / 2 - n / 2;
    let x0 = w / 2 - n / 2;
    let mut v = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            v[y * n + x] = gray[[y0 + y, x0 + x]];
        }
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, n);
    let mut band = 0.0f32;
    for k in 0..n * n {
        let (x, y) = (k % n, k / n);
        let (_, _, r) = freq_at_raw(x, y, n);
        if r >= r_lo && r < r_hi {
            band += raw[k].norm_sqr();
        }
    }
    band
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo + 1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Small separable gaussian blur on a coarse grid (smooths patch-quantized
/// maps so neighboring patches transition gradually, no fragment seams).
fn blur_grid(g: &Array2<f32>, sigma_cells: f32) -> Array2<f32> {
    let (h, w) = (g.shape()[0], g.shape()[1]);
    let r = (sigma_cells * 3.0).ceil() as isize;
    let mut k = Vec::new();
    let mut ks = 0.0f32;
    for i in -r..=r {
        let v = (-0.5 * (i as f32 / sigma_cells).powi(2)).exp();
        k.push(v);
        ks += v;
    }
    for v in k.iter_mut() {
        *v /= ks;
    }
    let mut tmp = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let xx = (x as isize + ki as isize - r).clamp(0, w as isize - 1) as usize;
                acc += g[[y, xx]] * kv;
            }
            tmp[[y, x]] = acc;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let yy = (y as isize + ki as isize - r).clamp(0, h as isize - 1) as usize;
                acc += tmp[[yy, x]] * kv;
            }
            out[[y, x]] = acc;
        }
    }
    out
}

/// Quantile-normalized local-contrast map upscaled to full frame
/// (bilinear over the patch grid: nearest leaves block seams between
/// patches with different pre/net texture).
pub fn edge01_map(gray: &Array2<f32>, cfg: &PrefilterCfg) -> Array2<f32> {
    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let (p, s) = (cfg.blend_edge_patch, cfg.blend_edge_stride);
    let ys = grid_1d(h, p, s);
    let xs = grid_1d(w, p, s);
    let mut cmap = Array2::<f32>::zeros((ys.len(), xs.len()));
    for (j, &y) in ys.iter().enumerate() {
        for (i, &x) in xs.iter().enumerate() {
            let y1 = (y + p).min(h);
            let x1 = (x + p).min(w);
            let mut acc = 0.0f32;
            let mut n = 0usize;
            for yy in (y..y1).step_by(4) {
                for xx in (x..x1).step_by(4) {
                    let xm = xx.saturating_sub(1);
                    let xp = (xx + 1).min(w - 1);
                    let ym = yy.saturating_sub(1);
                    let yp = (yy + 1).min(h - 1);
                    let dx = gray[[yy, xp]] - gray[[yy, xm]];
                    let dy = gray[[yp, xx]] - gray[[ym, xx]];
                    acc += (dx * dx + dy * dy).sqrt();
                    n += 1;
                }
            }
            cmap[[j, i]] = acc / n.max(1) as f32;
        }
    }
    let flat: Vec<f32> = cmap.iter().cloned().collect();
    let (lo, hi) = (
        quantile(flat.clone(), cfg.blend_edge_lo_q),
        quantile(flat, cfg.blend_edge_hi_q),
    );
    let c01: Array2<f32> =
        cmap.mapv(|v| ((v - lo) / (hi - lo + 1e-9)).clamp(0.0, 1.0));
    let smooth = blur_grid(&c01, 1.0);
    // bilinear upscale to full frame; row/col index maps built once O(h+w)
    let ny = ys.len();
    let nx = xs.len();
    let mut y_idx = vec![0usize; h];
    let mut y_tx = vec![0.0f32; h];
    for y in 0..h {
        let fy = y as f32 / h.max(2) as f32 * (ny - 1) as f32;
        let jj = (fy.floor() as usize).min(ny.saturating_sub(2));
        y_idx[y] = jj;
        y_tx[y] = (fy - jj as f32).clamp(0.0, 1.0);
    }
    let mut x_idx = vec![0usize; w];
    let mut x_tx = vec![0.0f32; w];
    for x in 0..w {
        let fx = x as f32 / w.max(2) as f32 * (nx - 1) as f32;
        let ii = (fx.floor() as usize).min(nx.saturating_sub(2));
        x_idx[x] = ii;
        x_tx[x] = (fx - ii as f32).clamp(0.0, 1.0);
    }
    Array2::from_shape_fn((h, w), |(y, x)| {
        let (j0, i0) = (y_idx[y], x_idx[x]);
        let (j1, i1) = ((j0 + 1).min(ny - 1), (i0 + 1).min(nx - 1));
        let (tx, ty) = (x_tx[x], y_tx[y]);
        smooth[[j0, i0]] * (1.0 - tx) * (1.0 - ty)
            + smooth[[j0, i1]] * tx * (1.0 - ty)
            + smooth[[j1, i0]] * (1.0 - tx) * ty
            + smooth[[j1, i1]] * tx * ty
    })
}

/// Per-pixel net weight from source luma: guards highlights (net crushes
/// them) and deep shadows (net grains them).
pub fn blend_map(src: &Array3<f32>, cfg: &PrefilterCfg) -> Array2<f32> {
    let (h, w) = (src.shape()[1], src.shape()[2]);
    let lum = luma_of(src);
    let edge = edge01_map(&lum, cfg);
    Array2::from_shape_fn((h, w), |(y, x)| {
        let l = lum[[y, x]];
        let hi = cfg.blend_hi_strength * smoothstep(cfg.blend_hi_lo, cfg.blend_hi_hi, l);
        let sh = cfg.blend_sh_strength * (1.0 - smoothstep(cfg.blend_sh_hi, cfg.blend_sh_lo, l));
        let e = cfg.blend_edge_guard * edge[[y, x]];
        ((1.0 - hi) * (1.0 - sh) * (1.0 - e)).clamp(0.0, 1.0)
    })
}

/// Block-DCT input features, exact mirror of `ml/train/dctfeat.py`:
/// orthonormal DCT-II 8x8, screen band 0.08<=r<0.28 (no DC), hi band r>=0.28,
/// log1p energies, nearest replicate. Output (2,H,W): [E_screen, E_hi].
pub fn dct_channels(luma: &Array2<f32>) -> Array3<f32> {
    const B: usize = 8;
    // cosine matrix C[k][n] = a(k)*cos(pi*(2n+1)*k/16)
    let mut c = [[0.0f32; B]; B];
    for k in 0..B {
        let a = if k == 0 {
            (1.0f32 / B as f32).sqrt()
        } else {
            (2.0f32 / B as f32).sqrt()
        };
        for n in 0..B {
            c[k][n] = a * (std::f32::consts::PI * (2 * n + 1) as f32 * k as f32 / 16.0).cos();
        }
    }
    let (h, w) = (luma.shape()[0], luma.shape()[1]);
    let (hp, wp) = ((h + B - 1) / B * B, (w + B - 1) / B * B);
    let at = |yy: usize, xx: usize| luma[[yy.min(h - 1), xx.min(w - 1)]];
    let mut es = Array2::<f32>::zeros((hp, wp));
    let mut eh = Array2::<f32>::zeros((hp, wp));
    let mut blk = [[0.0f32; B]; B];
    let mut tmp = [[0.0f32; B]; B];
    let mut dct = [[0.0f32; B]; B];
    for by in (0..hp).step_by(B) {
        for bx in (0..wp).step_by(B) {
            for y in 0..B {
                for x in 0..B {
                    blk[y][x] = at(by + y, bx + x);
                }
            }
            // rows: tmp = C * blk
            for y in 0..B {
                for k in 0..B {
                    let mut s = 0.0f32;
                    for x in 0..B {
                        s += c[k][x] * blk[y][x];
                    }
                    tmp[y][k] = s;
                }
            }
            // cols: dct = tmp * C^T ; accumulate band energies
            let mut se = 0.0f32;
            let mut sh = 0.0f32;
            for v in 0..B {
                for u in 0..B {
                    let mut s = 0.0f32;
                    for y in 0..B {
                        s += tmp[y][u] * c[v][y];
                    }
                    dct[v][u] = s;
                    if u == 0 && v == 0 {
                        continue;
                    }
                    let r = ((u * u + v * v) as f32).sqrt() / B as f32;
                    if r >= 0.08 && r < 0.28 {
                        se += s * s;
                    } else if r >= 0.28 {
                        sh += s * s;
                    }
                }
            }
            let (le, lh) = ((1.0 + se).ln(), (1.0 + sh).ln());
            for y in 0..B {
                for x in 0..B {
                    let (yy, xx) = (by + y, bx + x);
                    if yy < h && xx < w {
                        es[[yy, xx]] = le;
                        eh[[yy, xx]] = lh;
                    }
                }
            }
        }
    }
    let mut out = Array3::<f32>::zeros((2, h, w));
    for y in 0..h {
        for x in 0..w {
            out[[0, y, x]] = es[[y, x]];
            out[[1, y, x]] = eh[[y, x]];
        }
    }
    out
}

/// Порог гейта LF: пиковость осевой полосы должна упасть хотя бы на 15%,
/// иначе вычитание считается срезанием контента и откатывается.
pub const LF_PEAK_DROP: f32 = 0.85;

/// Пиковость осевой НЧ-полосы: max |F|² на осях / медиана вне осей (0.006–0.05).
///
/// Настоящая полоса сканера/бумаги даёт резкий осевой пик; контент даёт
/// широкий спектр без выделенного пика. Нужна как гейт для LF-стадии: если
/// после вычитания пиковость не упала, значит стадия срезала контент.
pub fn axis_peakiness(gray: &Array2<f32>) -> f32 {    let (h, w) = (gray.shape()[0], gray.shape()[1]);
    let n = h.min(w).min(1024);
    let y0 = h / 2 - n / 2;
    let x0 = w / 2 - n / 2;
    let mut v = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            v[y * n + x] = gray[[y0 + y, x0 + x]];
        }
    }
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    for val in v.iter_mut() {
        *val -= mean;
    }
    let raw = fft2_forward(&v, n);
    let mut on_axis: Vec<f32> = Vec::new();
    let mut off_axis: Vec<f32> = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let (fx, fy, r) = freq_at_raw(x, y, n);
            if !(0.006..0.05).contains(&r) {
                continue;
            }
            let e = raw[y * n + x].norm_sqr();
            if fx.abs() < 0.004 || fy.abs() < 0.004 {
                on_axis.push(e);
            } else {
                off_axis.push(e);
            }
        }
    }
    if on_axis.is_empty() || off_axis.is_empty() {
        return 0.0;
    }
    off_axis.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = off_axis[off_axis.len() / 2];
    let max = on_axis.iter().cloned().fold(0.0f32, f32::max);
    max / (med + 1e-30)
}

pub fn luma_of(img: &Array3<f32>) -> Array2<f32> {
    let (h, w) = (img.shape()[1], img.shape()[2]);
    Array2::from_shape_fn((h, w), |(y, x)| {
        0.299 * img[[0, y, x]] + 0.587 * img[[1, y, x]] + 0.114 * img[[2, y, x]]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_peakiness_separates_band_from_content() {
        // осевая полоса (дефект протяжки сканера), период 128 px, против шума
        let (h, w) = (256, 256);
        let mut band = Array2::<f32>::zeros((h, w));
        let mut noise = Array2::<f32>::zeros((h, w));
        let mut seed = 7u64;
        for y in 0..h {
            for x in 0..w {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let n = (seed & 0xFFFF) as f32 / 65535.0 - 0.5;
                band[[y, x]] =
                    0.5 + 0.05 * (2.0 * std::f32::consts::PI * x as f32 / 128.0).cos();
                noise[[y, x]] = 0.5 + 0.05 * n;
            }
        }
        let pk_band = axis_peakiness(&band);
        let pk_noise = axis_peakiness(&noise);
        eprintln!("axis_peakiness: band={pk_band:.0} noise={pk_noise:.1}");
        assert!(pk_band > 1e6, "periodic axis band must stand out: {pk_band}");
        assert!(pk_noise < 50.0, "content must not look periodic: {pk_noise}");
    }

    #[test]
    fn fft_roundtrip() {
        let n = 64usize;
        let v: Vec<f32> = (0..n * n).map(|k| ((k % 17) as f32) / 17.0).collect();
        let mut buf = fft2_forward(&v, n);
        fft2_inverse(&mut buf, n);
        for (a, b) in v.iter().zip(buf.iter()) {
            assert!((a - b.re).abs() < 1e-3, "roundtrip diverged");
            assert!(b.im.abs() < 1e-3);
        }
    }

    #[test]
    fn estimate_raster_finds_grating() {
        let n = 512usize;
        let g = Array2::from_shape_fn((n, n), |(_, x)| {
            0.5 + 0.1 * (2.0 * PI * 0.15 * x as f32).cos()
        });
        let cfg = PrefilterCfg::default();
        let est = estimate_raster(&g, &cfg, 1024); // roi клэмпится к 512
        assert!((est.r - 0.15).abs() < 0.01, "r={}", est.r);
        let ang = est.angle_deg.rem_euclid(180.0);
        assert!(ang < 10.0 || ang > 170.0, "ang={}", est.angle_deg);
        assert!(est.confidence > 0.5, "conf={}", est.confidence);
    }

    #[test]
    fn estimate_raster_multi_ignores_blank_centre() {
        // 2.43: растр есть только в углу, центр — плашка. Центральный ROI
        // несущую не видит, сетка окон обязана найти.
        let n = 2048usize;
        let s = 256usize;
        let g = Array2::from_shape_fn((n, n), |(y, x)| {
            if y < s && x < s {
                0.5 + 0.1 * (2.0 * PI * 0.15 * x as f32).cos()
            } else {
                0.5
            }
        });
        let cfg = PrefilterCfg::default();
        let center = estimate_raster(&g, &cfg, s);
        let multi = estimate_raster_multi(&g, &cfg, s, 3);
        assert!(
            (center.r - 0.15).abs() > 0.01,
            "центральный ROI не должен видеть растр, r={}",
            center.r
        );
        assert!((multi.r - 0.15).abs() < 0.01, "multi r={}", multi.r);
        assert!(multi.confidence > center.confidence, "conf {} vs {}", multi.confidence, center.confidence);
    }

    #[test]
    fn estimate_raster_multi_discounts_single_roi() {
        // Один и тот же растр во всех окнах: подтверждение соседями даёт
        // уверенность не ниже, чем у одиночного замера, и не более 1.
        let n = 1024usize;
        let g = Array2::from_shape_fn((n, n), |(_, x)| {
            0.5 + 0.1 * (2.0 * PI * 0.15 * x as f32).cos()
        });
        let cfg = PrefilterCfg::default();
        let one = estimate_raster(&g, &cfg, 1024);
        let multi = estimate_raster_multi(&g, &cfg, 1024, 3);
        assert!((multi.r - one.r).abs() < 0.01, "r {} vs {}", multi.r, one.r);
        assert!(multi.confidence >= one.confidence * 0.99);
        assert!(multi.confidence <= 1.0);
    }

    #[test]
    fn estimate_raster_multi_prefers_fundamental_over_harmonic() {
        // 2.43: гармоника (r√2) сильнее фундаментала в отдельном окне — сетка
        // обязана вернуть фундаментал, иначе V1 демодулирует не ту частоту.
        let s = 256usize;
        let n = 2048usize;
        let g = Array2::from_shape_fn((n, n), |(y, x)| {
            let xf = (x % s) as f32;
            let fund = (2.0 * PI * 0.15 * xf).cos();
            let harm = (2.0 * PI * 0.15 * 1.41421356 * xf).cos();
            // фундаментал — по всему кадру, гармоника — только в одном окне
            let in_top_right = y < s && x >= n - s;
            if in_top_right {
                0.5 + 0.04 * fund + 0.12 * harm
            } else {
                0.5 + 0.06 * fund
            }
        });
        let cfg = PrefilterCfg::default();
        let est = estimate_raster_multi(&g, &cfg, s, 3);
        assert!(
            (est.r - 0.15).abs() < 0.012,
            "ожидался фундаментал 0.15, получено {}",
            est.r
        );
    }

    #[test]
    fn estimate_raster_multi_ignores_lone_low_freq_peak() {
        // 2.45: на Magazine (2) правило «минимальный радиус» выбрало ложный
        // НЧ-пик 0.0735 (период 13.6 px), который видела ОДНА зона из девяти,
        // и V1 не сработал (coh 0.09). Подтверждённая несущая 0.15 в двух
        // окнах должна выигрывать у сильного, но одинокого НЧ-пика.
        let s = 256usize;
        let n = 2048usize;
        let g = Array2::from_shape_fn((n, n), |(y, x)| {
            let xf = (x % s) as f32;
            let carrier = 0.5 + 0.05 * (2.0 * PI * 0.15 * xf).cos();
            let low = 0.5 + 0.10 * (2.0 * PI * 0.0735 * xf).cos();
            let in_left = x < s;
            if in_left && (y < s || (y >= n - s && y < n)) {
                carrier
            } else if y < s && x >= n - s {
                low
            } else {
                0.5
            }
        });
        let cfg = PrefilterCfg::default();
        let est = estimate_raster_multi(&g, &cfg, s, 3);
        assert!(
            (est.r - 0.15).abs() < 0.012,
            "ожидалась подтверждённая несущая 0.15, получено {}",
            est.r
        );
    }

    #[test]
    fn notch_kills_grating_preserves_dc() {
        // synthetic AM-ish grating + gradient content
        let n = 256usize;
        let mut g = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let content = 0.35 + 0.3 * (x as f32 / n as f32);
                let grating = 0.08 * (2.0 * PI * 0.15 * x as f32).cos();
                g[y * n + x] = content + grating;
            }
        }
        let mean0: f32 = g.iter().sum::<f32>() / g.len() as f32;
        let raw = fft2_forward(&g, n);
        let lm = logmag_shifted(&raw, n);
        let bg = box_blur(&lm, n, 7);
        let diff: Vec<f32> = lm.iter().zip(bg.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
        let cfg = PrefilterCfg::default();
        let (peaks, _) = detect_peaks(&diff, n, &cfg);
        assert!(!peaks.is_empty(), "grating peak must be detected");
        let r0 = peaks[0].r;
        assert!((r0 - 0.15).abs() < 0.01, "detected r={r0}, want ~0.15");
        let mask = notch_mask(n, &[(r0, peaks[0].angle_deg)], 6.0 / 1024.0, 3, true);
        let mut patch = g.clone();
        apply_mask_patch(&mut patch, n, &mask);
        let mean1: f32 = patch.iter().sum::<f32>() / patch.len() as f32;
        assert!((mean0 - mean1).abs() < 1e-3, "DC must be preserved");
        // grating-band energy must collapse (band metric, not total energy:
        // gaussian notch is soft by design and spares content)
        let band = |v: &[f32]| {
            let raw = fft2_forward(v, n);
            let mut be = 0.0f32;
            let mut tot = 0.0f32;
            for k in 0..n * n {
                let e = raw[k].norm_sqr();
                tot += e;
                let (x, y) = (k % n, k / n);
                let (_, _, r) = freq_at_raw(x, y, n);
                if r >= 0.12 && r < 0.18 {
                    be += e;
                }
            }
            be / (tot + 1e-9)
        };
        let (e0, e1) = (band(&g), band(&patch));
        assert!(e1 < e0 * 0.2, "notch must kill the grating band: {e0} -> {e1}");
    }

    #[test]
    fn top4_flags_grating_and_notch_drops_it() {
        let n = 256usize;
        let mut g = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let content = 0.35 + 0.3 * (x as f32 / n as f32);
                let grating = 0.08 * (2.0 * PI * 0.15 * x as f32).cos();
                g[y * n + x] = content + grating;
            }
        }
        let arr = Array2::from_shape_vec((n, n), g.clone()).unwrap();
        let t4 = top4_peak_sum(&arr);
        // grating amplitude 0.08 on a flat ramp: dominant pair + mirror top4
        // must stand far above the noise floor
        assert!(t4 > 20.0, "grating top4 must be large, got {t4}");
        // notch filter at the detected frequency collapses the peaks
        let mean0: f32 = g.iter().sum::<f32>() / g.len() as f32;
        let raw = fft2_forward(&g, n);
        let lm = logmag_shifted(&raw, n);
        let bg = box_blur(&lm, n, 7);
        let diff: Vec<f32> = lm.iter().zip(bg.iter()).map(|(a, b)| (a - b).max(0.0)).collect();
        let cfg = PrefilterCfg::default();
        let (peaks, _) = detect_peaks(&diff, n, &cfg);
        assert!(!peaks.is_empty());
        let mask = notch_mask(n, &[(peaks[0].r, peaks[0].angle_deg)], 16.0 / 1024.0, 3, true);
        let mut patch = g.clone();
        apply_mask_patch(&mut patch, n, &mask);
        let mean1: f32 = patch.iter().sum::<f32>() / patch.len() as f32;
        assert!((mean0 - mean1).abs() < 1e-3);
        let arr2 = Array2::from_shape_vec((n, n), patch).unwrap();
        let t4b = top4_peak_sum(&arr2);
        // gaussian notch is soft by design; top4 must still lose the grating
        // dominance (observed 3975 -> ~1360 with a production-width notch)
        assert!(t4b < t4 * 0.5, "notch must drop top4: {t4} -> {t4b}");
    }

    #[test]
    fn bgswap_and_tone_keep_stats() {
        let (c, h, w) = (3usize, 64usize, 64usize);
        let src = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            (0.2 + 0.5 * (x as f32 / w as f32) + 0.05 * ch as f32).clamp(0.0, 1.0)
        });
        let net = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            (0.3 + 0.5 * (x as f32 / w as f32) + 0.05 * ch as f32).clamp(0.0, 1.0)
        });
        let out = return_background(&src, &net, 8.0);
        for ch in 0..c {
            let (mut ms, mut mo) = (0.0f64, 0.0);
            for y in 0..h {
                for x in 0..w {
                    ms += src[[ch, y, x]] as f64;
                    mo += out[[ch, y, x]] as f64;
                }
            }
            ms /= (h * w) as f64;
            mo /= (h * w) as f64;
            assert!((ms - mo).abs() < 0.01, "bgswap must preserve means");
        }
        let t = linear_tone(&net, &src);
        for ch in 0..c {
            let (mut ms, mut mo) = (0.0f64, 0.0);
            for y in 0..h {
                for x in 0..w {
                    ms += src[[ch, y, x]] as f64;
                    mo += t[[ch, y, x]] as f64;
                }
            }
            assert!(((ms - mo) / (h * w) as f64).abs() < 1e-6);
        }
    }

    #[test]
    fn linear_tone_does_not_inflate_texture() {
        // 2.52: растр в СКО скана — не тон. Раньше a = σ_scan/σ_out возвращал
        // разброс растра умножением всего кадра: на `1821146u` это +4.3 % к
        // бумажной фактуре и полосам 16–128 px. Теперь масштаб берётся по НЧ,
        // поэтому тон выравнивается, а фактура не растёт.
        let (c, h, w) = (3usize, 256usize, 256usize);
        let lf = |ch: usize, x: usize, y: usize| {
            (0.55 + 0.10 * x as f32 / w as f32 + 0.06 * y as f32 / h as f32 + 0.02 * ch as f32)
                .clamp(0.0, 1.0)
        };
        let src = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            let r = 0.03
                * (std::f32::consts::PI * x as f32 / 3.0).sin()
                * (std::f32::consts::PI * y as f32 / 3.0).cos();
            (lf(ch, x, y) + r).clamp(0.0, 1.0)
        });
        let net = Array3::from_shape_fn((c, h, w), |(ch, y, x)| lf(ch, x, y));
        let out = linear_tone(&net, &src);
        let win = 33usize;
        for ch in 0..c {
            let n = (h * w) as f64;
            let mut v = |a: &Array3<f32>| -> f64 {
                let buf: Vec<f32> = (0..h * w).map(|k| a[[ch, k / w, k % w]]).collect();
                let lo = box_blur_rect(&buf, h, w, win);
                let m: f64 = lo.iter().map(|&x| x as f64).sum::<f64>() / n;
                (lo.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / n).sqrt()
            };
            let (s0, s1) = (v(&net), v(&out));
            let keep = s1 / (s0 + 1e-12);
            assert!(
                (keep - 1.0).abs() < 0.02,
                "tone must not inflate texture: {s0} -> {s1} ({keep})"
            );
            // Среднее по-прежнему приходит из скана.
            let ms: f64 = (0..h * w).map(|k| src[[ch, k / w, k % w]] as f64).sum::<f64>() / n;
            let mo: f64 = (0..h * w).map(|k| out[[ch, k / w, k % w]] as f64).sum::<f64>() / n;
            assert!((ms - mo).abs() < 1e-5, "mean must come from the scan");
        }
    }


    #[test]
    fn gaussian_lowpass_does_not_wrap_edges() {
        // 2.49: НЧ-фильтр считался циклической свёрткой по всему кадру, из-за
        // чего на левом краю подмешивался правый (полосы по краям кадра).
        let (h, w) = (32usize, 256usize);
        let mut ch = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                ch[y * w + x] = if x < w / 2 { 0.2 } else { 0.8 };
            }
        }
        let lp = gaussian_lowpass_channel(&ch, h, w, 16.0);
        let mid = h / 2;
        let left = lp[mid * w];
        let right = lp[mid * w + w - 1];
        assert!(
            (left - 0.2).abs() < 0.01,
            "left edge must keep its own tone, got {left}"
        );
        assert!(
            (right - 0.8).abs() < 0.01,
            "right edge must keep its own tone, got {right}"
        );
    }

    #[test]
    fn local_tone_guard_is_identity_on_matching_frames() {
        let (c, h, w) = (3usize, 64usize, 96usize);
        let src = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            let base = if (x / 16 + y / 16) % 2 == 0 { 0.85 } else { 0.35 };
            (base + 0.02 * ch as f32 + 0.01 * ((x + y) % 3) as f32).clamp(0.0, 1.0)
        });
        let out = local_tone_guard(&src, &src, 16.0, 8.0);
        let mut worst = 0.0f32;
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    worst = worst.max((out[[ch, y, x]] - src[[ch, y, x]]).abs());
                }
            }
        }
        assert!(worst < 1e-3, "guard must be identity when net == src, worst {worst}");
    }

    #[test]
    fn local_tone_guard_restores_flat_tone() {
        // Плоская бумага: сеть уводит уровень и добавляет пятна 40–70 px —
        // страховка обязана вернуть средний уровень к скану, сохранив ВЧ.
        let (c, h, w) = (3usize, 96usize, 96usize);
        let src = Array3::from_shape_fn((c, h, w), |(ch, _y, _x)| 0.80 - 0.02 * ch as f32);
        let net = Array3::<f32>::from_shape_fn((c, h, w), |(ch, y, x)| {
            let blob = 0.03 * ((x as f32 / 9.0).sin() * (y as f32 / 11.0).cos());
            (src[[ch, y, x]] - 0.06 + blob).clamp(0.0, 1.0)
        });
        let out = local_tone_guard(&net, &src, 16.0, 8.0);
        for ch in 0..c {
            let mut mean = 0.0f64;
            for y in 0..h {
                for x in 0..w {
                    mean += out[[ch, y, x]] as f64;
                }
            }
            mean /= (h * w) as f64;
            let want = src[[ch, 0, 0]] as f64;
            assert!(
                (mean - want).abs() < 0.01,
                "flat tone must return to the scan: {mean} vs {want}"
            );
        }
    }

    #[test]
    fn local_tone_guard_does_not_reinject_carrier() {
        // 2.49: гаусс σ=3 пропускал 35 % суперячейки (13 px при r≈0.149) и
        // 2 % несущей — на белом оставалась регулярная решётка. Срез 16 px
        // обязан держать и несущую, и суперячейку ниже 0.1 уровня (0.4/255).
        let (c, h, w) = (3usize, 128usize, 128usize);
        let (r, r2) = (0.1487f32, 0.1487 / 2.0);
        let src = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            let a = 0.04 * (2.0 * std::f32::consts::PI * r * x as f32).sin();
            let b = 0.04 * (2.0 * std::f32::consts::PI * r2 * (x + y) as f32).cos();
            (0.80 + a + b + 0.01 * ch as f32).clamp(0.0, 1.0)
        });
        let net = Array3::<f32>::from_shape_fn((c, h, w), |(ch, _y, _x)| 0.80 + 0.01 * ch as f32);
        let out = local_tone_guard(&net, &src, 16.0, 8.0);
        // Края кадра не проверяем: replicate-паддинг обрывает синус ступенькой,
        // и её низкочастотный хвост живёт в первых ~40 px (у скана граница
        // непрерывна, там такого скачка нет).
        let mut worst = 0.0f32;
        for ch in 0..c {
            for y in 40..h - 40 {
                for x in 40..w - 40 {
                    worst = worst.max((out[[ch, y, x]] - net[[ch, y, x]]).abs());
                }
            }
        }
        assert!(
            worst < 0.4 / 255.0,
            "carrier and supercell must not come back, worst {worst}"
        );
    }

    #[test]
    fn local_tone_guard_drops_lf_chroma_of_the_print() {
        // 2.53: цветная «розетка» печати с периодом 8–32 px не должна
        // возвращаться из скана. Скан: ровный тон + чистая цветность (u-поле,
        // период 24 px, амплитуда 0.04 = 10 уровней). Сеть уже сняла и растр,
        // и его цветной след — её выход ровный. Страховка обязана вернуть тон,
        // но не цветность ниже окна `t_chroma` = 4·t_cut = 64 px.
        let (c, h, w) = (3usize, 160usize, 160usize);
        let chroma = |x: usize, y: usize| {
            0.04 * (2.0 * std::f32::consts::PI * x as f32 / 24.0).sin()
                * (2.0 * std::f32::consts::PI * y as f32 / 24.0).cos()
        };
        // Тон одинаков во всех каналах: поправка яркости должна уйти в ноль.
        let src = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            let t = chroma(x, y);
            let base = 0.76;
            let v = match ch {
                0 => base + 0.5 * t,
                1 => base - 0.5 * t,
                _ => base,
            };
            v.clamp(0.0, 1.0)
        });
        // Сеть сняла растр: ровное поле, средний уровень тот же (иначе
        // постоянная разность каналов даёт цветовой сдвиг, и тест измерял бы
        // его, а не утечку полосы 24 px).
        let net = Array3::<f32>::from_shape_fn((c, h, w), |(_ch, _y, _x)| 0.76);
        let out = local_tone_guard(&net, &src, 16.0, 8.0);
        let (mut e_in, mut e_out) = (0.0f64, 0.0f64);
        let mut n = 0.0f64;
        for y in 40..h - 40 {
            for x in 40..w - 40 {
                let cin = 0.5 * (src[[0, y, x]] - src[[1, y, x]]);
                let cout = 0.5 * (out[[0, y, x]] - out[[1, y, x]]) - 0.5 * (net[[0, y, x]] - net[[1, y, x]]);
                e_in += (cin as f64).powi(2);
                e_out += (cout as f64).powi(2);
                n += 1.0;
            }
        }
        let (rin, rout) = ((e_in / n).sqrt(), (e_out / n).sqrt());
        assert!(
            rout < 0.15 * rin,
            "LF chroma must stay out: in {rin:.4} out {rout:.4}"
        );
        // И при этом тон всё равно вернулся к скану.
        let mut m = 0.0f64;
        for ch in 0..c {
            for y in 40..h - 40 {
                for x in 40..w - 40 {
                    m += out[[ch, y, x]] as f64;
                }
            }
        }
        m /= (c * (h - 80) * (w - 80)) as f64;
        assert!((m - 0.76).abs() < 0.005, "tone must return to the scan, got {m}");
    }

    #[test]
    fn grid_scrub_kills_checkerboard() {
        // 2.51: на постоянном поле сеть рисует checkerboard с периодом 4 px —
        // пики 4.00 px по осям, 2.83 px по диагонали, 2.00 px. Разложение по
        // 16 фазам обязано снять его, не тронув наклон (тон).
        let (c, h, w) = (3usize, 128usize, 128usize);
        let ramp = |ch: usize, x: usize, y: usize| {
            0.45 + 0.02 * ch as f32 + 0.20 * x as f32 / w as f32 + 0.10 * y as f32 / h as f32
        };
        let pat = |x: usize, y: usize| {
            let px = std::f32::consts::PI * x as f32 / 2.0;
            let py = std::f32::consts::PI * y as f32 / 2.0;
            0.004 * px.cos() * py.cos() + 0.003 * (px + py).cos()
        };
        let img = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            (ramp(ch, x, y) + pat(x, y)).clamp(0.0, 1.0)
        });
        let out = grid_scrub(&img, 32);
        let (mut num, mut den, mut worst) = (0.0f64, 0.0f64, 0.0f32);
        for ch in 0..c {
            for y in 16..h - 16 {
                for x in 16..w - 16 {
                    let d = out[[ch, y, x]] - ramp(ch, x, y);
                    num += (d as f64).powi(2);
                    den += (pat(x, y) as f64).powi(2);
                    worst = worst.max((out[[ch, y, x]] - ramp(ch, x, y)).abs());
                }
            }
        }
        let keep = (num / den).sqrt();
        assert!(keep < 0.1, "checkerboard must be removed, kept {keep}");
        assert!(
            worst < 0.4 / 255.0,
            "no residual grid above 0.4 level, worst {worst}"
        );
    }

    #[test]
    fn grid_scrub_keeps_mean_and_detail() {
        // Гладкая фактура (не периодичная с шагом 4 px) должна выжить: scrub
        // убирает только фазово-когерентную часть.
        let (c, h, w) = (3usize, 192usize, 192usize);
        let img = Array3::from_shape_fn((c, h, w), |(ch, y, x)| {
            let blob = 0.05 * ((x as f32 / 23.0).sin() * (y as f32 / 17.0).cos());
            (0.60 + 0.01 * ch as f32 + blob).clamp(0.0, 1.0)
        });
        let out = grid_scrub(&img, 32);
        for ch in 0..c {
            let (mut m0, mut m1, mut v0, mut v1) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            // Поля 64 px от кромки не берём: у фазовых средних и у общего
            // среднего разная длина clamp-повтора на границе кадра.
            for y in 64..h - 64 {
                for x in 64..w - 64 {
                    m0 += img[[ch, y, x]] as f64;
                    m1 += out[[ch, y, x]] as f64;
                    v0 += (img[[ch, y, x]] as f64).powi(2);
                    v1 += (out[[ch, y, x]] as f64).powi(2);
                }
            }
            let n = ((h - 128) * (w - 128)) as f64;
            // Остаточный сдвиг — сотые доли уровня (окна фазовых средних и
            // общего среднего совпадают лишь приближённо, 4·pw-1 против 4·pw).
            assert!((m0 - m1).abs() / n < 2e-4, "mean must stay: {m0} vs {m1}");
            let (s0, s1) = ((v0 / n).sqrt(), (v1 / n).sqrt());
            assert!(
                (s1 - s0).abs() < 0.02 * s0,
                "smooth texture must survive: {s0} vs {s1}"
            );
        }
    }

    #[test]
    fn dct_channels_match_naive() {        // naive f64 DCT-II reference (direct cosine sums, independent path)
        fn naive(luma: &Array2<f32>) -> Array3<f32> {
            const B: usize = 8;
            let (h, w) = (luma.shape()[0], luma.shape()[1]);
            let cos = |k: usize, n: usize| {
                (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos()
            };
            let alpha = |k: usize| {
                if k == 0 {
                    (1.0f64 / B as f64).sqrt()
                } else {
                    (2.0f64 / B as f64).sqrt()
                }
            };
            let mut out = Array3::<f64>::zeros((2, h, w));
            for by in (0..h).step_by(B) {
                for bx in (0..w).step_by(B) {
                    let mut se = 0.0f64;
                    let mut sh = 0.0f64;
                    for v in 0..B {
                        for u in 0..B {
                            if u == 0 && v == 0 {
                                continue;
                            }
                            let mut s = 0.0f64;
                            for y in 0..B {
                                for x in 0..B {
                                    let yy = (by + y).min(h - 1);
                                    let xx = (bx + x).min(w - 1);
                                    s += luma[[yy, xx]] as f64
                                        * alpha(u)
                                        * alpha(v)
                                        * cos(u, x)
                                        * cos(v, y);
                                }
                            }
                            let r = ((u * u + v * v) as f64).sqrt() / B as f64;
                            if r >= 0.08 && r < 0.28 {
                                se += s * s;
                            } else if r >= 0.28 {
                                sh += s * s;
                            }
                        }
                    }
                    for y in 0..B {
                        for x in 0..B {
                            if by + y < h && bx + x < w {
                                out[[0, by + y, bx + x]] = (1.0 + se).ln();
                                out[[1, by + y, bx + x]] = (1.0 + sh).ln();
                            }
                        }
                    }
                }
            }
            out.mapv(|v| v as f32)
        }

        // DC-only input -> zero band energies
        let flat = Array2::<f32>::zeros((16, 16));
        let f0 = dct_channels(&flat);
        assert!(f0.iter().all(|&v| v == 0.0));

        // pseudo-random input vs naive reference
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let rnd = Array2::from_shape_fn((16, 16), |_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed & 0xFFFF) as f32 / 65535.0
        });
        let fast = dct_channels(&rnd);
        let ref_ = naive(&rnd);
        let maxd = fast
            .iter()
            .zip(ref_.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxd < 1e-4, "DCT impl diverges from naive: {maxd}");
    }
}
