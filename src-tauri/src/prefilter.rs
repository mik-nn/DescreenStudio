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
                out[[ch, y, x]] = acc[[ch, y, x]] / weight[[y, x]].max(1e-9);
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
    let mask = notch_mask(h.max(w), &peaks, cfg.lf_notch_r, cfg.lf_notch_order, false);
    // NOTE: square mask sized max(h,w); applied on a square canvas then cropped.
    let n = h.max(w);
    let mut out = img.clone();
    for ch in 0..c {
        let mut sq = vec![0.0f32; n * n];
        for y in 0..h {
            for x in 0..w {
                sq[y * n + x] = img[[ch, y, x]];
            }
        }
        // replicate-pad remainder
        for y in 0..n {
            for x in 0..n {
                if y >= h || x >= w {
                    sq[y * n + x] = img[[ch, y.min(h - 1), x.min(w - 1)]];
                }
            }
        }
        apply_mask_patch(&mut sq, n, &mask);
        for y in 0..h {
            for x in 0..w {
                out[[ch, y, x]] = sq[y * n + x];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Paper background return + linear tone
// ---------------------------------------------------------------------------

fn gaussian_lowpass_channel(ch: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
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

pub fn linear_tone(out: &Array3<f32>, src: &Array3<f32>) -> Array3<f32> {
    let (c, h, w) = (out.shape()[0], out.shape()[1], out.shape()[2]);
    let mut res = Array3::<f32>::zeros((c, h, w));
    for ch in 0..c {
        let (mut mo, mut vo, mut ms, mut vs, mut n) = (0.0f64, 0.0, 0.0f64, 0.0, 0usize);
        for y in 0..h {
            for x in 0..w {
                mo += out[[ch, y, x]] as f64;
                ms += src[[ch, y, x]] as f64;
                n += 1;
            }
        }
        mo /= n as f64;
        ms /= n as f64;
        for y in 0..h {
            for x in 0..w {
                vo += (out[[ch, y, x]] as f64 - mo).powi(2);
                vs += (src[[ch, y, x]] as f64 - ms).powi(2);
            }
        }
        let a = ((vs / n as f64).sqrt() / ((vo / n as f64).sqrt() + 1e-12)) as f32;
        for y in 0..h {
            for x in 0..w {
                res[[ch, y, x]] = ((out[[ch, y, x]] - mo as f32) * a + ms as f32).clamp(0.0, 1.0);
            }
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
