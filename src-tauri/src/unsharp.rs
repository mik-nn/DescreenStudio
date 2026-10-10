//! Adaptive unsharp-mask — порт `ml/preproc/unsharp.py` (2026-10-10).
//!
//! Обратная связь без GT: меряем `sharp_ratio` = дисперсия лапласиана
//! (диагональная 4-точечная схема) на центральном ROI 1024² результата
//! относительно источника; amount итеративно подкручивается шагом ×1.4,
//! потолок AMOUNT_MAX (кап упирался на M(2) run27, поднят с 1.5 до 2.2).
//!
//! Паритет: границы гаусса — отражение без дублирования края (np.pad
//! 'reflect'): расхождение со scipy только в полосе радиуса у края кадра,
//! метрика смотрит на центр — на петлю не влияет.

use ndarray::{Array2, Array3};

pub const SHARP_TARGET: f32 = 1.0;
pub const SHARP_TOL: f32 = 0.05;
pub const AMOUNT_MAX: f32 = 2.2;
pub const AMOUNT_STEP: f32 = 1.4;
pub const SIGMA: f32 = 1.0;
const CENTER_ROI: usize = 1024;
const MAX_ITERS: usize = 8;

/// Центральный квадрат: span = 2*floor(s/2), s = min(CENTER_ROI, h, w) —
/// как metrics.center_roi (span в Python усекает нечётный s на 1px).
fn center_square(g: &Array2<f32>) -> Array2<f32> {
    let (h, w) = (g.dim().0, g.dim().1);
    let s = CENTER_ROI.min(h).min(w);
    let s = s - (s & 1);
    let y0 = h / 2 - s / 2;
    let x0 = w / 2 - s / 2;
    g.slice(ndarray::s![y0..y0 + s, x0..x0 + s]).to_owned()
}

/// Дисперсия диагонального лапласиана (metrics._lapvar): 4 диагональных
/// соседа − 4×центр.
fn lapvar(g: &Array2<f32>) -> f64 {
    let (h, w) = g.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0u64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let v = (g[[y - 1, x - 1]]
                + g[[y - 1, x + 1]]
                + g[[y + 1, x - 1]]
                + g[[y + 1, x + 1]]
                - 4.0 * g[[y, x]]) as f64;
            sum += v;
            sum2 += v * v;
            n += 1;
        }
    }
    let mean = sum / n as f64;
    (sum2 / n as f64 - mean * mean).max(0.0)
}

/// sharp_ratio = lapvar(center(out)) / (lapvar(center(src)) + 1e-12).
pub fn sharp_ratio(src_l: &Array2<f32>, out_l: &Array2<f32>) -> f32 {
    let a = lapvar(&center_square(out_l));
    let b = lapvar(&center_square(src_l));
    (a / (b + 1e-12)) as f32
}

/// Separable gaussian with np.pad-style reflect (no edge duplication):
/// idx<0 -> -idx, idx>=n -> 2n-2-idx.
fn blur_channel(ch: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let (h, w) = ch.dim();
    let radius = (4.0 * sigma + 0.5) as usize;
    let mut kern = Vec::with_capacity(2 * radius + 1);
    let mut ksum = 0.0f32;
    for i in -(radius as isize)..=(radius as isize) {
        let v = (-0.5 * (i as f32 / sigma).powi(2)).exp();
        kern.push(v);
        ksum += v;
    }
    for v in kern.iter_mut() {
        *v /= ksum;
    }
    let map = |i: isize, n: usize| -> usize {
        if i < 0 {
            (-i) as usize
        } else if i >= n as isize {
            (2 * n as isize - 2 - i) as usize
        } else {
            i as usize
        }
    };
    // rows
    let mut tmp = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, kv) in kern.iter().enumerate() {
                let xi = x as isize + k as isize - radius as isize;
                acc += kv * ch[[y, map(xi, w)]];
            }
            tmp[[y, x]] = acc;
        }
    }
    // cols
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, kv) in kern.iter().enumerate() {
                let yi = y as isize + k as isize - radius as isize;
                acc += kv * tmp[[map(yi, h), x]];
            }
            out[[y, x]] = acc;
        }
    }
    out
}

/// img + amount*(img - gauss(img)), clip [0,1].
pub fn unsharp(img: &Array3<f32>, amount: f32, sigma: f32) -> Array3<f32> {
    if amount <= 0.0 {
        return img.clone();
    }
    let (_, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let mut out = Array3::<f32>::zeros((3, h, w));
    for c in 0..3 {
        let ch = img.slice(ndarray::s![c, .., ..]).to_owned();
        let blur = blur_channel(&ch, sigma);
        for y in 0..h {
            for x in 0..w {
                out[[c, y, x]] = (img[[c, y, x]] + amount * (img[[c, y, x]] - blur[[y, x]]))
                    .clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// Итеративный подбор amount к цели SHARP_TARGET (±SHARP_TOL), максимум
/// MAX_ITERS замеров, шаг AMOUNT_STEP, потолок AMOUNT_MAX. Возвращает
/// (out, amount, sharp_ratio).
pub fn adaptive_unsharp(
    img: &Array3<f32>,
    src: &Array3<f32>,
) -> (Array3<f32>, f32, f32) {
    let src_l = crate::prefilter::luma_of(src);
    let out_l = |a: &Array3<f32>| crate::prefilter::luma_of(a);
    let r0 = sharp_ratio(&src_l, &out_l(img));
    if r0 >= SHARP_TARGET - SHARP_TOL {
        return (img.clone(), 0.0, r0);
    }
    let mut amount = 0.5f32;
    let mut best = img.clone();
    let mut best_d = (r0 - SHARP_TARGET).abs();
    for _ in 0..MAX_ITERS {
        let cand = unsharp(img, amount, SIGMA);
        let r = sharp_ratio(&src_l, &out_l(&cand));
        let d = (r - SHARP_TARGET).abs();
        if d < best_d {
            best = cand.clone();
            best_d = d;
        }
        if d <= SHARP_TOL {
            return (cand, amount, r);
        }
        if r < SHARP_TARGET {
            amount = (AMOUNT_MAX).min(amount * AMOUNT_STEP);
        } else {
            amount /= AMOUNT_STEP;
        }
        if amount >= AMOUNT_MAX && r < SHARP_TARGET {
            break;
        }
    }
    let r = sharp_ratio(&src_l, &out_l(&best));
    (best, amount, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy_gradient(h: usize, w: usize, seed: &mut u64) -> Array3<f32> {
        Array3::from_shape_fn((3, h, w), |(_, y, x)| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let n = ((*seed & 0xFFFF) as f32) / 65535.0;
            let g = 0.3 + 0.4 * (x as f32 / w as f32 + y as f32 / h as f32);
            (g + 0.05 * (n - 0.5)).clamp(0.0, 1.0)
        })
    }

    #[test]
    fn sharp_ratio_of_identity_is_one() {
        let img = noisy_gradient(160, 160, &mut 42);
        let l = crate::prefilter::luma_of(&img);
        let r = sharp_ratio(&l, &l);
        assert!((r - 1.0).abs() < 1e-5, "identity ratio {r}");
    }

    #[test]
    fn adaptive_skips_already_sharp() {
        let img = noisy_gradient(160, 160, &mut 7);
        let (out, amount, r) = adaptive_unsharp(&img, &img);
        assert_eq!(amount, 0.0, "must not sharpen an identity");
        assert!((r - 1.0).abs() < 1e-5);
        assert_eq!(out.shape(), img.shape());
    }

    #[test]
    fn adaptive_recovers_blurred_detail() {
        let src = noisy_gradient(320, 320, &mut 11);
        // soft = 3x3 box blur (unsharp() сам не размывает: amount<=0 = no-op)
        let (_, h, w) = (src.shape()[0], src.shape()[1], src.shape()[2]);
        let mut soft = src.clone();
        for c in 0..3 {
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let mut acc = 0.0;
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            acc += src[[c, (y as isize + dy) as usize, (x as isize + dx) as usize]];
                        }
                    }
                    soft[[c, y, x]] = acc / 9.0;
                }
            }
        }
        let r_before = {
            let sl = crate::prefilter::luma_of(&src);
            let bl = crate::prefilter::luma_of(&soft);
            sharp_ratio(&sl, &bl)
        };
        assert!(r_before < SHARP_TARGET - SHARP_TOL, "must start soft: {r_before}");
        let (out, amount, r_after) = adaptive_unsharp(&soft, &src);
        assert!(amount > 0.0, "must pick a positive amount");
        assert!(
            (r_after - SHARP_TARGET).abs() < (r_before - SHARP_TARGET).abs(),
            "feedback must move toward target: {r_before} -> {r_after}"
        );
        assert_eq!(out.dim(), soft.dim());
    }
}
