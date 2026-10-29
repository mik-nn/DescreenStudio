//! Найквист–Котельников нормализация входа: приведение периода растра к
//! `1/r_out` px — «продукт в рабочем разрешении» (2.27, пересмотр 2.39).
//!
//! Даунсэмпл — интегрирующий box-фильтр (эквивалент апертуры сканера),
//! обратный апскейл результата — Lanczos3. Масштаб выбирается по A0-оценке
//! растра (scale = r/r_out) и verify-loop'ом доводится до измеренного
//! r_out ≈ r_target (недооценка дала бы алиасинг — 2.27).
//!
//! `r_out = 0.5` — предел Котельникова: период растра ровно 2 px, рабочая
//! полоса кадра обрезана на частоте растра (2.39: 20–62 % энергии скана
//! уходило в ноль — «мягкая пластиковая» картинка). `r_out = 0.25` даёт
//! вдвое более широкую полосу (период 4 px) при том же трейне.

use ndarray::{s, Array3};

use crate::prefilter::{estimate_raster, luma_of, PrefilterCfg};

/// Потолок масштаба: всегда сжимаем хотя бы до 65 % линейного размера. При
/// `r_out = 0.25` это срабатывает только на сканах, где A0 переоценил растр
/// (Magazine (2): 0.244 вместо 0.142) — иначе кадр остался бы в нативном
/// разрешении и сеть работала бы вне распределения на 70 Мпикс.
const SCALE_MAX: f32 = 0.65;

/// Параметры нормализации. `r_out` задаёт рабочую полосу кадра: 0.5 —
/// Котельников (период 2 px), 0.25 — вдвое шире (период 4 px).
#[derive(Clone, Copy, Debug)]
pub struct NyqCfg {
    pub r_out: f32,
}

impl Default for NyqCfg {
    fn default() -> Self {
        Self { r_out: 0.5 }
    }
}

impl NyqCfg {
    /// `DESCREEN_R_OUT` — переопределение рабочей частоты растра.
    pub fn from_env() -> Self {
        let r = std::env::var("DESCREEN_R_OUT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5);
        Self { r_out: r.clamp(0.15, 0.5) }
    }
}
/// Никогда не ужимаем сильнее 10× (страховка от мусорной оценки; обычный
/// минимум по референсу 2.27 — 0.1294, Offset2).
const SCALE_MIN: f32 = 0.1;
/// Масштабы ≥ этого значения не требуют ресемплинга.
pub const SCALE_SKIP: f32 = 0.995;

pub struct NyquistPlan {
    /// A0-оценка растра на исходном разрешении.
    pub r_in: f32,
    pub conf_in: f32,
    /// Итоговый линейный масштаб даунсэмпла (1.0 = ресемплинг не нужен).
    pub scale: f32,
    /// Измеренный r растра в ресемплированном входе.
    pub r_out: f32,
    pub out_w: usize,
    pub out_h: usize,
    pub iters: usize,
}

fn resize_box(v: &[f32], w_in: usize, w_out: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w_out];
    let a = w_in as f64 / w_out.max(1) as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let x0 = i as f64 * a;
        let x1 = (i + 1) as f64 * a;
        let lo = x0.floor() as usize;
        let hi = ((x1.ceil() as usize).max(lo + 1)).min(w_in);
        let (mut acc, mut wsum) = (0.0f64, 0.0f64);
        for k in lo..hi {
            // площадь пересечения исходного интервала [k, k+1) с окном [x0, x1)
            let wl = ((k + 1) as f64).min(x1) - (k as f64).max(x0);
            if wl > 0.0 {
                acc += v[k] as f64 * wl;
                wsum += wl;
            }
        }
        *o = (acc / wsum.max(1e-12)) as f32;
    }
    out
}

/// Интегрирующий даунсэмпл (box, фракционные окна) всего RGB-плейна.
pub fn resize_box_rgb(img: &Array3<f32>, scale: f32) -> Array3<f32> {
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let w2 = ((w as f64 * scale as f64).round() as usize).max(8);
    let h2 = ((h as f64 * scale as f64).round() as usize).max(8);
    let mut tmp = Array3::<f32>::zeros((c, h, w2));
    for ch in 0..c {
        for y in 0..h {
            let row: Vec<f32> = img.slice(s![ch, y, ..]).to_vec();
            let r = resize_box(&row, w, w2);
            for x in 0..w2 {
                tmp[[ch, y, x]] = r[x];
            }
        }
    }
    let mut out = Array3::<f32>::zeros((c, h2, w2));
    for ch in 0..c {
        for x in 0..w2 {
            let col: Vec<f32> = tmp.slice(s![ch, .., x]).to_vec();
            let r = resize_box(&col, h, h2);
            for y in 0..h2 {
                out[[ch, y, x]] = r[y];
            }
        }
    }
    out
}

fn lanczos3(t: f32) -> f32 {
    if t.abs() < 1e-6 {
        1.0
    } else if t.abs() < 3.0 {
        let x = std::f32::consts::PI * t;
        3.0 * x.sin() * (x / 3.0).sin() / (std::f32::consts::PI * std::f32::consts::PI * t * t)
    } else {
        0.0
    }
}

fn resize_lanczos3(v: &[f32], w_in: usize, w_out: usize) -> Vec<f32> {
    if w_out == w_in {
        return v.to_vec();
    }
    let mut out = vec![0.0f32; w_out];
    let a = w_in as f64 / w_out.max(1) as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let x = (i as f64 + 0.5) * a - 0.5;
        let k0 = x.floor() as i64;
        let (mut acc, mut wsum) = (0.0f32, 0.0f32);
        for k in k0 - 2..=k0 + 3 {
            let kk = k.clamp(0, w_in as i64 - 1) as usize;
            let wt = lanczos3(((k as f64) - x).abs() as f32);
            if wt != 0.0 {
                acc += v[kk] * wt;
                wsum += wt;
            }
        }
        *o = (acc / wsum.max(1e-9)) as f32;
    }
    out
}

/// Обратный апскейл результата (Lanczos3, сохранение DC нормировкой весов).
pub fn upsample_rgb(img: &Array3<f32>, w_out: usize, h_out: usize) -> Array3<f32> {
    let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    if w_out == w && h_out == h {
        return img.clone();
    }
    let mut tmp = Array3::<f32>::zeros((c, h, w_out));
    for ch in 0..c {
        for y in 0..h {
            let row: Vec<f32> = img.slice(s![ch, y, ..]).to_vec();
            let r = resize_lanczos3(&row, w, w_out);
            for x in 0..w_out {
                tmp[[ch, y, x]] = r[x];
            }
        }
    }
    let mut out = Array3::<f32>::zeros((c, h_out, w_out));
    for ch in 0..c {
        for x in 0..w_out {
            let col: Vec<f32> = tmp.slice(s![ch, .., x]).to_vec();
            let r = resize_lanczos3(&col, h, h_out);
            for y in 0..h_out {
                out[[ch, y, x]] = r[y];
            }
        }
    }
    out
}

/// План нормализации: scale = r_in / r_out по A0-оценке нативного растра.
///
/// Verify-loop по chain-кадру убран (2.39): на границе Найквиста детектор
/// читает алиас второй гармоники (Pack 0.013 при истинном 0.500, Offset2
/// 0.297) и «доводил» сжатие вдвое сильнее расчётного — Pack терял ровно
/// половину рабочей полосы. A0 на нативном кадре точен (±0.0005 на шести
/// сканах, сверено с прямым пиком спектра), поэтому план берётся из него.
pub fn plan(src: &Array3<f32>, pcfg: &PrefilterCfg) -> NyquistPlan {
    plan_with(src, pcfg, &NyqCfg::default())
}

pub fn plan_with(src: &Array3<f32>, pcfg: &PrefilterCfg, cfg: &NyqCfg) -> NyquistPlan {
    let target = cfg.r_out;
    let (h, w) = (src.shape()[1], src.shape()[2]);
    let e0 = estimate_raster(&luma_of(src), pcfg, 1024);
    // растр уже у цели / не найден — ресемплинг не нужен
    if e0.confidence <= 0.05 || e0.r < 0.01 || e0.r >= target * 0.96 {
        return NyquistPlan {
            r_in: e0.r,
            conf_in: e0.confidence,
            scale: 1.0,
            r_out: e0.r,
            out_w: w,
            out_h: h,
            iters: 0,
        };
    }
    let scale = (e0.r / target).clamp(SCALE_MIN, SCALE_MAX) as f32;
    let out_w = ((w as f64 * scale as f64).round() as usize).max(8);
    let out_h = ((h as f64 * scale as f64).round() as usize).max(8);
    // r_out — расчётная частота растра в рабочем кадре (не измеренная:
    // измерение на границе Найквиста недостоверно, см. docstring выше).
    let r_out = e0.r / scale;
    NyquistPlan {
        r_in: e0.r,
        conf_in: e0.confidence,
        scale,
        r_out,
        out_w,
        out_h,
        iters: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_downsample_matches_block_average() {
        let row: Vec<f32> = (0..16).map(|x| (x as f32 % 7.0) / 7.0).collect();
        // точное интегрирование окна в 1D: целочисленный масштаб 16→8
        let got = resize_box(&row, 16, 8);
        for x in 0..8 {
            let want = (row[2 * x] + row[2 * x + 1]) / 2.0;
            assert!((got[x] - want).abs() < 1e-5, "x={x} {} vs {}", got[x], want);
        }
        // дробный масштаб 16→5: окна [x0, x1) с частичными весами
        let got5 = resize_box(&row, 16, 5);
        let ms: f32 = row.iter().sum::<f32>() / row.len() as f32;
        let mg: f32 = got5.iter().sum::<f32>() / got5.len() as f32;
        assert!((ms - mg).abs() < 0.02, "{ms} vs {mg}");
    }

    #[test]
    fn box_mean_preserved() {
        let src: Vec<f32> = (0..99).map(|i| (i * 13 % 17) as f32 / 16.0).collect();
        let got = resize_box(&src, 99, 31);
        let ms: f32 = src.iter().sum::<f32>() / src.len() as f32;
        let mg: f32 = got.iter().sum::<f32>() / got.len() as f32;
        assert!((ms - mg).abs() < 0.02, "{ms} vs {mg}");
    }

    #[test]
    fn lanczos_upsample_preserves_mean() {
        // гладкий вход: нормированные веса Lanczos3 сохраняют среднее
        let src: Vec<f32> = (0..24).map(|i| i as f32 / 47.0).collect();
        let got = resize_lanczos3(&src, 24, 48);
        let ms: f32 = src.iter().sum::<f32>() / src.len() as f32;
        let mg: f32 = got.iter().sum::<f32>() / got.len() as f32;
        assert!((ms - mg).abs() < 0.01, "{ms} vs {mg}");
        // интерполяция гладкая: выход монотонен
        for i in 1..got.len() {
            assert!(got[i] >= got[i - 1] - 2e-4, "non-monotone at {i}: {} -> {}", got[i - 1], got[i]);
        }
    }

    #[test]
    fn roundtrip_rgb_shapes() {
        // гладкий градиент + слабый косинус: дрейф roundtrip должен быть мал
        let img = Array3::from_shape_fn((3, 51, 37), |(_, y, x)| {
            let t = (y as f32 / 50.0 + x as f32 / 36.0) * 0.5
                + 0.05 * ((std::f32::consts::PI * 2.0 * x as f32 / 18.0).sin());
            t.clamp(0.0, 1.0)
        });
        let down = resize_box_rgb(&img, 0.3);
        assert_eq!(down.shape()[1..], [15, 11]);
        let up = upsample_rgb(&down, 37, 51);
        assert_eq!(up.shape(), img.shape());
        let d: f32 = (&up - &img).mapv(f32::abs).mean().unwrap();
        assert!(d < 0.05, "roundtrip drift {d}");
    }
}
