//! V1: параметрическая модель растра «оценить → синтезировать → вычесть».
//!
//! Растр — квазипериодический АМ-сигнал `s(x) = a(x)·cos(2π f·x + φ(x))` с
//! медленными `a` и `φ`. Оценка самих огибающей и фазы (а не «дырки» в спектре)
//! описывает и несущую, и её юбку, поэтому вычитание снимает больше, чем notch:
//!
//! ```text
//! z     = x_ac · exp(-i2π f·x)      комплексная демодуляция
//! z_lp  = НЧ(z)                     огибающая (a/2)·exp(iφ)
//! s     = 2·Re{ z_lp · exp(+i2π f·x) }
//! out   = x − w·s,   w = smoothstep(gate_lo, gate_hi, coh)
//! coh   = 2|z_lp| / НЧ(|z|)          1 у чистой синусоиды, ~0 у широкополосного
//! ```
//!
//! Гейт по когерентности обязателен: в той же полосе живёт контент (текст,
//! тонкие штрихи), не когерентный несущей, — его вычитать нельзя.
//!
//! **Домен.** Демодуляция выполняется в НАТИВНОМ разрешении. На chain-кадре
//! (r = 0.5, период ровно 2 px) `cos(2π·0.5·n + φ) = (−1)ⁿ cos φ`: боковые
//! полосы АМ складываются друг с другом и демодуляция вырождена.
//!
//! **Замер (JOURNAL 2.32).** На контонном эталоне (4 пары): PSNR вход 19.10 →
//! notch 17.42 → V1 21.39. На реальных сканах полоса несущей снимается:
//! 1821145u_1 71.7 % → 94.9 %, BookCOver 2.6 % → 53.1 %, Offset1 14.5 % → 26.9 %,
//! Pack 12.7 % → 22.8 %; на скане без растра гейт закрывается и подмена не
//! происходит.

use ndarray::Array3;

use crate::prefilter::{build_notch_lists, luma_of, PrefilterCfg};

/// Параметры V1. Значения выведены из измерений на 11 реальных сканах и
/// сквозной проверки на контонном эталоне (JOURNAL 2.32), не подобраны под
/// отдельный скан.
#[derive(Clone, Copy, Debug)]
pub struct V1Cfg {
    /// НЧ огибающей в пикселях нативного скана. Уже, чем 1/σ, полосы не снимает.
    pub sigma: f32,
    /// Нижний/верхний порог гейта по когерентности.
    pub gate_lo: f32,
    pub gate_hi: f32,
    /// Несущих на канал (плиты CMYK дают несколько углов на одном радиусе).
    pub max_carriers: usize,
    /// Допуск по радиусу относительно опорной оценки r.
    pub r_tol: f32,
    /// Средняя когерентность, ниже которой вычитание считается несостоявшимся
    /// (тогда разрешено откатиться на классический notch).
    pub min_coh: f32,
    /// Гармоники несущей (k = 1..harmonics).
    pub harmonics: usize,
}

impl Default for V1Cfg {
    fn default() -> Self {
        Self {
            sigma: 16.0,
            gate_lo: 0.0,
            gate_hi: 0.5,
            // 4 = число печатных плит CMYK: на Calendar1 при 2 несущих оставалась
            // горизонтальная полосатость от третьей плиты (JOURNAL 2.32).
            max_carriers: 4,
            r_tol: 0.02,
            min_coh: 0.15,
            harmonics: 1,
        }
    }
}

impl V1Cfg {
    /// Настройки из окружения (для бенчмарка и A/B без пересборки).
    /// `DESCREEN_V1_SIGMA`, `DESCREEN_V1_GATE` ("lo,hi"), `DESCREEN_V1_CARRIERS`.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Ok(v) = std::env::var("DESCREEN_V1_SIGMA") {
            if let Ok(x) = v.parse::<f32>() {
                if x > 0.0 {
                    c.sigma = x;
                }
            }
        }
        if let Ok(v) = std::env::var("DESCREEN_V1_GATE") {
            let parts: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if parts.len() == 2 && parts[0] < parts[1] {
                c.gate_lo = parts[0];
                c.gate_hi = parts[1];
            }
        }
        if let Ok(v) = std::env::var("DESCREEN_V1_CARRIERS") {
            if let Ok(x) = v.parse::<usize>() {
                if x > 0 {
                    c.max_carriers = x;
                }
            }
        }
        c
    }
}

/// Отчёт о применении: что нашли и насколько когерентно.
pub struct V1Report {
    pub r_ref: f32,
    pub carriers: Vec<Vec<(f32, f32)>>,
    pub mean_coh: f32,
    pub acted: bool,
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo + 1e-12)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Сепарабельный box-усреднитель с reflect-краями, O(1) на пиксель.
fn box_pass(src: &[f32], dst: &mut [f32], h: usize, w: usize, r: usize, rows: bool) {
    let mut acc: Vec<f32> = vec![0.0; if rows { w + 1 } else { h + 1 }];
    let map = |i: isize, n: usize| -> usize {
        if i < 0 {
            (-i) as usize
        } else if i >= n as isize {
            (2 * n as isize - 2 - i).max(0) as usize
        } else {
            i as usize
        }
    };
    if rows {
        for y in 0..h {
            let base = y * w;
            acc[0] = 0.0;
            for x in 0..w {
                acc[x + 1] = acc[x] + src[base + x];
            }
            for x in 0..w {
                let lo = x as isize - r as isize;
                let hi = x as isize + r as isize + 1;
                let mut s = 0.0f32;
                // разбиение на [lo,0) [0,w) [w,hi) с reflect
                let mut i = lo;
                while i < hi {
                    if i < 0 {
                        s += src[base + map(i, w)];
                    } else if i >= w as isize {
                        s += src[base + map(i, w)];
                    } else {
                        // непрерывный участок внутри кадра — берём префиксные суммы
                        let a = i as usize;
                        let b = (hi.min(w as isize)) as usize;
                        s += acc[b] - acc[a];
                        i = b as isize;
                        continue;
                    }
                    i += 1;
                }
                dst[base + x] = s / (2 * r + 1) as f32;
            }
        }
    } else {
        for x in 0..w {
            acc[0] = 0.0;
            for y in 0..h {
                acc[y + 1] = acc[y] + src[y * w + x];
            }
            for y in 0..h {
                let lo = y as isize - r as isize;
                let hi = y as isize + r as isize + 1;
                let mut s = 0.0f32;
                let mut i = lo;
                while i < hi {
                    if i < 0 || i >= h as isize {
                        s += src[map(i, h) * w + x];
                        i += 1;
                    } else {
                        let a = i as usize;
                        let b = (hi.min(h as isize)) as usize;
                        s += acc[b] - acc[a];
                        i = b as isize;
                    }
                }
                dst[y * w + x] = s / (2 * r + 1) as f32;
            }
        }
    }
}

/// Приближение гаусса тремя box-проходами (σ² = 3·(k²−1)/12).
fn gauss_approx(src: &[f32], h: usize, w: usize, sigma: f32, scratch: &mut [f32]) -> Vec<f32> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let k = ((4.0 * sigma * sigma + 1.0).sqrt().round() as usize).max(1);
    let r = (k / 2).max(0);
    let mut a = src.to_vec();
    let mut b = vec![0.0f32; src.len()];
    for _ in 0..3 {
        box_pass(&a, &mut b, h, w, r, true);
        box_pass(&b, &mut a, h, w, r, false);
    }
    let _ = scratch;
    a
}

/// НЧ одной вещественной плоскости (для `|z|`).
fn lowpass(a: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let mut scratch = Vec::new();
    gauss_approx(a, h, w, sigma, &mut scratch)
}

/// Несущие на канал: цели notch, попавшие в кольцо ±r_tol от опорного r.
fn carriers_for(
    img: &Array3<f32>, cfg: &PrefilterCfg, r_ref: f32, v1: &V1Cfg, roi: usize
) -> Vec<Vec<(f32, f32)>> {
    let lists = build_notch_lists(img, cfg, roi);
    lists
        .into_iter()
        .map(|lst| {
            let mut out: Vec<(f32, f32)> = Vec::new();
            for (r, ang) in lst {
                if (r - r_ref).abs() > v1.r_tol {
                    continue;
                }
                let dup = out.iter().any(|(_, a)| {
                    let d = ((ang - a + 180.0) % 360.0 - 180.0).abs();
                    d < 5.0 || (d - 180.0).abs() < 5.0
                });
                if !dup {
                    out.push((r, ang));
                }
                if out.len() >= v1.max_carriers {
                    break;
                }
            }
            out
        })
        .collect()
}

/// Вычитание растра на одной несущей, одна плоскость (`plane` меняется на месте).
///
/// Фаза считается как произведение разделяемых векторов `exp(-i·2π·f_x·x)` и
/// `exp(-i·2π·f_y·y)`, каждый в f64. Прямое `sin_cos` от накопленного
/// аргумента в f32 недопустимо: на кадре 9000 px при f≈0.14 аргумент достигает
/// тысяч радиан, и точность f32 даёт ошибку фазы порядка 0.3 рад — вычитание
/// начинает бить не в фазе и ДОБАВЛЯЕТ энергию вместо снятия.
fn subtract_carrier(
    plane: &mut [f32], xac: &[f32], h: usize, w: usize, fx: f32, fy: f32, v1: &V1Cfg
) -> f32 {
    let len = h * w;
    let mut zre = vec![0.0f32; len];
    let mut zim = vec![0.0f32; len];
    let mut coh_sum = 0.0f64;
    for k in 1..=v1.harmonics.max(1) {
        let kf = k as f64;
        let ex: Vec<(f32, f32)> = (0..w)
            .map(|x| {
                let a = -2.0 * std::f64::consts::PI * kf * fx as f64 * x as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let ey: Vec<(f32, f32)> = (0..h)
            .map(|y| {
                let a = -2.0 * std::f64::consts::PI * kf * fy as f64 * y as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        // e^{-i(θx+θy)} = (cx−i·sx)(cy−i·sy)
        for y in 0..h {
            let (cy, sy) = ey[y];
            for x in 0..w {
                let (cx, sx) = ex[x];
                let ct = cx * cy - sx * sy;
                let st = sx * cy + cx * sy;
                let v = xac[y * w + x];
                zre[y * w + x] = v * ct;
                zim[y * w + x] = -v * st;
            }
        }
        let zlp_re = lowpass(&zre, h, w, v1.sigma);
        let zlp_im = lowpass(&zim, h, w, v1.sigma);
        // |z| = |xac| (модуль фазового множителя равен 1)
        let abs_lp = lowpass(&xac.iter().map(|v| v.abs()).collect::<Vec<f32>>(), h, w, v1.sigma);
        for y in 0..h {
            let (cy, sy) = ey[y];
            for x in 0..w {
                let i = y * w + x;
                let mag = (zlp_re[i] * zlp_re[i] + zlp_im[i] * zlp_im[i]).sqrt();
                let coh = 2.0 * mag / (abs_lp[i] + 1e-9);
                if k == 1 {
                    coh_sum += coh as f64;
                }
                let wgt = smoothstep(v1.gate_lo, v1.gate_hi, coh);
                if wgt <= 0.0 {
                    continue;
                }
                let (cx, sx) = ex[x];
                // Re{zlp·e^{+iθ}} = a·cosθ − b·sinθ
                let ct = cx * cy - sx * sy;
                let st = sx * cy + cx * sy;
                let synth = 2.0 * (zlp_re[i] * ct - zlp_im[i] * st);
                plane[i] -= wgt * synth;
            }
        }
    }
    (coh_sum / len as f64) as f32
}

/// Применить V1 к изображению (нативное разрешение, до Найквист-даунсэмпла).
pub fn apply(img: &mut Array3<f32>, r_ref: f32, v1: &V1Cfg, cfg: &PrefilterCfg, roi: usize) -> V1Report {
    let (_, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let carriers = if r_ref > 0.0 {
        carriers_for(img, cfg, r_ref, v1, roi)
    } else {
        vec![Vec::new(); 3]
    };
    let mut cohs: Vec<f32> = Vec::new();
    for ch in 0..3 {
        if carriers[ch].is_empty() {
            continue;
        }
        let mut plane: Vec<f32> = (0..h * w).map(|i| img[[ch, i / w, i % w]]).collect();
        // Снять гладкий фон: иначе постоянная составляющая при демодуляции
        // попадает в комплексную огибающую и глушит и амплитуду, и гейт.
        let f_mag = carriers[ch][0].0.max(1e-6);
        let sigma_dc = (1.5 / f_mag).max(2.0);
        let bg = lowpass(&plane, h, w, sigma_dc);
        let xac: Vec<f32> = (0..plane.len()).map(|i| plane[i] - bg[i]).collect();
        let mut ch_coh = 0.0f32;
        for (r, ang) in &carriers[ch] {
            let th0 = ang.to_radians();
            let fx = r * th0.cos();
            let fy = r * th0.sin();
            ch_coh = ch_coh.max(subtract_carrier(&mut plane, &xac, h, w, fx, fy, v1));
        }
        for i in 0..plane.len() {
            img[[ch, i / w, i % w]] = plane[i].clamp(0.0, 1.0);
        }
        cohs.push(ch_coh);
    }
    let mean_coh = if cohs.is_empty() {
        0.0
    } else {
        cohs.iter().sum::<f32>() / cohs.len() as f32
    };
    V1Report {
        r_ref,
        carriers,
        mean_coh,
        acted: mean_coh >= v1.min_coh && !cohs.is_empty(),
    }
}

/// Абсолютная энергия полосы в кольце (для отчёта/тестов).
///
/// Абсолютная, а не доля: после удаления растра доля бессмысленна —
/// знаменатель сжимается вместе с сигналом.
pub fn band_energy(gray: &ndarray::Array2<f32>, r_lo: f32, r_hi: f32) -> f32 {
    crate::prefilter::band_energy_abs(gray, r_lo, r_hi)
}

/// То же по яркости изображения (для логов/тестов).
pub fn luma_band(img: &Array3<f32>, r_lo: f32, r_hi: f32) -> f32 {
    band_energy(&luma_of(img), r_lo, r_hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn grating(h: usize, w: usize, fx: f32, fy: f32, amp: f32, tone: f32) -> Array3<f32> {
        let mut img = Array3::<f32>::zeros((3, h, w));
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let ph = 2.0 * std::f32::consts::PI * (fx * x as f32 + fy * y as f32);
                    img[[c, y, x]] = tone + amp * ph.cos();
                }
            }
        }
        img
    }

    #[test]
    fn box_matches_plain_average_interior() {
        let (h, w) = (16, 16);
        let src: Vec<f32> = (0..h * w).map(|i| (i % 7) as f32).collect();
        let mut dst = vec![0.0f32; h * w];
        box_pass(&src, &mut dst, h, w, 2, true);
        // внутренняя точка: среднее по окну 5
        let (y, x) = (8, 8);
        let want: f32 =
            (-2..=2).map(|d| src[y * w + (x as isize + d) as usize]).sum::<f32>() / 5.0;
        assert!((dst[y * w + x] - want).abs() < 1e-5);
    }

    #[test]
    fn v1_removes_coherent_grating() {
        // АМ-растр: несущая + медленная огибающая, поверх гладкого тона.
        let (h, w) = (256, 256);
        let (fx, fy) = (0.13f32, 0.045f32);
        let mut img = grating(h, w, fx, fy, 0.08, 0.5);
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let env = 1.0 + 0.3 * ((x as f32 / w as f32) * 6.28).sin();
                    img[[c, y, x]] = 0.5 + 0.08 * env
                        * (2.0 * std::f32::consts::PI * (fx * x as f32 + fy * y as f32)).cos();
                }
            }
        }
        let before = band_energy(&luma_of(&img), 0.10, 0.15);
        let cfg = PrefilterCfg::default();
        let v1 = V1Cfg::default();
        let mut work = img.clone();
        let mut coh_dbg = 0.0f32;
        // несущие задаются напрямую: у синтетической решётки детектор найдёт их сам
        for c in 0..3 {
            let mut plane: Vec<f32> = (0..h * w).map(|i| work[[c, i / w, i % w]]).collect();
            let f_mag = (fx * fx + fy * fy).sqrt();
            let sigma_dc = (1.5 / f_mag).max(2.0);
            let bg = lowpass(&plane, h, w, sigma_dc);
            let xac: Vec<f32> = (0..plane.len()).map(|i| plane[i] - bg[i]).collect();
            coh_dbg = subtract_carrier(&mut plane, &xac, h, w, fx, fy, &v1);
            for i in 0..plane.len() {
                work[[c, i / w, i % w]] = plane[i].clamp(0.0, 1.0);
            }
        }
        let after = band_energy(&luma_of(&work), 0.10, 0.15);
        // остаточная амплитуда на несущей: проекция на exp(i·2πf·x)
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let th = 2.0 * std::f64::consts::PI * (fx as f64 * x as f64 + fy as f64 * y as f64);
                let v = work[[0, y, x]] as f64 - 0.5;
                re += v * th.cos();
                im += v * th.sin();
            }
        }
        let amp = 2.0 * (re * re + im * im).sqrt() / (h * w) as f64;
        let mut rms = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let d = work[[0, y, x]] as f64 - 0.5;
                rms += d * d;
            }
        }
        rms = (rms / (h * w) as f64).sqrt();
        eprintln!("coh={coh_dbg:.3} before={before:.4} after={after:.4} residual_amp={amp:.6} rms={rms:.6}");
        assert!(
            after < before * 0.25,
            "V1 must drop the carrier band: {before} -> {after} (coh={coh_dbg}, amp={amp})"
        );
        let _ = cfg;
    }

    #[test]
    fn v1_apply_on_large_frame_removes_carrier() {
        // Кадр реального масштаба: на больших размерах ошибки фазы/блюра
        // проявляются так, как их не видно на 256 px.
        let (h, w) = (1536, 1536);
        let (fx, fy) = (0.13f32 * 0.766, 0.13f32 * 0.643);
        let mut img = Array3::<f32>::zeros((3, h, w));
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed & 0xFFFF) as f32 / 65535.0
        };
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    let env = 1.0 + 0.3 * ((x as f32 / w as f32) * 6.28).sin();
                    let content = 0.10 * (rnd() - 0.5);
                    img[[c, y, x]] = 0.5
                        + content
                        + 0.07 * env
                            * (2.0 * std::f32::consts::PI * (fx * x as f32 + fy * y as f32)).cos();
                }
            }
        }
        let before = band_energy(&luma_of(&img), 0.09, 0.12);
        let cfg = PrefilterCfg::default();
        let mut work = img.clone();
        let rep = apply(&mut work, 0.13, &V1Cfg::default(), &cfg, 1024);
        let after = band_energy(&luma_of(&work), 0.09, 0.12);
        eprintln!(
            "apply: coh={:.2} acted={} band {:.4e} -> {:.4e} ({:.1}%)",
            rep.mean_coh,
            rep.acted,
            before,
            after,
            100.0 * (1.0 - after / before)
        );
        assert!(rep.acted, "V1 must detect the carrier");
        assert!(
            after < before * 0.35,
            "V1 must drop the carrier band on a large frame: {before} -> {after}"
        );
    }

    #[test]
    fn gaussian_approx_preserves_mean() {        let (h, w) = (64, 64);
        let src: Vec<f32> = (0..h * w).map(|i| ((i * 37) % 91) as f32 / 90.0).collect();
        let out = lowpass(&src, h, w, 4.0);
        let m0: f32 = src.iter().sum::<f32>() / src.len() as f32;
        let m1: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((m0 - m1).abs() < 0.02, "{m0} vs {m1}");
    }

    #[test]
    fn band_energy_peaks_at_carrier() {
        let img = grating(256, 256, 0.13, 0.045, 0.1, 0.5);
        let inb = band_energy(&luma_of(&img), 0.10, 0.16);
        let outb = band_energy(&luma_of(&img), 0.30, 0.40);
        assert!(inb > outb * 10.0, "band {inb} vs {outb}");
    }
}
