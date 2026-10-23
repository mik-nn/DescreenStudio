use candle_core::{DType, Device, Tensor};
use std::path::Path;
use std::time::Instant;
use clap::Parser;

use descreen_studio::candle_engine::build_cond_vec;
use descreen_studio::candle_model::FourierUNet;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long, default_value = "cuda")]
    device: String,
    /// Высота входа, px
    #[arg(long, default_value_t = 512)]
    h: usize,
    /// Ширина входа, px
    #[arg(long, default_value_t = 512)]
    w: usize,
    /// Число итераций бенчмарка
    #[arg(long, default_value_t = 10)]
    runs: usize,
    /// Размер тайла; 0 = прогон целого изображения без тайлинга
    #[arg(long, default_value_t = 0)]
    tile: usize,
    /// Перекрытие тайлов, px
    #[arg(long, default_value_t = 64)]
    overlap: usize,
    /// Тайлов за один forward (GPU-утилизация); 1 = прежнее поведение
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Сверить тайловый результат с цельным прогоном (mean |diff|)
    #[arg(long, default_value_t = false)]
    verify: bool,
    /// Путь к входному изображению вместо синтетического randn
    #[arg(long)]
    image: Option<String>,
    /// A0-cond для FiLM-моделей: "r,angle_deg,confidence" (напр. 0.147,15.0,0.8)
    #[arg(long)]
    cond: Option<String>,
    /// Куда сохранить результат (PNG)
    #[arg(long)]
    out: Option<String>,
}

/// Загрузка изображения в [3, H, W] f32 [0, 1] (luma-сохраняющий порядок RGB).
fn load_image(path: &str) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for (i, p) in img.pixels().enumerate() {
        data[i] = p[0] as f32 / 255.0;
        data[h * w + i] = p[1] as f32 / 255.0;
        data[2 * h * w + i] = p[2] as f32 / 255.0;
    }
    Ok((h, w, data))
}

/// Сохранение [3, H, W] f32 [0, 1] в PNG (pixel-major RGB!).
fn save_image(path: &str, h: usize, w: usize, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let n = h * w;
    let mut bytes = vec![0u8; 3 * n];
    for i in 0..n {
        bytes[3 * i] = (data[i].clamp(0.0, 1.0) * 255.0).round() as u8;
        bytes[3 * i + 1] = (data[n + i].clamp(0.0, 1.0) * 255.0).round() as u8;
        bytes[3 * i + 2] = (data[2 * n + i].clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    let img = image::RgbImage::from_raw(w as u32, h as u32, bytes)
        .ok_or("shape mismatch on save")?;
    img.save(path)?;
    Ok(())
}

/// Тайловый прогон с Hann overlap-add; зеркалит `InferenceEngine::descreen_progress`
/// (src/inference.rs). Возвращает (время прогона, число тайлов, delta шва col62/70,
/// средн. |tiled-whole| если verify).
fn run_tiled(
    model: &FourierUNet,
    device: &Device,
    h: usize,
    w: usize,
    tile: usize,
    overlap: usize,
    iters: usize,
    verify: bool,
    src: Vec<f32>,
    batch: usize,
    cond: Option<[f32; 4]>,
) -> Result<(std::time::Duration, usize, f32, Option<f32>, Vec<f32>), candle_core::Error> {
    let stride = tile.saturating_sub(overlap).max(2);

    // Replicate-pad: сетка тайлов заканчивается ровно на краю padded-изображения.
    let ty = ((h as isize - tile as isize) as f64 / stride as f64).ceil().max(0.0) as usize;
    let tx = ((w as isize - tile as isize) as f64 / stride as f64).ceil().max(0.0) as usize;
    let ph = ty * stride + tile;
    let pw = tx * stride + tile;

    let mut padded = vec![0f32; 3 * ph * pw];
    for c in 0..3 {
        for y in 0..ph {
            let sy = y.min(h - 1);
            for x in 0..pw {
                let sx = x.min(w - 1);
                padded[c * ph * pw + y * pw + x] = src[c * h * w + sy * w + sx];
            }
        }
    }
    let padded = Tensor::from_vec(padded, (1, 3, ph, pw), device)?;

    // Окно Ханна [1, 1, tile, tile] на устройстве.
    let win1d: Vec<f32> = (0..tile)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / (tile as f32 - 1.0);
            0.5 * (1.0 - t.cos())
        })
        .collect();
    let mut win2d = vec![0f32; tile * tile];
    for y in 0..tile {
        for x in 0..tile {
            win2d[y * tile + x] = win1d[y] * win1d[x];
        }
    }
    let window = Tensor::from_vec(win2d, (1, 1, tile, tile), device)?;
    let _window = window.broadcast_as((1, 3, tile, tile))?;

    let tiles_x = (pw - tile) / stride + 1;
    let tiles_y = (ph - tile) / stride + 1;
    let total = tiles_x * tiles_y;

    // Накопители на CPU: acc [3, ph, pw], weight [ph, pw].
    let mut acc = vec![0f32; 3 * ph * pw];
    let mut weight = vec![0f32; ph * pw];

    let t0 = std::time::Instant::now();
    let cond_t = match cond {
        Some(c) => Some(Tensor::from_vec(c.to_vec(), (1, 4), device)?),
        None => None,
    };
    // Координаты всех тайлов; идём чанками по `batch` — один forward на чанк.
    let coords: Vec<(usize, usize)> = (0..tiles_y)
        .flat_map(|ty_i| (0..tiles_x).map(move |tx_i| (ty_i * stride, tx_i * stride)))
        .collect();
    for _ in 0..iters {
        for chunk in coords.chunks(batch.max(1)) {
            let mut patches = Vec::with_capacity(chunk.len());
            for &(y0, x0) in chunk {
                let patch = padded
                    .narrow(2, y0, tile)?
                    .narrow(3, x0, tile)?
                    .squeeze(0)?
                    .contiguous()?;
                patches.push(patch);
            }
            let batch_in = Tensor::stack(&patches, 0)?;
            let outs = model.forward_cond(&batch_in, cond_t.as_ref())?.to_dtype(DType::F32)?;
            for (b, &(y0, x0)) in chunk.iter().enumerate() {
                let out = outs.narrow(0, b, 1)?.flatten_all()?.to_vec1::<f32>()?;
                if std::env::var("CANDLE_DUMP").is_ok() && b == 0 {
                    let m = out.iter().sum::<f32>() / out.len() as f32;
                    eprintln!("tile fwd out: mean={m:.5} first4={:?}", &out[..4]);
                }
                for y in 0..tile {
                    let wy = win1d[y];
                    for x in 0..tile {
                        let wv = wy * win1d[x];
                        let wi = (y0 + y) * pw + (x0 + x);
                        // Вес добавляем ОДИН раз на пиксель (вне цикла по каналам!),
                        // иначе result = out·wv/(3·wv) = out/3.
                        weight[wi] += wv;
                        for c in 0..3 {
                            acc[c * ph * pw + wi] += out[c * tile * tile + y * tile + x] * wv;
                        }
                    }
                }
            }
        }
    }
    device.synchronize()?;
    let dur = t0.elapsed();

    // Нормировка и контроль шва (col 62 vs 70), как в descreen_bench.
    let mut seam = 0f32;
    for c in 0..3 {
        for y in 0..h {
            let a = acc[c * ph * pw + y * pw + 62] / weight[y * pw + 62].max(f32::EPSILON);
            let b = acc[c * ph * pw + y * pw + 70] / weight[y * pw + 70].max(f32::EPSILON);
            seam += (a.clamp(0.0, 1.0) - b.clamp(0.0, 1.0)).abs();
        }
    }
    seam /= (3 * h) as f32;

    // Нормированный результат (только реальные пиксели [3, h, w]).
    let mut result = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                result[c * h * w + y * w + x] =
                    (acc[c * ph * pw + y * pw + x] / weight[y * pw + x].max(f32::EPSILON))
                        .clamp(0.0, 1.0);
            }
        }
    }
    if std::env::var("CANDLE_DUMP").is_ok() {
        let wm: f32 = (0..h).map(|y| (0..w).map(|x| weight[y * pw + x]).sum::<f32>()).sum::<f32>()
            / (h * w) as f32;
        let rm = result.iter().sum::<f32>() / result.len() as f32;
        eprintln!("weight mean={wm:.5} result mean={rm:.5}");
        let (cy, cx) = (48usize, 48usize);
        eprintln!(
            "px(48,48): weight={:.5} acc={:.5} result={:.5}",
            weight[cy * pw + cx],
            acc[cy * pw + cx],
            result[cy * w + cx]
        );
    }

    // Verify: сравнение с цельным прогоном без тайлинга (только если влезает).
    let verify_diff = if verify {
        let input = Tensor::from_vec(src, (1, 3, h, w), device)?;
        let whole = model.forward_cond(&input, cond_t.as_ref())?;
        let whole = whole.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        // Сравниваем только интерьер: края искажены replicate-pad'ом.
        let m = (tile / 2).min(h / 4).min(w / 4);
        let mut sum = 0f32;
        let mut cnt = 0f32;
        for c in 0..3 {
            for y in m..h - m {
                for x in m..w - m {
                    let i = c * h * w + y * w + x;
                    sum += (result[i] - whole[i].clamp(0.0, 1.0)).abs();
                    cnt += 1.0;
                }
            }
        }
        Some(sum / cnt)
    } else {
        None
    };

    Ok((dur, total, seam, verify_diff, result))
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cond_vec = args.cond.as_deref().map(|s| {
        let p: Vec<f32> = s
            .split(',')
            .map(|x| x.trim().parse::<f32>().expect("cond: r,angle_deg,confidence"))
            .collect();
        assert_eq!(p.len(), 3, "cond expects r,angle_deg,confidence");
        build_cond_vec(p[0], p[1], p[2])
    });
    if let Some(c) = &cond_vec {
        println!("cond A0 -> [{:.3},{:.3},{:.3},{:.3}]", c[0], c[1], c[2], c[3]);
    }
    
    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or(Device::Cpu),
        "metal" => Device::new_metal(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    };
    let cond_t = match cond_vec {
        Some(c) => Some(Tensor::from_vec(c.to_vec(), (1, 4), &device)?),
        None => None,
    };
    
    println!("Устройство: {:?}", device);
    println!("Загрузка: {}", args.model);
    
    if !Path::new(&args.model).exists() {
        eprintln!("Файл не найден");
        return Ok(());
    }

    match FourierUNet::load(&device, DType::F32, &args.model) {
        Ok(model) => {
            println!("Модель загружена успешно!");

            if args.tile > 0 {
                let tile = args.tile.max(64);
                let (h, w, src) = match &args.image {
                    Some(path) => load_image(path)?,
                    None => {
                        let t = Tensor::randn(0f32, 1f32, (3, args.h, args.w), &Device::Cpu)?;
                        (args.h, args.w, t.flatten_all()?.to_vec1::<f32>()?)
                    }
                };
                let (dur, n, seam, vdiff, result) =
                    run_tiled(&model, &device, h, w, tile, args.overlap, args.runs, args.verify, src, args.batch, cond_vec)?;
                println!(
                    "Тайловый бенчмарк ({} ит., {}x{}, тайл {}, перекрытие {}, батч {}):",
                    args.runs, h, w, tile, args.overlap, args.batch
                );
                println!("Тайлов за прогон: {n}");
                println!("Среднее время за прогон: {:?}", dur / args.runs as u32);
                println!(
                    "  {:.0} мс/тайл, ~{:.1} с/страница A4 (48 тайлов)",
                    dur.as_secs_f64() / args.runs as f64 / n as f64 * 1000.0,
                    dur.as_secs_f64() / args.runs as f64 / n as f64 * 48.0
                );
                println!("Seam delta (col 62 vs 70): {:.4}", seam);
                if let Some(d) = vdiff {
                    println!("Verify vs whole-image (mean |diff|): {:.5}", d);
                }
                if let Some(out) = &args.out {
                    save_image(out, h, w, &result)?;
                    println!("Сохранено: {out}");
                }
                return Ok(());
            }

            let h = args.h;
            let w = args.w;
            let input = Tensor::randn(0f32, 1f32, (1, 3, h, w), &device)?;
            
            println!("Warmup...");
            let _ = model.forward_cond(&input, cond_t.as_ref())?;
            device.synchronize()?;

            let iters = args.runs;
            println!("Бенчмарк ({} итераций, {}x{})...", iters, h, w);
            let start = Instant::now();
            for _ in 0..iters {
                let _ = model.forward_cond(&input, cond_t.as_ref())?;
            }
            device.synchronize()?;
            let dur = start.elapsed();
            
            println!("Среднее время: {:?}", dur / iters as u32);
            println!("FPS: {:.2}", iters as f64 / dur.as_secs_f64());
        },
        Err(e) => eprintln!("Ошибка загрузки: {}", e),
    }

    Ok(())
}