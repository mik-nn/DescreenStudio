use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, conv2d, Conv2dConfig, ConvTranspose2dConfig, conv_transpose2d};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use clap::Parser;

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
                    .contiguous()?;
                patches.push(patch);
            }
            let batch_in = Tensor::stack(&patches, 0)?;
            let outs = model.forward(&batch_in)?.to_dtype(DType::F32)?;
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
        let whole = model.forward(&input)?;
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

// --- Типы спектральных блоков ---

enum SpectralType {
    Simple(SpectralBlockSimple),
    Complex(SpectralBlockComplex),
}

// Простой блок (encs.0, decs.1) - ожидает вес прямо в текущем vb: "spectral.weight"
struct SpectralBlockSimple {
    conv: candle_nn::Conv2d,
}

impl SpectralBlockSimple {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self, candle_core::Error> {
        // Ключ: ...spectral.weight (vb уже указывает на префикс "spectral")
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv = conv2d(channels, channels, 3, cfg, vb)?; 
        Ok(Self { conv })
    }
}

impl Module for SpectralBlockSimple {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.conv.forward(xs)
    }
}

include!("spectral_block_impl.rs");

// --- FFC Блок ---

struct FFCBlock {
    spatial: candle_nn::Conv2d,
    spectral: SpectralType,
    fuse: candle_nn::Conv2d,
}

impl FFCBlock {
    fn new(vb: VarBuilder, channels: usize, is_simple: bool) -> Result<Self, candle_core::Error> {
        // 1. Spatial: ключ "...spatial.weight"
        let cfg_spatial = Conv2dConfig { padding: 1, ..Default::default() };
        let spatial = conv2d(channels, channels, 3, cfg_spatial, vb.pp("spatial"))?;
        
        // 2. Spectral: 
        // Если is_simple=true: ключ "...spectral.weight" -> передаем vb.pp("spectral") в Simple
        // Если is_simple=false: ключи "...spectral.conv_real.0.weight" -> передаем vb.pp("spectral") в Complex
        let spectral_vb = vb.pp("spectral");
        let spectral = if is_simple {
            SpectralType::Simple(SpectralBlockSimple::new(spectral_vb, channels)?)
        } else {
            SpectralType::Complex(SpectralBlockComplex::new(spectral_vb, channels)?)
        };
        
        // 3. Fuse: ключ "...fuse.0.weight"
        let cfg_fuse = Conv2dConfig { padding: 0, ..Default::default() };
        let fuse = conv2d(channels * 2, channels, 1, cfg_fuse, vb.pp("fuse").pp("0"))?;

        Ok(Self { spatial, spectral, fuse })
    }
}

impl Module for FFCBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x_spatial = self.spatial.forward(xs)?;
        let x_spectral = match &self.spectral {
            SpectralType::Simple(b) => b.forward(xs)?,
            SpectralType::Complex(b) => b.forward(xs)?,
        };
        
        let cat = Tensor::cat(&[&x_spatial, &x_spectral], 1)?;
        let out = self.fuse.forward(&cat)?.gelu_erf()?;
        
        (&out + xs)?.to_dtype(xs.dtype())
    }
}

// --- Основная модель ---

struct FourierUNet {
    inc: candle_nn::Conv2d,
    enc0: FFCBlock,      
    down0: candle_nn::Conv2d, 
    enc1: FFCBlock,      
    down1: candle_nn::Conv2d, 
    
    b0: FFCBlock,        
    b1: FFCBlock,        
    b2: FFCBlock,        
    
    up0: candle_nn::ConvTranspose2d, 
    dec0: FFCBlock,      
    up1: candle_nn::ConvTranspose2d, 
    dec1: FFCBlock,      
    outc: candle_nn::Conv2d,
}

impl FourierUNet {
    fn load(device: &Device, dtype: DType, model_path: &str) -> Result<Self, candle_core::Error> {
        let buffer = std::fs::read(model_path)?;
        let safetensors = SafeTensors::deserialize(&buffer)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for name in safetensors.names() {
            let view = safetensors.tensor(name)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let tensor = Tensor::from_raw_buffer(view.data(), match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                _ => DType::F32,
            }, view.shape(), device)?;
            tensors.insert(name.to_string(), tensor);
        }

        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        // 1. Input: 3 -> 96 (inc.weight)
        let inc = conv2d(3, 96, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("inc"))?;

        // 2. Encoder 0: 96 -> 96 (Simple: encs.0.0.spectral.weight)
        // Путь: encs -> 0 -> 0. Внутри этого vb ищем spatial, spectral, fuse.
        let enc0 = FFCBlock::new(vb.pp("encs").pp(0).pp(0), 96, true)?;
        
        // 3. Down 0: 96 -> 192 (downs.0.op.0.weight)
        let down0 = conv2d(96, 192, 3, Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("downs").pp(0).pp("op").pp(0))?;

        // 4. Encoder 1: 192 -> 192 (Complex: encs.1.0.spectral.conv_real...)
        let enc1 = FFCBlock::new(vb.pp("encs").pp(1).pp(0), 192, false)?;

        // 5. Down 1: 192 -> 384 (downs.1.op.0.weight)
        let down1 = conv2d(192, 384, 3, Conv2dConfig { padding: 1, stride: 2, ..Default::default() }, vb.pp("downs").pp(1).pp("op").pp(0))?;

        // 6. Bottleneck: 384 -> 384 (Complex: bottleneck.0.spectral...)
        let b0 = FFCBlock::new(vb.pp("bottleneck").pp(0), 384, false)?;
        let b1 = FFCBlock::new(vb.pp("bottleneck").pp(1), 384, false)?;
        let b2 = FFCBlock::new(vb.pp("bottleneck").pp(2), 384, false)?;

        // 7. Up 0: 384 -> 192 (ups.0.op.0.weight [384, 192, 4, 4])
        let up0_cfg = ConvTranspose2dConfig { padding: 1, stride: 2, ..Default::default() };
        let up0 = conv_transpose2d(384, 192, 4, up0_cfg, vb.pp("ups").pp(0).pp("op").pp(0))?;

        // 8. Decoder 0: 192 -> 192 (Complex: decs.0.0.spectral.conv_real...)
        let dec0 = FFCBlock::new(vb.pp("decs").pp(0).pp(0), 192, false)?;

        // 9. Up 1: 192 -> 96 (ups.1.op.0.weight [192, 96, 4, 4])
        let up1_cfg = ConvTranspose2dConfig { padding: 1, stride: 2, ..Default::default() };
        let up1 = conv_transpose2d(192, 96, 4, up1_cfg, vb.pp("ups").pp(1).pp("op").pp(0))?;

        // 10. Decoder 1: 96 -> 96 (Simple: decs.1.0.spectral.weight [96, 96, 3, 3])
        let dec1 = FFCBlock::new(vb.pp("decs").pp(1).pp(0), 96, true)?;

        // 11. Output: 96 -> 3 (outc.weight)
        let outc = conv2d(96, 3, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("outc"))?;

        Ok(Self { inc, enc0, down0, enc1, down1, b0, b1, b2, up0, dec0, up1, dec1, outc })
    }
}

impl Module for FourierUNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        // Зеркалит ml/model/unet.py: GELU после inc/down/up, skip-connections
        // (feats — выходы энкодеров до down), выход = clamp(x - residual, 0, 1).
        let dump = std::env::var("CANDLE_DUMP").is_ok();
        let stat = |t: &Tensor, tag: &str| {
            if dump {
                if let Ok(v) = t.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
                    let m = v.iter().sum::<f32>() / v.len() as f32;
                    let var = v.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / v.len() as f32;
                    eprintln!("stage {tag}: mean={m:.5} std={:.5} first4={:?}", var.sqrt(), &v[..4]);
                }
            }
        };
        let input = xs.clone();
        let mut x = self.inc.forward(xs)?.gelu_erf()?;
        stat(&x, "inc");

        x = self.enc0.forward(&x)?;
        stat(&x, "enc0");
        let skip0 = x.clone();
        x = self.down0.forward(&x)?.gelu_erf()?;

        x = self.enc1.forward(&x)?;
        stat(&x, "enc1");
        let skip1 = x.clone();
        x = self.down1.forward(&x)?.gelu_erf()?;

        x = self.b0.forward(&x)?;
        x = self.b1.forward(&x)?;
        x = self.b2.forward(&x)?;
        stat(&x, "bottleneck");

        x = self.up0.forward(&x)?.gelu_erf()?;
        x = (&x + &skip1)?;
        x = self.dec0.forward(&x)?;
        stat(&x, "dec0");

        x = self.up1.forward(&x)?.gelu_erf()?;
        x = (&x + &skip0)?;
        x = self.dec1.forward(&x)?;
        stat(&x, "dec1");

        let residual = self.outc.forward(&x)?;
        stat(&residual, "residual");
        input.sub(&residual)?.clamp(0f64, 1f64)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or(Device::Cpu),
        "metal" => Device::new_metal(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
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
                    run_tiled(&model, &device, h, w, tile, args.overlap, args.runs, args.verify, src, args.batch)?;
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
            let _ = model.forward(&input)?;
            device.synchronize()?;

            let iters = args.runs;
            println!("Бенчмарк ({} итераций, {}x{})...", iters, h, w);
            let start = Instant::now();
            for _ in 0..iters {
                let _ = model.forward(&input)?;
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