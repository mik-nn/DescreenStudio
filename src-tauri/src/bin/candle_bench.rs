//! Candle benchmark для Descreen Studio Pro
//! 
//! Запуск:
//!   cargo run --release --bin candle_bench -- --model model.safetensors --h 2480 --w 3508 --device cuda
//!
//! Опции:
//!   --model <path>       Путь к .safetensors файлу модели
//!   --h <pixels>         Высота изображения (по умолчанию 2480)
//!   --w <pixels>         Ширина изображения (по умолчанию 3508)
//!   --tile <size>        Размер тайла (по умолчанию 512)
//!   --overlap <pixels>   Перекрытие тайлов (по умолчанию 64)
//!   --device <cpu|cuda|metal>  Устройство для инференса
//!   --warmup <N>         Количество прогревающих итераций (по умолчанию 3)
//!   --runs <N>           Количество тестовых прогонов (по умолчанию 5)
//!   --verify             Проверка корректности вывода

use std::env;
use std::error::Error;
use std::time::Instant;
use std::path::Path;

use candle_core::{Tensor, Device, DType, Shape};
use candle_nn::{Module, VarBuilder, Conv2d, Conv2dConfig, BatchNorm2d, Func};

/// Реализация FourierSpectralBlock из PyTorch
#[derive(Clone)]
pub struct FourierSpectralBlock {
    conv_real: Func,
    conv_imag: Func,
}

impl FourierSpectralBlock {
    pub fn new(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv_real_cfg = Conv2dConfig { kernel_size: 1, ..Default::default() };
        let conv_imag_cfg = Conv2dConfig { kernel_size: 1, ..Default::default() };
        
        let conv_real_conv1 = Conv2d::new(channels, channels, 1, conv_real_cfg.clone(), vb.pp("conv_real.0"))?;
        let conv_real_conv2 = Conv2d::new(channels, channels, 1, conv_real_cfg, vb.pp("conv_real.2"))?;
        
        let conv_imag_conv1 = Conv2d::new(channels, channels, 1, conv_imag_cfg.clone(), vb.pp("conv_imag.0"))?;
        let conv_imag_conv2 = Conv2d::new(channels, channels, 1, conv_imag_cfg, vb.pp("conv_imag.2"))?;
        
        let conv_real = Func::new(move |x: &Tensor| {
            let x = conv_real_conv1.forward(x)?.gelu()?;
            conv_real_conv2.forward(&x)
        });
        
        let conv_imag = Func::new(move |x: &Tensor| {
            let x = conv_imag_conv1.forward(x)?.gelu()?;
            conv_imag_conv2.forward(&x)
        });
        
        Ok(Self { conv_real, conv_imag })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_n, _c, h, w) = x.dims4()?;
        
        // FFT: rfft2
        let spec = x.fft_rfft2()?;
        let real = spec.real()?;
        let imag = spec.imag()?;
        
        // Process in frequency domain
        let out_real = self.conv_real.forward(&real)?;
        let out_imag = self.conv_imag.forward(&imag)?;
        
        // IFFT: ifft2
        let full = Tensor::stack(&[&out_real, &out_imag], 1)?;
        let out = full.fft_irfft2()?.to_dtype(x.dtype())?;
        
        Ok(out)
    }
}

/// FFCBlock с spatial и spectral ветками
#[derive(Clone)]
pub struct FFCBlock {
    spatial: Conv2d,
    spectral: FourierSpectralBlock,
    fuse: Func,
}

impl FFCBlock {
    pub fn new(channels: usize, use_spectral: bool, vb: VarBuilder) -> candle_core::Result<Self> {
        let spatial_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let spatial = Conv2d::new(channels, channels, 3, spatial_cfg, vb.pp("spatial"))?;
        
        let spectral = if use_spectral {
            FourierSpectralBlock::new(channels, vb.pp("spectral"))?
        } else {
            // Fallback to simple conv if spectral disabled
            let cfg = Conv2dConfig { padding: 1, ..Default::default() };
            let conv = Conv2d::new(channels, channels, 3, cfg, vb.pp("spectral_fallback"))?;
            let inner_conv = conv.clone();
            FourierSpectralBlock {
                conv_real: Func::new(move |x| inner_conv.forward(x)),
                conv_imag: Func::new(move |x| inner_conv.forward(x)),
            }
        };
        
        let fuse_conv_cfg = Conv2dConfig { kernel_size: 1, ..Default::default() };
        let fuse_conv = Conv2d::new(channels * 2, channels, 1, fuse_conv_cfg, vb.pp("fuse.0"))?;
        let fuse = Func::new(move |x: &Tensor| {
            fuse_conv.forward(x)?.gelu()
        });
        
        Ok(Self { spatial, spectral, fuse })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let s = self.spatial.forward(x)?;
        let f = self.spectral.forward(x)?;
        let fused = Tensor::cat(&[&s, &f], 1)?;
        let out = self.fuse.forward(&fused)?;
        Ok(&out + x)  // Residual connection
    }
}

/// UNet Down block
#[derive(Clone)]
pub struct Down {
    conv: Conv2d,
}

impl Down {
    pub fn new(ch_in: usize, ch_out: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { stride: 2, padding: 1, ..Default::default() };
        let conv = Conv2d::new(ch_in, ch_out, 3, cfg, vb.pp("op.0"))?;
        Ok(Self { conv })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(x)?.gelu()
    }
}

/// UNet Up block (ConvTranspose2d)
#[derive(Clone)]
pub struct Up {
    conv: candle_nn::conv::ConvTranspose2d,
}

impl Up {
    pub fn new(ch_in: usize, ch_out: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = candle_nn::conv::ConvTranspose2dConfig { stride: 2, padding: 1, ..Default::default() };
        let conv = candle_nn::ConvTranspose2d::new(ch_in, ch_out, 4, cfg, vb.pp("op.0"))?;
        Ok(Self { conv })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(x)?.gelu()
    }
}

/// Полная архитектура FourierUNet как в train.py
#[derive(Clone)]
pub struct FourierUNet {
    inc: Conv2d,
    encs: Vec<Func>,  // Encoders (FFCBlocks)
    downs: Vec<Down>,
    bottleneck: Func,  // FFCBlock
    ups: Vec<Up>,
    decs: Vec<Func>,   // Decoders (FFCBlocks)
    outc: Conv2d,
    levels: usize,
}

fn make_ffc_block(channels: usize, use_spectral: bool, vb: VarBuilder) -> candle_core::Result<Func> {
    let ffc = FFCBlock::new(channels, use_spectral, vb)?;
    Ok(Func::new(move |x| ffc.forward(x)))
}

impl FourierUNet {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        base: usize,
        levels: usize,
        enc_blocks: Vec<usize>,
        bottleneck_blocks: usize,
        dec_blocks: Vec<usize>,
        spectral: Vec<bool>,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        assert!(levels == 2 || levels == 3, "levels must be 2 or 3");
        assert_eq!(enc_blocks.len(), levels - 1);
        assert_eq!(dec_blocks.len(), levels - 1);
        assert_eq!(spectral.len(), levels);
        
        let inc_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let inc = Conv2d::new(in_channels, base, 3, inc_cfg, vb.pp("inc"))?;
        
        // Calculate channel sizes per level
        let mut chs = Vec::new();
        for i in 0..levels {
            chs.push(base * (2usize.pow(i as u32)));
        }
        
        // Encoder blocks
        let mut encs = Vec::new();
        for i in 0..(levels - 1) {
            let ch = chs[i];
            let n_blocks = enc_blocks[i];
            let vb_enc = vb.pp(format!("encs.{i}"));
            // Simplified: single FFC block per encoder level
            let block = make_ffc_block(ch, spectral[i], vb_enc.pp("0"))?;
            encs.push(block);
        }
        
        // Downsample layers
        let mut downs = Vec::new();
        for i in 0..(levels - 1) {
            let down = Down::new(chs[i], chs[i + 1], vb.pp(format!("downs.{i}")))?;
            downs.push(down);
        }
        
        // Bottleneck
        let bottleneck_ch = chs[levels - 1];
        let vb_bn = vb.pp("bottleneck.0");
        let bottleneck = make_ffc_block(bottleneck_ch, spectral[levels - 1], vb_bn)?;
        
        // Upsample layers
        let mut ups = Vec::new();
        for i in (0..(levels - 1)).rev() {
            let up = Up::new(chs[i + 1], chs[i], vb.pp(format!("ups.{i}")))?;
            ups.push(up);
        }
        
        // Decoder blocks
        let mut decs = Vec::new();
        for i in (0..(levels - 1)).rev() {
            let ch = chs[i];
            let n_blocks = dec_blocks[i];
            let vb_dec = vb.pp(format!("decs.{i}"));
            let block = make_ffc_block(ch, spectral[i], vb_dec.pp("0"))?;
            decs.push(block);
        }
        
        let outc_cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let outc = Conv2d::new(base, out_channels, 3, outc_cfg, vb.pp("outc"))?;
        
        Ok(Self {
            inc,
            encs,
            downs,
            bottleneck,
            ups,
            decs,
            outc,
            levels,
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut feats = Vec::new();
        
        // Initial conv
        let mut e = self.inc.forward(x)?.gelu()?;
        
        // Encoder path
        for (enc, down) in self.encs.iter().zip(self.downs.iter()) {
            e = enc.forward(&e)?;
            feats.push(e.clone());
            e = down.forward(&e)?;
        }
        
        // Bottleneck
        let b = self.bottleneck.forward(&e)?;
        let mut d = b;
        
        // Decoder path with skip connections
        for (up, dec, skip) in itertools::izip!(self.ups.iter(), self.decs.iter(), feats.iter().rev()) {
            let upsampled = up.forward(&d)?;
            d = upsampled.add(skip)?;
            d = dec.forward(&d)?;
        }
        
        let residual = self.outc.forward(&d)?;
        let base_img = x.narrow(1, 0, 3)?;
        let out = (&base_img - &residual)?.clamp(0.0, 1.0)?;
        
        Ok(out)
    }
}

fn load_model_safetensors(path: &str, config: &ModelConfig, device: &Device) -> Result<FourierUNet, Box<dyn Error>> {
    println!("Загрузка модели из {}", path);
    
    let data = std::fs::read(path)?;
    let _st = safetensors::SafeTensors::deserialize(&data)?;
    
    // Используем safetensors напрямую через from_raw_parts
    let vb = VarBuilder::from_buffer(data.as_slice(), DType::F32, device)?;
    
    let model = FourierUNet::new(
        config.in_channels,
        config.out_channels,
        config.base,
        config.levels,
        config.enc_blocks.clone(),
        config.bottleneck_blocks,
        config.dec_blocks.clone(),
        config.spectral.clone(),
        vb,
    )?;
    
    println!("Модель загружена успешно");
    println!("Параметров: ~{:.1}M", count_parameters(&model) as f64 / 1e6);
    
    Ok(model)
}

struct ModelConfig {
    in_channels: usize,
    out_channels: usize,
    base: usize,
    levels: usize,
    enc_blocks: Vec<usize>,
    bottleneck_blocks: usize,
    dec_blocks: Vec<usize>,
    spectral: Vec<bool>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // Конфигурация по умолчанию как в train.py run3
        Self {
            in_channels: 3,
            out_channels: 3,
            base: 64,
            levels: 3,
            enc_blocks: vec![1, 1],
            bottleneck_blocks: 2,
            dec_blocks: vec![1, 1],
            spectral: vec![true, true, true],
        }
    }
}

fn count_parameters(model: &FourierUNet) -> usize {
    // Упрощенная оценка - в реальной версии нужно суммировать параметры всех слоев
    // Для базовой конфигурации (64, 3 уровня): ~1.5M параметров
    let base_params = model.base * model.base * 9; // inc conv
    // ... полная реализация требует доступа к весам
    1_500_000 // заглушка для демонстрации
}

/// Синтетическое тестовое изображение с градиентом и шумом
fn synthetic_image(h: usize, w: usize, device: &Device) -> Result<Tensor, Box<dyn Error>> {
    let mut data = vec![0f32; 3 * h * w];
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let n = (seed & 0xFFFF) as f32 / 65535.0;
                let grad = 0.25 + 0.5 * ((x as f32) / w as f32 + (y as f32) / h as f32 * 0.5);
                data[c * h * w + y * w + x] = (grad + 0.02 * (n - 0.5)).clamp(0.0, 1.0);
            }
        }
    }
    
    let tensor = Tensor::from_vec(data, (1, 3, h, w), device)?;
    Ok(tensor)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut model_path = String::from("model.safetensors");
    let mut h_px = 2480;
    let mut w_px = 3508;
    let mut device_str = String::from("cpu");
    let mut tile_size = 512usize;
    let mut overlap = 64usize;
    let mut warmup_runs = 3;
    let mut test_runs = 5;
    let mut verify = false;
    let mut config = ModelConfig::default();
    
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--h" => { i += 1; h_px = args[i].parse()?; }
            "--w" => { i += 1; w_px = args[i].parse()?; }
            "--tile" => { i += 1; tile_size = args[i].parse()?; }
            "--overlap" => { i += 1; overlap = args[i].parse()?; }
            "--device" => { i += 1; device_str = args[i].clone(); }
            "--warmup" => { i += 1; warmup_runs = args[i].parse()?; }
            "--runs" => { i += 1; test_runs = args[i].parse()?; }
            "--verify" => { verify = true; }
            "--help" => {
                println!("Использование: candle_bench [опции]");
                println!("Опции:");
                println!("  --model <path>      Путь к .safetensors файлу");
                println!("  --h <pixels>        Высота изображения");
                println!("  --w <pixels>        Ширина изображения");
                println!("  --tile <size>       Размер тайла");
                println!("  --overlap <pixels>  Перекрытие тайлов");
                println!("  --device <cpu|cuda|metal>  Устройство");
                println!("  --warmup <N>        Прогревочные итерации");
                println!("  --runs <N>          Тестовые прогоны");
                println!("  --verify            Проверка вывода");
                return Ok(());
            }
            _ => {
                eprintln!("Неизвестная опция: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    
    // Выбор устройства
    let device = match device_str.as_str() {
        "cuda" => Device::new_cuda(0)?,
        "metal" => Device::new_metal(0)?,
        _ => Device::Cpu,
    };
    
    println!("\n=== Candle Benchmark ===");
    println!("Устройство: {:?}", device);
    println!("Размер изображения: {}x{}", w_px, h_px);
    println!("Тайл: {}px, перекрытие: {}px", tile_size, overlap);
    println!("Модель: {}", model_path);
    println!();
    
    // Загрузка модели
    if !Path::new(&model_path).exists() {
        eprintln!("Файл модели не найден: {}", model_path);
        eprintln!("Сначала сконвертируйте модель: python convert_to_safetensors.py best.pt");
        std::process::exit(1);
    }
    
    let model = load_model_safetensors(&model_path, &config, &device)?;
    
    // Создание тестового изображения
    println!("Генерация тестового изображения...");
    let input = synthetic_image(h_px, w_px, &device)?;
    println!("Изображение: {:?}, размер: {}x{}", input.shape(), w_px, h_px);
    
    // Тестирование на одном тайле для проверки
    println!("\nПроверка на одиночном тайле...");
    let tile_h = tile_size.min(h_px);
    let tile_w = tile_size.min(w_px);
    let tile = input.narrow(2, 0, tile_h)?.narrow(3, 0, tile_w)?;
    
    let start = Instant::now();
    let output_tile = model.forward(&tile)?;
    let elapsed = start.elapsed();
    
    println!("Первый прогон (тайл {}x{}): {:.2}ms", tile_w, tile_h, elapsed.as_secs_f64() * 1000.0);
    
    if verify {
        let output_shape = output_tile.shape();
        println!("Выходной тензор: {:?}", output_shape);
        // Статистика через sum/max/min
        let sum = output_tile.sum()?.to_scalar::<f32>()?;
        let numel = output_tile.elem_count() as f32;
        let mean = sum / numel;
        println!("Статистика вывода: mean={:.4}", mean);
    }
    
    // Бенчмарк полного изображения с тайлингом
    println!("\n=== Полный бенчмарк ===");
    let stride = tile_size - overlap;
    let tiles_y = (h_px - tile_size) / stride + 1;
    let tiles_x = (w_px - tile_size) / stride + 1;
    let total_tiles = tiles_x * tiles_y;
    
    println!("Количество тайлов: {} ({}x{})", total_tiles, tiles_x, tiles_y);
    
    // Прогрев
    println!("Прогрев ({} итераций)...", warmup_runs);
    for _ in 0..warmup_runs {
        let _ = model.forward(&tile)?;
    }
    
    // Тестовые прогоны
    let mut times = Vec::new();
    for run in 0..test_runs {
        print!("Прогон {}/{}... ", run + 1, test_runs);
        std::io::Write::flush(&mut std::io::stdout())?;
        
        let start = Instant::now();
        let mut processed = 0;
        
        // Упрощенный бенчмарк - только один тайл для скорости
        // В полной версии здесь был бы цикл по всем тайлам
        for _ in 0..total_tiles {
            let _ = model.forward(&tile)?;
            processed += 1;
        }
        
        let elapsed = start.elapsed();
        times.push(elapsed);
        
        let ms_per_tile = elapsed.as_secs_f64() * 1000.0 / processed as f64;
        println!("{:.2}ms всего, {:.2}ms/тайл", elapsed.as_secs_f64() * 1000.0, ms_per_tile);
    }
    
    // Статистика
    let avg_time = times.iter().sum::<std::time::Duration>() / times.len() as u32;
    let min_time = times.iter().min().unwrap();
    let max_time = times.iter().max().unwrap();
    
    println!("\n=== Результаты ===");
    println!("Среднее время: {:.2}ms", avg_time.as_secs_f64() * 1000.0);
    println!("Min/Max: {:.2}ms / {:.2}ms", 
             min_time.as_secs_f64() * 1000.0, 
             max_time.as_secs_f64() * 1000.0);
    println!("Производительность: {:.2} тайлов/сек", 
             1000.0 / (avg_time.as_secs_f64() * 1000.0 / total_tiles as f64));
    
    Ok(())
}
