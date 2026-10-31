//! FourierUNet — candle-порт `ml/model/unet.py` (+ `ml/model/film.py`).
//!
//! Скелет зафиксирован run3-эпохой (base96 / L3 / bneck3 / spec L0=plain,
//! L1..=complex), ключи safetensors совпадают с torch state_dict 1:1.
//! Модели с `cond=true` (run9+, run27_hyb) несут `film.mlp.*` — FiLM-модуляция
//! bottleneck'а вектором замера A0 (см. `candle_engine::build_cond_vec`).

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Conv2dConfig, ConvTranspose2dConfig, Module, VarBuilder, conv2d, conv_transpose2d, linear,
};
use safetensors::SafeTensors;
use std::collections::HashMap;

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

include!("inc/spectral_block_impl.rs");

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

// --- FiLM-кондиционирование (ml/model/film.py) ---

/// MLP cond [B, 4] -> (gamma, beta) [B, 2*channels]; последний слой нулевой
/// на старте (zero-init) — без cond модель тождественна run7-поведению.
struct FiLM {
    mlp0: candle_nn::Linear,
    mlp2: candle_nn::Linear,
    channels: usize,
}

impl FiLM {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self, candle_core::Error> {
        // Ключи: film.mlp.0.{weight,bias}, film.mlp.2.{weight,bias}
        let mlp0 = linear(4, 32, vb.pp("mlp").pp(0))?;
        let mlp2 = linear(32, 2 * channels, vb.pp("mlp").pp(2))?;
        Ok(Self { mlp0, mlp2, channels })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor, candle_core::Error> {
        // torch: gb = self.mlp(cond.to(x.dtype)); gamma, beta = gb.chunk(2, -1)
        let cond = cond.to_dtype(x.dtype())?;
        let gb = self.mlp0.forward(&cond)?.gelu_erf()?;
        let gb = self.mlp2.forward(&gb)?;
        let gamma = gb.narrow(1, 0, self.channels)?;
        let beta = gb.narrow(1, self.channels, self.channels)?;
        let g = gamma
            .reshape((gamma.dim(0)?, self.channels, 1, 1))?
            .broadcast_as(x.dims4()?)?;
        let b = beta
            .reshape((beta.dim(0)?, self.channels, 1, 1))?
            .broadcast_as(x.dims4()?)?;
        // x * (1 + gamma) + beta
        x.broadcast_mul(&g.affine(1.0, 1.0)?)?.broadcast_add(&b)
    }
}

// --- Основная модель ---

pub struct FourierUNet {
    inc: candle_nn::Conv2d,
    enc0: FFCBlock,
    down0: candle_nn::Conv2d,
    enc1: FFCBlock,
    down1: candle_nn::Conv2d,

    b0: FFCBlock,
    b1: FFCBlock,
    b2: FFCBlock,

    film: Option<FiLM>,

    up0: candle_nn::ConvTranspose2d,
    dec0: FFCBlock,
    up1: candle_nn::ConvTranspose2d,
    dec1: FFCBlock,
    outc: candle_nn::Conv2d,
}

impl FourierUNet {
    /// Загрузка из safetensors. Наличие `film.mlp.*` включает FiLM-ветку
    /// (torch-семантика: cond=None -> модуляция не применяется).
    pub fn load(device: &Device, dtype: DType, model_path: &str) -> Result<Self, candle_core::Error> {
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
        let has_film = tensors.contains_key("film.mlp.0.weight");

        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        // 1. Input: 3 -> 96 (inc.weight)
        let inc = conv2d(3, 96, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb.pp("inc"))?;

        // 2. Encoder 0: 96 -> 96 (Simple: encs.0.0.spectral.weight)
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

        // 6b. FiLM после bottleneck, до декодера (unet.py:94-96)
        let film = if has_film { Some(FiLM::new(vb.pp("film"), 384)?) } else { None };

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

        Ok(Self { inc, enc0, down0, enc1, down1, b0, b1, b2, film, up0, dec0, up1, dec1, outc })
    }

    pub fn has_film(&self) -> bool {
        self.film.is_some()
    }

    /// Forward с опциональным FiLM-cond [B, 4] (или [1, 4] — бродкаст на батч).
    /// cond=None при наличии film -> модуляция не применяется (как в torch).
    pub fn forward_cond(
        &self,
        xs: &Tensor,
        cond: Option<&Tensor>,
    ) -> Result<Tensor, candle_core::Error> {
        // Зеркалит ml/model/unet.py: GELU после inc/down/up, skip-connections
        // (feats — выходы энкодеров до down), выход = clamp(x - residual, 0, 1).
        let dump_dir = std::env::var("CANDLE_DUMP_DIR").ok();
        let dump = dump_dir.is_some() || std::env::var("CANDLE_DUMP").is_ok();
        let stat = |t: &Tensor, tag: &str| {
            if dump {
                if let Ok(v) = t.flatten_all().and_then(|f| f.to_vec1::<f32>()) {
                    let m = v.iter().sum::<f32>() / v.len() as f32;
                    let var = v.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / v.len() as f32;
                    eprintln!("stage {tag}: mean={m:.5} std={:.5} first4={:?}", var.sqrt(), &v[..4]);
                    if let Some(dir) = &dump_dir {
                        let dims = t.dims().to_vec();
                        let _ = std::fs::write(
                            format!("{dir}/rust_{tag}.dims"),
                            dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" "),
                        );
                        let mut bytes = Vec::with_capacity(v.len() * 4);
                        for a in &v {
                            bytes.extend_from_slice(&a.to_le_bytes());
                        }
                        let _ = std::fs::write(format!("{dir}/rust_{tag}.bin"), bytes);
                    }
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

        if let (Some(film), Some(c)) = (&self.film, cond) {
            x = film.forward(&x, c)?;
            stat(&x, "film");
        }

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

impl Module for FourierUNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        self.forward_cond(xs, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det_input(seed: u64) -> Vec<f32> {
        let (c, h, w) = (3usize, 64usize, 64usize);
        let mut s = seed;
        (0..c * h * w)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as u32 & 0xFFFFFF) as f32 / 16_777_215.0
            })
            .collect()
    }

    /// Постадийный паритет с torch: читает `Temp/parity/in.bin` (f32, 3xHxW),
    /// гоняет модель и пишет стадии в `Temp/parity/rust_<stage>.bin`
    /// (см. `Temp/parity_torch.py`). Пропускается, если входа нет.
    #[test]
    fn parity_stages_dump() -> Result<(), candle_core::Error> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let dir = std::env::var("PARITY_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| root.join("Temp").join("parity"));
        let inp = std::env::var("PARITY_IN")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| dir.join("in.bin"));
        if !inp.exists() {
            eprintln!("skip: {} not found", inp.display());
            return Ok(());
        }
        let model_path = match std::env::var("PARITY_MODEL") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => root
                .join("ml/train/_chain_native_band/descreen_chain_native_band.safetensors"),
        };
        let dev = Device::Cpu;
        let model = FourierUNet::load(&dev, DType::F32, model_path.to_str().unwrap())?;

        let raw = std::fs::read(&inp).unwrap();
        let n = raw.len() / 4;
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let side = ((n / 3) as f64).sqrt() as usize;
        let xs = Tensor::from_vec(data, (1, 3, side, side), &dev)?;
        let cond = Tensor::from_vec(vec![0.2f32, 1.0, -0.7, 0.7], (1, 4), &dev)?;

        std::env::set_var("CANDLE_DUMP_DIR", &dir);
        std::fs::create_dir_all(&dir).unwrap();
        let out = model.forward_cond(&xs, Some(&cond))?;
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for a in &v {
            bytes.extend_from_slice(&a.to_le_bytes());
        }
        std::fs::write(dir.join("rust_out.bin"), &bytes).unwrap();
        eprintln!("parity dump -> {}", dir.display());
        Ok(())
    }

    #[test]
    fn batch_matches_single_and_cond_broadcasts() -> Result<(), candle_core::Error> {
        let Ok(path) = crate::runtime::resolve_safetensors(None) else {
            eprintln!("skip: descreen.safetensors not found");
            return Ok(());
        };
        let dev = Device::Cpu;
        let model =
            FourierUNet::load(&dev, DType::F32, path.to_str().unwrap()).expect("load model");
        assert!(model.has_film(), "run27_hyb must carry FiLM");
        let mk = |seed: u64| Tensor::from_vec(det_input(seed), (1, 3, 64, 64), &dev).unwrap();
        let x1 = mk(1);
        let x2 = mk(2);
        let batch = Tensor::cat(&[&x1, &x2], 0).unwrap();
        let cond = Tensor::from_vec(vec![0.5f32, 0.8, 0.9, 0.4], (1, 4), &dev).unwrap();

        let o1 = model.forward_cond(&x1, Some(&cond)).unwrap();
        let o2 = model.forward_cond(&x2, Some(&cond)).unwrap();
        let ob = model.forward_cond(&batch, Some(&cond)).unwrap();
        for (i, single) in [&o1, &o2].iter().enumerate() {
            let d = ob
                .narrow(0, i, 1)?
                .sub(*single)?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let m = d.iter().cloned().fold(0f32, f32::max);
            assert!(m < 1e-4, "batch sample {i}: max|diff|={m}");
        }

        // cond обязан влиять на выход (обученный FiLM), а не игнорироваться
        let o_nc = model.forward_cond(&x1, None)?;
        let d = o_nc.sub(&o1)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
        let m = d.iter().cloned().fold(0f32, f32::max);
        assert!(m > 1e-6, "FiLM cond must affect output, max|diff|={m}");
        Ok(())
    }
}
