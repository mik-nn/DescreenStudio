//! Candle-движок продуктового инференса: safetensors (run27_hyb + FiLM)
//! на candle-core (CUDA с fallback на CPU), тайлинг 1:1 с
//! [`crate::inference::InferenceEngine::descreen_progress`] (replicate-pad,
//! Hann overlap-add, f32-аккумулятор) — чтобы смена движка не меняла
//! геометрию швов.
//!
//! Отличия от ort-ветки:
//! - батч тайлов за forward (`DESCREEN_BATCH`, по умолчанию 4);
//! - cond [1,4] из замера A0 подаётся в FiLM-ветку модели.

use candle_core::{DType, Device, Tensor};
use ndarray::{Array3, Array4, s};

use crate::candle_model::FourierUNet;
use crate::inference::{TileParams, accumulate, array2_zeros, pad_replicate_edges};

/// Нормировка r для cond-вектора (ml/model/film.py, COND_R_MAX).
pub const COND_R_MAX: f32 = 0.65;

/// Вектор кондиционирования из замера A0 (порт `ml.model.film.build_cond_vec`):
/// `[r/COND_R_MAX clamp 0..1.5, conf clamp 0..1, cos(ang), sin(ang)]`.
pub fn build_cond_vec(r: f32, angle_deg: f32, conf: f32) -> [f32; 4] {
    let t = angle_deg.to_radians();
    [
        (r.max(0.0) / COND_R_MAX).min(1.5),
        conf.clamp(0.0, 1.0),
        t.cos(),
        t.sin(),
    ]
}

pub struct CandleEngine {
    device: Device,
    model: FourierUNet,
    cond: Option<[f32; 4]>,
    batch: usize,
}

impl CandleEngine {
    /// `DESCREEN_DEVICE=cpu` — форс CPU; иначе CUDA с fallback на CPU.
    /// `DESCREEN_BATCH` — тайлов за forward (default 4, min 1).
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let device = match std::env::var("DESCREEN_DEVICE").as_deref() {
            Ok("cpu") => Device::Cpu,
            _ => Device::new_cuda(0).unwrap_or_else(|e| {
                eprintln!("candle: CUDA unavailable ({e}), falling back to CPU");
                Device::Cpu
            }),
        };
        let model = FourierUNet::load(&device, DType::F32, model_path)
            .map_err(|e| format!("candle model load failed: {e}"))?;
        let batch = std::env::var("DESCREEN_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|b| *b >= 1)
            .unwrap_or(4);
        eprintln!(
            "[stage] candle: device={device:?} batch={batch} film={} model={model_path}",
            model.has_film()
        );
        Ok(Self { device, model, cond: None, batch })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Вектор A0-замера для FiLM (None -> модуляция не применяется).
    pub fn set_cond(&mut self, cond: Option<[f32; 4]>) {
        self.cond = cond;
    }

    /// Полный прогон: зеркалит ort-тайлинг; батчит чанки тайлов в один forward.
    pub fn descreen_progress(
        &mut self,
        img: &Array3<f32>,
        tp: TileParams,
        on_tile: &mut dyn FnMut(usize, usize),
    ) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
        let tile = tp.tile.max(64);
        let stride = tile.saturating_sub(tp.overlap);
        assert!(stride >= 2, "overlap must be < tile");

        let ty = ((h as isize - tile as isize) as f64 / stride as f64).ceil().max(0.0) as usize;
        let tx = ((w as isize - tile as isize) as f64 / stride as f64).ceil().max(0.0) as usize;
        let ph = ty * stride + tile;
        let pw = tx * stride + tile;
        let mut padded = Array4::<f32>::zeros((1, c, ph, pw));
        {
            let mut dst = padded.slice_mut(s![0, .., 0..h, 0..w]);
            dst.assign(img);
        }
        pad_replicate_edges(&mut padded, h, w);

        let (mut acc, mut weight) = (Array3::<f32>::zeros((c, ph, pw)), array2_zeros(ph, pw));

        let tiles_x = (pw - tile) / stride + 1;
        let tiles_y = (ph - tile) / stride + 1;
        let total = tiles_x * tiles_y;
        let coords: Vec<(usize, usize)> = (0..tiles_y)
            .flat_map(|ty_i| (0..tiles_x).map(move |tx_i| (ty_i * stride, tx_i * stride)))
            .collect();

        let mut done = 0usize;
        for chunk in coords.chunks(self.batch) {
            let n = chunk.len();
            let mut data = vec![0f32; n * 3 * tile * tile];
            for (i, &(y0, x0)) in chunk.iter().enumerate() {
                let base = i * 3 * tile * tile;
                for ch in 0..3 {
                    for y in 0..tile {
                        for x in 0..tile {
                            data[base + ch * tile * tile + y * tile + x] =
                                padded[(0, ch, y0 + y, x0 + x)];
                        }
                    }
                }
            }
            let batch_in = Tensor::from_vec(data, (n, 3, tile, tile), &self.device)?;
            let cond = match self.cond {
                Some(c) => Some(Tensor::from_vec(c.to_vec(), (1, 4), &self.device)?),
                None => None,
            };
            let outs = self
                .model
                .forward_cond(&batch_in, cond.as_ref())?
                .to_dtype(DType::F32)?
                .contiguous()?;
            for (i, &(y0, x0)) in chunk.iter().enumerate() {
                let flat = outs
                    .narrow(0, i, 1)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let out3 = Array3::from_shape_vec((3, tile, tile), flat)?;
                accumulate(&mut acc, &out3.view(), &mut weight, y0, x0, tile);
            }
            done += n;
            on_tile(done, total);
        }

        let mut result = Array3::<f32>::zeros((c, h, w));
        for i in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let wt = weight[[0, y, x]];
                    result[[i, y, x]] = (acc[[i, y, x]] / wt.max(f32::EPSILON)).clamp(0.0, 1.0);
                }
            }
        }
        Ok(result)
    }
}
