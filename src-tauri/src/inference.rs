use std::error::Error;

use ndarray::{Array3, Array4, Ix4, s};
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ep {
    DirectML,
    Cpu,
}

#[derive(Clone, Copy, Debug)]
pub struct TileParams {
    pub tile: usize,
    pub overlap: usize,
}

impl Default for TileParams {
    fn default() -> Self {
        Self { tile: 512, overlap: 64 }
    }
}

pub struct InferenceEngine {
    session: Session,
    ep_requested: Ep,
}

impl InferenceEngine {
    pub fn new(model_path: &str, want: Ep) -> Result<Self, Box<dyn Error>> {
        let _ = ort::init().commit();
        let providers: Vec<ExecutionProviderDispatch> = match want {
            Ep::DirectML => vec![ep::DirectML::default().build()],
            Ep::Cpu => vec![],
        };
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .commit_from_file(model_path)?;
        Ok(Self { session, ep_requested: want })
    }

    pub fn ep(&self) -> Ep {
        self.ep_requested
    }

    /// Run one 4-D batch [1, C, H, W], channels-first, fp32 in [0, 1].
    pub fn process_tile(&mut self, tile: &Array4<f32>) -> Result<Array4<f32>, Box<dyn Error>> {
        let value = Tensor::from_array(tile.clone())?;
        let outputs = self.session.run(ort::inputs![value])?;
        let view = outputs[0].try_extract_array::<f32>()?;
        Ok(view.to_owned().into_dimensionality::<Ix4>()?)
    }

    /// Full-scan descreen: tiled inference with Hann overlap-add.
    /// Input: [C, H, W] fp32 in [0, 1]. Returns same shape (clipped to [0, 1]).
    pub fn descreen(&mut self, img: &Array3<f32>, tp: TileParams) -> Result<Array3<f32>, Box<dyn Error>> {
        let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
        let tile = tp.tile.max(64);
        let stride = tile.saturating_sub(tp.overlap);
        assert!(stride >= 2, "overlap must be < tile");

        // Replicate-pad so the tile grid ends exactly at the padded edge and
        // every real pixel is covered by an interior window.
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
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let y0 = ty * stride;
                let x0 = tx * stride;
                let patch = padded.slice(s![0..1, .., y0..y0 + tile, x0..x0 + tile]).to_owned();
                let out = self.process_tile(&patch)?;
                let out3 = out.index_axis(ndarray::Axis(0), 0);
                accumulate(&mut acc, &out3, &mut weight, y0, x0, tile);
            }
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

fn array2_zeros(h: usize, w: usize) -> Array3<f32> {
    Array3::<f32>::zeros((1, h, w))
}

/// Replicate edge rows/cols of padded [1, C, ph, pw] that were left zero so the
/// whole padded tensor is a replicate-border extension of [0..h, 0..w).
fn pad_replicate_edges(arr: &mut Array4<f32>, h: usize, w: usize) {
    let (c, ph, pw) = (arr.shape()[1], arr.shape()[2], arr.shape()[3]);
    let n = *arr.shape().first().unwrap();
    for b in 0..n {
        for ch in 0..c {
            for y in 0..ph {
                let sy = y.min(h - 1);
                for x in 0..pw {
                    if y >= h || x >= w {
                        let sx = x.min(w - 1);
                        arr[[b, ch, y, x]] = arr[[b, ch, sy, sx]];
                    }
                }
            }
        }
    }
}

/// Weighted accumulate of one output tile with its separable Hann window.
fn accumulate(acc: &mut Array3<f32>, out: &ndarray::ArrayView3<f32>, weight: &mut Array3<f32>, y0: usize, x0: usize, tile: usize) {
    // Hann window, 1-D, period length tile (matches v3.7 export path).
    let win1d: Vec<f32> = (0..tile)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / (tile as f32 - 1.0);
            0.5 * (1.0 - t.cos())
        })
        .collect();
    let c = out.shape()[0];
    for y in 0..tile {
        let wy = win1d[y];
        for x in 0..tile {
            let wv = wy * win1d[x];
            weight[[0, y0 + y, x0 + x]] += wv;
            for i in 0..c {
                acc[[i, y0 + y, x0 + x]] += out[[i, y, x]] * wv;
            }
        }
    }
}