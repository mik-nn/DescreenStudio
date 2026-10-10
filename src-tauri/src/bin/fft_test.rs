use candle_core::{DType, Device, Tensor};

// Копия Fft2 из candle_bench.rs для изолированной проверки.
struct Fft2 {
    fr_h: Tensor, fi_h: Tensor,
    fr_w: Tensor, fi_w: Tensor,
    gr_h: Tensor, gi_h: Tensor,
    gr_w: Tensor, gi_w: Tensor,
}

impl Fft2 {
    fn new(device: &Device, h: usize, w: usize) -> Result<Self, candle_core::Error> {
        let dft = |n: usize| -> Result<(Tensor, Tensor), candle_core::Error> {
            let idx = Tensor::arange(0, n as u32, device)?.to_dtype(DType::F32)?;
            let ang = idx
                .reshape((n, 1))?
                .broadcast_mul(&idx.reshape((1, n))?)?
                .affine(-2.0 * std::f64::consts::PI / n as f64, 0.0)?;
            Ok((ang.cos()?, ang.sin()?))
        };
        let (fr_h, fi_h) = dft(h)?;
        let (fr_w, fi_w) = dft(w)?;
        let scale_h = 1.0 / (h as f64).sqrt();
        let scale_w = 1.0 / (w as f64).sqrt();
        Ok(Self {
            fr_h: fr_h.affine(scale_h, 0.0)?,
            fi_h: fi_h.affine(scale_h, 0.0)?,
            fr_w: fr_w.affine(scale_w, 0.0)?,
            fi_w: fi_w.affine(scale_w, 0.0)?,
            gr_h: fr_h.affine(scale_h, 0.0)?,
            gi_h: fi_h.affine(-scale_h, 0.0)?,
            gr_w: fr_w.affine(scale_w, 0.0)?,
            gi_w: fi_w.affine(-scale_w, 0.0)?,
        })
    }

    fn fft2(&self, x: &Tensor) -> Result<(Tensor, Tensor), candle_core::Error> {
        let (b, c, h, w) = x.dims4()?;
        let xh = x.reshape((c, h, w))?.permute((1, 0, 2))?.contiguous()?.reshape((h, c * w))?;
        let p = self.fr_h.matmul(&xh)?;
        let q = self.fi_h.matmul(&xh)?;
        let side = |m: &Tensor| -> Result<Tensor, candle_core::Error> {
            Ok(m.reshape((h, c, w))?.permute((1, 0, 2))?.contiguous()?.reshape((c * h, w))?)
        };
        let p = side(&p)?;
        let q = side(&q)?;
        let re = p.matmul(&self.fr_w)?.sub(&q.matmul(&self.fi_w)?)?.reshape((b, c, h, w))?;
        let im = p.matmul(&self.fi_w)?.add(&q.matmul(&self.fr_w)?)?.reshape((b, c, h, w))?;
        Ok((re, im))
    }

    fn ifft2_re(&self, re: &Tensor, im: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, c, h, w) = re.dims4()?;
        let rh = re.reshape((c, h, w))?.permute((1, 0, 2))?.contiguous()?.reshape((h, c * w))?;
        let ih = im.reshape((c, h, w))?.permute((1, 0, 2))?.contiguous()?.reshape((h, c * w))?;
        let p = self.gr_h.matmul(&rh)?.sub(&self.gi_h.matmul(&ih)?)?;
        let q = self.gi_h.matmul(&rh)?.add(&self.gr_h.matmul(&ih)?)?;
        let side = |m: &Tensor| -> Result<Tensor, candle_core::Error> {
            Ok(m.reshape((h, c, w))?.permute((1, 0, 2))?.contiguous()?.reshape((c * h, w))?)
        };
        let p = side(&p)?;
        let q = side(&q)?;
        p.matmul(&self.gr_w)?.sub(&q.matmul(&self.gi_w)?)?.reshape((b, c, h, w))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_cuda(0)?;
    let (h, w) = (8usize, 12usize);
    let fft = Fft2::new(&dev, h, w)?;
    // Псевдослучайный вход с фиксированным seed (простой LCG).
    let mut seed = 12345u64;
    let mut vals = vec![0f32; 3 * h * w];
    for v in vals.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    let x = Tensor::from_vec(vals, (1, 3, h, w), &dev)?;
    let (re, im) = fft.fft2(&x)?;
    let x2 = fft.ifft2_re(&re, &im)?;
    let a = x.flatten_all()?.to_vec1::<f32>()?;
    let b = x2.flatten_all()?.to_vec1::<f32>()?;
    let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("roundtrip max|diff| = {max_diff:e}");
    // Энергия Парсеваля: sum(x^2) == sum(|X|^2) при norm="ortho"
    let e_x: f32 = a.iter().map(|v| v * v).sum();
    let re_v = re.flatten_all()?.to_vec1::<f32>()?;
    let im_v = im.flatten_all()?.to_vec1::<f32>()?;
    let e_xf: f32 = re_v.iter().zip(im_v.iter()).map(|(r, i)| r * r + i * i).sum();
    println!("energy: x={e_x:.6}  X={e_xf:.6}");
    // DC-компонента: X[0,0] = sum(x)/sqrt(h*w) (действительная)
    let dc_re = re_v[0];
    let dc_im = im_v[0];
    let sum_x: f32 = a.iter().sum();
    println!("DC: re={dc_re:.4} im={dc_im:.4}  expected={:.4}", sum_x / ((h * w) as f32).sqrt());
    Ok(())
}
