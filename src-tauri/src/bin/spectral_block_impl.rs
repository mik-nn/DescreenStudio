// Сложный блок (encs.1, bottleneck, decs.0) — аналог PyTorch FourierSpectralBlock:
// rfft2 -> complex 1x1 convs (Conv-GELU-Conv) по real/imag -> ifft2 (real part).
// candle-core не имеет FFT, поэтому FFT через DFT-матрицы (матmul по H и W).
// Веса: "conv_real.0/2.weight|bias", "conv_imag.0/2.weight|bias" (1x1).
struct SpectralBlockComplex {
    conv_real_0: candle_nn::Conv2d,
    conv_real_2: candle_nn::Conv2d,
    conv_imag_0: candle_nn::Conv2d,
    conv_imag_2: candle_nn::Conv2d,
    fft: std::sync::Mutex<Option<Fft2>>,
}

/// DFT-матрицы для fft2/ifft2 (ортонормированный масштаб, как norm="ortho" в PyTorch).
/// Хранятся 2D [n, n]; матmul через reshape/permute, т.к. candle matmul
/// не бродкастит batch-размеры.
struct Fft2 {
    fr_h: Tensor, fi_h: Tensor, // [h, h], e^{-2pi i jk/n}
    fr_w: Tensor, fi_w: Tensor, // [w, w]
    gr_h: Tensor, gi_h: Tensor, // conj(F)/n
    gr_w: Tensor, gi_w: Tensor,
    p_h: Tensor, // [h, h] перестановка строк: ky = (h - iy) % h
    p_m: Tensor, // [l, l] анти-диагональ, l = w/2 - 1 (зеркало внутри src)
    m: usize,    // w/2 + 1
}

impl Fft2 {
    fn new(device: &Device, h: usize, w: usize) -> Result<Self, candle_core::Error> {
        assert!(w % 2 == 0, "Hermitian-зеркало реализовано только для чётного W");
        let dft = |n: usize| -> Result<(Tensor, Tensor), candle_core::Error> {
            let idx = Tensor::arange(0, n as u32, device)?.to_dtype(DType::F32)?;
            let ang = idx
                .reshape((n, 1))?
                .broadcast_mul(&idx.reshape((1, n))?)?
                .affine(-2.0 * std::f64::consts::PI / n as f64, 0.0)?;
            Ok((ang.cos()?, ang.sin()?))
        };
        let onehot = |n: usize, f: &dyn Fn(usize) -> usize| -> Result<Tensor, candle_core::Error> {
            let mut data = vec![0f32; n * n];
            for i in 0..n {
                data[i * n + f(i)] = 1.0;
            }
            Tensor::from_vec(data, (n, n), device)
        };
        // p_h[iy, ky] = 1 при ky = (h - iy) % h  (строковое зеркало, DC-строка сама себе)
        let p_h = onehot(h, &|i| (h - i) % h)?;
        // p_m[i, j] = 1 при j = l - 1 - i  (анти-диагональ, зеркало частот внутри src)
        let m = w / 2 + 1;
        let l = m - 2;
        let p_m = onehot(l, &|i| l - 1 - i)?;

        let (fr_h, fi_h) = dft(h)?;
        let (fr_w, fi_w) = dft(w)?;
        let scale_h = 1.0 / (h as f64).sqrt();
        let scale_w = 1.0 / (w as f64).sqrt();
        // norm="ortho": 1/sqrt(n) на каждую сторону и в прямом, и в обратном.
        Ok(Self {
            fr_h: fr_h.affine(scale_h, 0.0)?,
            fi_h: fi_h.affine(scale_h, 0.0)?,
            fr_w: fr_w.affine(scale_w, 0.0)?,
            fi_w: fi_w.affine(scale_w, 0.0)?,
            gr_h: fr_h.affine(scale_h, 0.0)?,
            gi_h: fi_h.affine(-scale_h, 0.0)?,
            gr_w: fr_w.affine(scale_w, 0.0)?,
            gi_w: fi_w.affine(-scale_w, 0.0)?,
            p_h, p_m, m,
        })
    }

    /// Умножение слева на [h, h]-матрицу для [1, c, h, l]: y[iy] = Σ_ky M[iy,ky] x[ky].
    fn left_m(&self, x: &Tensor, m_left: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, c, h, l) = x.dims4()?;
        let xh = x
            .reshape((c, h, l))?
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((h, c * l))?;
        let y = m_left.matmul(&xh)?; // [h, c*l]
        y.reshape((h, c, l))?
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((b, c, h, l))
    }

    /// Умножение справа на [l, l]-матрицу для [1, c, h, l].
    fn right_m(&self, x: &Tensor, m_right: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, c, h, l) = x.dims4()?;
        let xl = x
            .reshape((c, h, l))?
            .reshape((c * h, l))?
            .contiguous()?;
        let y = xl.matmul(m_right)?; // [c*h, l]
        y.reshape((b, c, h, l))
    }

    /// fft2: [1, c, h, w] (real) -> (re, im) [1, c, h, w]
    fn fft2(&self, x: &Tensor) -> Result<(Tensor, Tensor), candle_core::Error> {
        let (b, c, h, w) = x.dims4()?;
        // Сторона H: [c, h, w] -> [h, c*w], левый матmul F[h,h].
        let xh = x
            .reshape((c, h, w))?
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((h, c * w))?;
        let p = self.fr_h.matmul(&xh)?; // [h, c*w]
        let q = self.fi_h.matmul(&xh)?;
        // Сторона W: [h, c, w] -> [c*h, w], правый матmul F[w,w].
        let side = |m: &Tensor| -> Result<Tensor, candle_core::Error> {
            Ok(m
                .reshape((h, c, w))?
                .permute((1, 0, 2))?
                .contiguous()?
                .reshape((c * h, w))?)
        };
        let p = side(&p)?; // [c*h, w]
        let q = side(&q)?;
        let re = p
            .matmul(&self.fr_w)?
            .sub(&q.matmul(&self.fi_w)?)?
            .reshape((c, h, w))?
            .reshape((b, c, h, w))?;
        let im = p
            .matmul(&self.fi_w)?
            .add(&q.matmul(&self.fr_w)?)?
            .reshape((c, h, w))?
            .reshape((b, c, h, w))?;
        Ok((re, im))
    }

    /// ifft2 -> действительная часть [1, c, h, w]
    fn ifft2_re(&self, re: &Tensor, im: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (b, c, h, w) = re.dims4()?;
        let rh = re
            .reshape((c, h, w))?
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((h, c * w))?;
        let ih = im
            .reshape((c, h, w))?
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((h, c * w))?;
        let p = self.gr_h.matmul(&rh)?.sub(&self.gi_h.matmul(&ih)?)?; // [h, c*w]
        let q = self.gi_h.matmul(&rh)?.add(&self.gr_h.matmul(&ih)?)?;
        let side = |m: &Tensor| -> Result<Tensor, candle_core::Error> {
            Ok(m
                .reshape((h, c, w))?
                .permute((1, 0, 2))?
                .contiguous()?
                .reshape((c * h, w))?)
        };
        let p = side(&p)?;
        let q = side(&q)?;
        // Действительная часть: (P - iQ)(Gr + iGw) -> re = P Gr - Q Gi
        p.matmul(&self.gr_w)?
            .sub(&q.matmul(&self.gi_w)?)?
            .reshape((c, h, w))?
            .reshape((b, c, h, w))
    }
}

impl SpectralBlockComplex {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self, candle_core::Error> {
        let cfg_1x1 = Conv2dConfig { padding: 0, ..Default::default() };

        let pp_real = vb.pp("conv_real");
        let pp_imag = vb.pp("conv_imag");

        let conv_real_0 = conv2d(channels, channels, 1, cfg_1x1.clone(), pp_real.pp("0"))?;
        let conv_real_2 = conv2d(channels, channels, 1, cfg_1x1.clone(), pp_real.pp("2"))?;

        let conv_imag_0 = conv2d(channels, channels, 1, cfg_1x1.clone(), pp_imag.pp("0"))?;
        let conv_imag_2 = conv2d(channels, channels, 1, cfg_1x1, pp_imag.pp("2"))?;

        Ok(Self {
            conv_real_0, conv_real_2,
            conv_imag_0, conv_imag_2,
            fft: std::sync::Mutex::new(None),
        })
    }

    fn fft_for(&self, device: &Device, h: usize, w: usize) -> Result<(), candle_core::Error> {
        let mut guard = self.fft.lock().map_err(|e| {
            candle_core::Error::Msg(format!("fft mutex poisoned: {e}"))
        })?;
        if guard.is_none() {
            *guard = Some(Fft2::new(device, h, w)?);
        }
        Ok(())
    }
}

impl Module for SpectralBlockComplex {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (_, _, h, w) = xs.dims4()?;
        self.fft_for(xs.device(), h, w)?;
        let fft = self.fft.lock().map_err(|e| {
            candle_core::Error::Msg(format!("fft mutex poisoned: {e}"))
        })?;
        let fft = fft.as_ref().unwrap();
        let m = fft.m;

        // FFT2(real x) -> onesided-половина спектра [.., m] -> complex 1x1 conv
        // (Conv-GELU-Conv) -> Hermitian-зеркало (как в PyTorch ffblock.py) ->
        // IFFT2, действительная часть.
        let (re, im) = fft.fft2(xs)?;
        let re = re.narrow(3, 0, m)?;
        let im = im.narrow(3, 0, m)?;

        // Complex 1x1 convs: каждая ветка применяется И к re, И к im.
        // out_re = conv_real(re) - conv_imag(im); out_im = conv_real(im) + conv_imag(re)
        let a_re = self.conv_real_0.forward(&re)?.gelu_erf()?;
        let a_im = self.conv_real_0.forward(&im)?.gelu_erf()?;
        let b_re = self.conv_imag_0.forward(&re)?.gelu_erf()?;
        let b_im = self.conv_imag_0.forward(&im)?.gelu_erf()?;
        let out_re = self.conv_real_2.forward(&a_re)?.sub(&self.conv_imag_2.forward(&b_im)?)?;
        let out_im = self.conv_real_2.forward(&a_im)?.add(&self.conv_imag_2.forward(&b_re)?)?;

        // Зеркало: src = [.., 1..m-1]; mir_re = src[ih, iw], mir_im = -src_im[ih, iw],
        // ih = (h - iy) % h (строки), iw = обратный порядок частот (p_m).
        let l = m - 2;
        let src_re = out_re.narrow(3, 1, l)?;
        let src_im = out_im.narrow(3, 1, l)?;
        let mir_re = fft.right_m(&fft.left_m(&src_re, &fft.p_h)?, &fft.p_m)?;
        let mir_im = fft.right_m(&fft.left_m(&src_im, &fft.p_h)?, &fft.p_m)?.neg()?;

        let full_re = Tensor::cat(&[&out_re, &mir_re], 3)?;
        let full_im = Tensor::cat(&[&out_im, &mir_im], 3)?;
        fft.ifft2_re(&full_re, &full_im)
    }
}
