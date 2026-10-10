use candle_core::{DType, Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_cuda(0)?;
    let (ph, pw, tile, stride) = (1408usize, 1408usize, 512usize, 448usize);
    // [1, 3, ph, pw] со значением = y*pw + x (одинаковое по каналам)
    let vals: Vec<f32> = (0..3 * ph * pw)
        .map(|i| {
            let j = i % (ph * pw); // одинаковое значение по всем каналам
            let y = j / pw;
            let x = j % pw;
            (y * pw + x) as f32
        })
        .collect();
    let padded = Tensor::from_vec(vals, (1, 3, ph, pw), &dev)?;
    let tiles = (ph - tile) / stride + 1;
    for ty in 0..tiles {
        for tx in 0..tiles {
            let y0 = ty * stride;
            let x0 = tx * stride;
            let patch = padded
                .narrow(2, y0, tile)?
                .narrow(3, x0, tile)?
                .contiguous()?;
            let v = patch.flatten_all()?.to_vec1::<f32>()?;
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let first = v[0];
            // ожидаемый mean: среднее (y*pw+x) по окну
            let mut exp = 0f64;
            for y in y0..y0 + tile {
                for x in x0..x0 + tile {
                    exp += (y * pw + x) as f64;
                }
            }
            exp /= (tile * tile) as f64;
            println!(
                "tile({ty},{tx}) y0={y0} x0={x0}: mean={mean:.1} expected={exp:.1} first={first:.0} expected_first={}",
                (y0 * pw + x0) as f32
            );
        }
    }
    let _ = DType::F32;
    Ok(())
}
