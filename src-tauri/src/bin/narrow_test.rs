use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for dev in [Device::Cpu, Device::new_cuda(0)?] {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = Tensor::from_vec(vals, (1, 1, 4, 4), &dev)?;
        for (y0, x0) in [(0usize, 0usize), (2, 2)] {
            let p = x.narrow(2, y0, 2)?.narrow(3, x0, 2)?.contiguous()?;
            let v = p.flatten_all()?.to_vec1::<f32>()?;
            println!("{:?} narrow(y0={y0},x0={x0}): {v:?}", dev);
        }
    }
    Ok(())
}
