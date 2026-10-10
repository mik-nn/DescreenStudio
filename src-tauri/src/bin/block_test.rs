use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d, Conv2dConfig, Module, VarBuilder};

include!("../inc/spectral_block_impl.rs");

fn read_safetensors(path: &str) -> Result<std::collections::HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let buf = std::fs::read(path)?;
    let hlen = u64::from_le_bytes(buf[0..8].try_into()?) as usize;
    let header: serde_json::Value = serde_json::from_slice(&buf[8..8 + hlen])?;
    let base = 8 + hlen;
    let mut map = std::collections::HashMap::new();
    for (name, meta) in header.as_object().unwrap() {
        if name == "__metadata__" { continue; }
        let start = meta["data_offsets"][0].as_u64().unwrap() as usize + base;
        let end = meta["data_offsets"][1].as_u64().unwrap() as usize + base;
        let shape: Vec<usize> = meta["shape"].as_array().unwrap().iter()
            .map(|v| v.as_u64().unwrap() as usize).collect();
        let n: usize = shape.iter().product();
        let mut data = vec![0f32; n];
        for i in 0..n {
            data[i] = f32::from_le_bytes(buf[start + 4 * i..start + 4 * i + 4].try_into()?);
        }
        map.insert(name.clone(), Tensor::from_vec(data, shape, &Device::Cpu)?);
    }
    let _ = DType::F32;
    Ok(map)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::Cpu;
    let st = read_safetensors(concat!(env!("CARGO_MANIFEST_DIR"), "/../ml/train/_run3/best.safetensors"))?;
    let prefix = "bottleneck.0";
    let get = |k: &str| st.get(k).cloned().unwrap();

    let cfg1 = Conv2dConfig { padding: 0, ..Default::default() };
    let cfg3 = Conv2dConfig { padding: 1, ..Default::default() };
    let mut hm = std::collections::HashMap::new();
    for k in ["conv_real.0.weight", "conv_real.0.bias", "conv_real.2.weight", "conv_real.2.bias",
              "conv_imag.0.weight", "conv_imag.0.bias", "conv_imag.2.weight", "conv_imag.2.bias"] {
        hm.insert(format!("spectral.{k}"), get(&format!("{prefix}.spectral.{k}")));
    }
    let spectral = SpectralBlockComplex::new(
        VarBuilder::from_tensors(hm, DType::F32, &dev).pp("spectral"), 384)?;

    let mut hm2 = std::collections::HashMap::new();
    hm2.insert("weight".to_string(), get(&format!("{prefix}.spatial.weight")));
    hm2.insert("bias".to_string(), get(&format!("{prefix}.spatial.bias")));
    let spatial = conv2d(384, 384, 3, cfg3, VarBuilder::from_tensors(hm2, DType::F32, &dev))?;

    let mut hm3 = std::collections::HashMap::new();
    hm3.insert("0.weight".to_string(), get(&format!("{prefix}.fuse.0.weight")));
    hm3.insert("0.bias".to_string(), get(&format!("{prefix}.fuse.0.bias")));
    let fuse = conv2d(768, 384, 1, cfg1, VarBuilder::from_tensors(hm3, DType::F32, &dev).pp("0"))?;

    // Вход: детерминированный rand 384x16x16
    let mut seed = 0x12345678u64;
    let mut vals = vec![0f32; 384 * 16 * 16];
    for v in vals.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    let x = Tensor::from_vec(vals, (1, 384, 16, 16), &dev)?;

    // Дамп out_re/out_im (после complex convs, до зеркала), канал 0
    {
        spectral.fft_for(&dev, 16, 16)?;
        let fft = spectral.fft.lock().unwrap();
        let fft = fft.as_ref().unwrap();
        let m = fft.m;
        let (re, im) = fft.fft2(&x)?;
        let re = re.narrow(3, 0, m)?;
        let im = im.narrow(3, 0, m)?;
        use candle_nn::Module;
        let a = spectral.conv_real_0.forward(&re)?.gelu_erf()?;
        let b = spectral.conv_imag_0.forward(&im)?.gelu_erf()?;
        let out_re = spectral.conv_real_2.forward(&a)?.sub(&spectral.conv_imag_2.forward(&b)?)?;
        let out_im = spectral.conv_real_2.forward(&b)?.add(&spectral.conv_imag_2.forward(&a)?)?;
        let orv = out_re.flatten_all()?.to_vec1::<f32>()?;
        let oiv = out_im.flatten_all()?.to_vec1::<f32>()?;
        println!("out_re mean={:.6} [:16]={:?}", orv.iter().sum::<f32>()/orv.len() as f32, &orv[..16]);
        println!("out_im mean={:.6} [:16]={:?}", oiv.iter().sum::<f32>()/oiv.len() as f32, &oiv[..16]);
    }
    let f = spectral.forward(&x)?;
    // Проверка fft2 против эталона: onesided re/im канала 0
    {
        let fft = Fft2::new(&dev, 16, 16)?;
        let (re, im) = fft.fft2(&x)?;
        let m = fft.m;
        let re = re.narrow(3, 0, m)?.flatten_all()?.to_vec1::<f32>()?;
        let im = im.narrow(3, 0, m)?.flatten_all()?.to_vec1::<f32>()?;
        println!("fft2 re[0][:16]={:?}", &re[..16]);
        println!("fft2 im[0][:16]={:?}", &im[..16]);
    }
    {
        let fv = f.flatten_all()?.to_vec1::<f32>()?;
        let fm = fv.iter().sum::<f32>() / fv.len() as f32;
        let fvar = fv.iter().map(|a| (a - fm) * (a - fm)).sum::<f32>() / fv.len() as f32;
        println!("spectral f: mean={fm:.6} std={:.6} first16={:?}", fvar.sqrt(), &fv[..16]);
    }
    let s = spatial.forward(&x)?;
    {
        let sv = s.flatten_all()?.to_vec1::<f32>()?;
        let sm = sv.iter().sum::<f32>() / sv.len() as f32;
        println!("spatial s: mean={sm:.6} first={:?}", &sv[..4]);
    }
    let cat = Tensor::cat(&[&s, &f], 1)?;
    let out = fuse.forward(&cat)?.gelu_erf()?;
    let out = (&out + &x)?;
    let v = out.flatten_all()?.to_vec1::<f32>()?;
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / v.len() as f32;
    println!("FFCBlock(bottleneck.0) out: mean={mean:.6} std={:.6} first16={:?}", var.sqrt(), &v[..16]);
    Ok(())
}
