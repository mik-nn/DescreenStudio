use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d, conv_transpose2d, Conv2dConfig, ConvTranspose2dConfig, VarBuilder};
use candle_nn::Module;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_cuda(0)?;
    // conv2d: in [1,1,4,4]=0..15, w [0,1,2;-1,0,1;2,-2,0], b=0.5, pad 1
    let x = Tensor::arange(0f32, 16f32, &dev)?
        .to_dtype(DType::F32)?
        .reshape((1, 1, 4, 4))?;
    let w = Tensor::from_vec(
        vec![0f32, 1., 2., -1., 0., 1., 2., -2., 0.],
        (1, 1, 3, 3),
        &dev,
    )?;
    let b = Tensor::from_vec(vec![0.5f32], (1,), &dev)?;
    let mut hm = std::collections::HashMap::new();
    hm.insert("weight".to_string(), w);
    hm.insert("bias".to_string(), b);
    let vb = VarBuilder::from_tensors(hm, DType::F32, &dev);
    let conv = conv2d(1, 1, 3, Conv2dConfig { padding: 1, ..Default::default() }, vb)?;
    let y = conv.forward(&x)?;
    println!("conv2d candle: {:?}", y.flatten_all()?.to_vec1::<f32>()?);

    // conv_transpose2d: in [1,1,2,2]=[1,2,3,4], w=0..15-8 [4x4], stride 2 pad 1
    let x2 = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &dev)?;
    let w2 = Tensor::arange(0f32, 16f32, &dev)?
        .to_dtype(DType::F32)?
        .affine(1.0, -8.0)?
        .reshape((1, 1, 4, 4))?;
    let mut hm2 = std::collections::HashMap::new();
    hm2.insert("weight".to_string(), w2);
    hm2.insert("bias".to_string(), Tensor::zeros((1,), DType::F32, &dev)?);
    let vb2 = VarBuilder::from_tensors(hm2, DType::F32, &dev);
    let ct = conv_transpose2d(
        1,
        1,
        4,
        ConvTranspose2dConfig { stride: 2, padding: 1, ..Default::default() },
        vb2,
    )?;
    let y2 = ct.forward(&x2)?;
    println!("convT candle shape {:?}:", y2.shape());
    println!("convT candle: {:?}", y2.flatten_all()?.to_vec1::<f32>()?);
    Ok(())
}
