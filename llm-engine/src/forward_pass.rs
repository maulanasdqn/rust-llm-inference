use candle_core::{Device, Tensor};

pub fn test_candle() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device)?;
    let c = (a + b)?;
    println!("Candle Tensor Addition: {:?}", c.to_vec1::<f32>()?);
    Ok(())
}
