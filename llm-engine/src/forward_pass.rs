use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = candle_core::DType::F32;
        let x = x.to_dtype(internal_dtype)?;
        let pow_x = (&x * &x)?;
        let mean_x = pow_x.mean_keepdim(candle_core::D::Minus1)?;
        let norm_x = x.broadcast_div(&(mean_x + self.eps)?.sqrt()?)?;
        let res = norm_x.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        Ok(res)
    }
}
