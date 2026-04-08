use candle_core::{Result, Tensor};

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, s, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * s, v))?;
    let targets_flat = targets.reshape((b * s,))?;
    candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
}

pub fn perplexity(loss_scalar: f32) -> f64 {
    (loss_scalar as f64).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn cross_entropy_returns_scalar() {
        let dev = Device::Cpu;

        let logits = Tensor::randn(0f32, 1f32, (2usize, 4usize, 32usize), &dev).unwrap();
        let targets = Tensor::zeros((2usize, 4usize), DType::U32, &dev).unwrap();
        let loss = cross_entropy_loss(&logits, &targets).unwrap();
        assert!(loss.dims().is_empty(), "loss should be a scalar (0-D tensor)");
    }

    #[test]
    fn cross_entropy_is_positive() {
        let dev = Device::Cpu;
        let logits = Tensor::randn(0f32, 1f32, (1usize, 3usize, 16usize), &dev).unwrap();
        let targets = Tensor::zeros((1usize, 3usize), DType::U32, &dev).unwrap();
        let loss: f32 = cross_entropy_loss(&logits, &targets).unwrap().to_scalar().unwrap();
        assert!(loss > 0.0, "cross-entropy loss should be positive");
    }

    #[test]
    fn shift_trims_one_position() {
        let dev = Device::Cpu;
        let logits = Tensor::randn(0f32, 1f32, (1usize, 6usize, 32usize), &dev).unwrap();
        let labels = Tensor::zeros((1usize, 6usize), DType::U32, &dev).unwrap();
        let (sl, tl) = shift_for_causal_lm(&logits, &labels).unwrap();
        assert_eq!(sl.dims(), &[1, 5, 32]);
        assert_eq!(tl.dims(), &[1, 5]);
    }

    #[test]
    fn perplexity_of_zero_loss_is_one() {
        assert!((perplexity(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn perplexity_increases_with_loss() {
        assert!(perplexity(1.0) > perplexity(0.5));
    }
}

pub fn shift_for_causal_lm(
    logits: &Tensor,
    labels: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let seq = logits.dim(1)?;
    let shifted_logits = logits.narrow(1, 0, seq - 1)?;
    let shifted_labels = labels.narrow(1, 1, seq - 1)?;
    Ok((shifted_logits, shifted_labels))
}
