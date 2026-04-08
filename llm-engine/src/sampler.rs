use candle_core::{Result, Tensor, D};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;

pub trait Sampler: Send + Sync {
    fn sample(&self, logits: &Tensor) -> Result<u32>;
}

pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&self, logits: &Tensor) -> Result<u32> {
        flatten_logits(logits)?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()
    }
}

pub struct Temperature {
    pub temp: f64,
}

impl Temperature {
    pub fn new(temp: f64) -> Self {
        assert!(temp > 0.0, "temperature must be positive");
        Self { temp }
    }
}

impl Sampler for Temperature {
    fn sample(&self, logits: &Tensor) -> Result<u32> {
        let logits = (flatten_logits(logits)? / self.temp)?;
        let probs = softmax_to_vec(&logits)?;
        multinomial_sample(&probs)
    }
}

pub struct TopK {
    pub k:    usize,
    pub temp: f64,
}

impl TopK {
    pub fn new(k: usize, temp: f64) -> Self {
        assert!(k > 0, "k must be > 0");
        assert!(temp > 0.0, "temperature must be positive");
        Self { k, temp }
    }
}

impl Sampler for TopK {
    fn sample(&self, logits: &Tensor) -> Result<u32> {
        let logits = flatten_logits(logits)?;
        let mut scored: Vec<(f32, u32)> = logits
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();


        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(self.k);


        let max_logit = scored[0].0;
        let weights: Vec<f32> = scored
            .iter()
            .map(|(v, _)| ((v - max_logit) as f64 / self.temp).exp() as f32)
            .collect();
        let sum: f32 = weights.iter().sum();
        let probs: Vec<f32> = weights.iter().map(|w| w / sum).collect();

        let idx = multinomial_sample(&probs)? as usize;
        Ok(scored[idx].1)
    }
}

pub struct TopP {
    pub p:    f64,
    pub temp: f64,
}

impl TopP {
    pub fn new(p: f64, temp: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
        assert!(temp > 0.0, "temperature must be positive");
        Self { p, temp }
    }
}

impl Sampler for TopP {
    fn sample(&self, logits: &Tensor) -> Result<u32> {
        let logits = flatten_logits(logits)?;
        let mut scored: Vec<(f32, u32)> = logits
            .to_vec1::<f32>()?
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();


        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());


        let max_logit = scored[0].0;
        let exp: Vec<f32> = scored
            .iter()
            .map(|(v, _)| ((v - max_logit) as f64 / self.temp).exp() as f32)
            .collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|e| e / sum).collect();


        let mut cum = 0.0f32;
        let nucleus_len = probs
            .iter()
            .position(|&p| {
                cum += p;
                cum >= self.p as f32
            })
            .map(|i| i + 1)
            .unwrap_or(probs.len());

        let nucleus_probs = &probs[..nucleus_len];
        let sum_n: f32 = nucleus_probs.iter().sum();
        let renorm: Vec<f32> = nucleus_probs.iter().map(|p| p / sum_n).collect();

        let idx = multinomial_sample(&renorm)? as usize;
        Ok(scored[idx].1)
    }
}

fn flatten_logits(logits: &Tensor) -> Result<Tensor> {
    logits.squeeze(0)?.squeeze(0)
}

fn softmax_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?
        .squeeze(0)?
        .to_vec1::<f32>()
}

fn multinomial_sample(probs: &[f32]) -> Result<u32> {
    let dist = WeightedIndex::new(probs)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(dist.sample(&mut rng()) as u32)
}
