use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, VarBuilder};

use crate::forward_pass::{Block, RmsNorm};
use crate::kv_cache::KvCache;

pub struct Config {
    pub vocab_size:          usize,
    pub hidden_size:         usize,
    pub intermediate_size:   usize,
    pub num_hidden_layers:   usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps:        f64,
    pub rope_theta:          f32,
    pub bos_token_id:        Option<u32>,
    pub eos_token_id:        Option<u32>,
}

impl Config {
    pub fn llama_3_8b() -> Self {
        Self {
            vocab_size:          128256,
            hidden_size:         4096,
            intermediate_size:   14336,
            num_hidden_layers:   32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps:        1e-5,
            rope_theta:          500_000.0,
            bos_token_id:        Some(128000),
            eos_token_id:        Some(128001),
        }
    }
}

pub struct Llama {
    embed_tokens: candle_nn::Embedding,
    layers:       Vec<Block>,
    norm:         RmsNorm,
    lm_head:      candle_nn::Linear,
    pub config:   Config,
}

impl Llama {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Block::new(&cfg, vb_m.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head =
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self { embed_tokens, layers, norm, lm_head, config: cfg })
    }






    pub fn forward(
        &self,
        input_ids: &Tensor,
        offset: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (_b, seq) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;


        let mask = if seq == 1 {
            None
        } else {
            Some(causal_mask(seq, offset, x.dtype(), x.device())?)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let cached = cache.get(i);
            let (new_x, k, v) = layer.forward(&x, mask.as_ref(), offset, cached)?;
            cache.update(i, k, v)?;
            x = new_x;
        }

        let x = self.norm.forward(&x)?;

        let last = x.narrow(1, seq - 1, 1)?;
        self.lm_head.forward(&last)
    }
}

pub(crate) fn causal_mask(seq: usize, offset: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let total = seq + offset;
    let mask: Vec<f32> = (0..seq)
        .flat_map(|i| {
            (0..total).map(move |j| {
                if j <= i + offset {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::new(mask.as_slice(), dev)?
        .reshape((seq, total))?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;
    use crate::forward_pass::tests::tiny_cfg;



    #[test]
    fn causal_mask_shape_no_offset() {
        let dev = Device::Cpu;
        let mask = causal_mask(4, 0, DType::F32, &dev).unwrap();

        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn causal_mask_shape_with_offset() {
        let dev = Device::Cpu;

        let mask = causal_mask(1, 4, DType::F32, &dev).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 1, 5]);
    }

    #[test]
    fn causal_mask_values_lower_triangular() {
        let dev = Device::Cpu;
        let mask = causal_mask(4, 0, DType::F32, &dev).unwrap();

        let flat: Vec<f32> = mask.squeeze(0).unwrap().squeeze(0).unwrap()
            .flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(flat[0], 0.0);

        assert!(flat[1].is_infinite() && flat[1] < 0.0);

        assert_eq!(flat[3 * 4 + 3], 0.0);

        assert!(flat[2 * 4 + 3].is_infinite());
    }

    #[test]
    fn causal_mask_with_offset_all_visible() {

        let dev = Device::Cpu;
        let mask = causal_mask(1, 3, DType::F32, &dev).unwrap();
        let flat: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&v| v == 0.0), "all positions should be visible");
    }



    #[test]
    fn llama_prefill_logits_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vocab_size = cfg.vocab_size;
        let num_layers = cfg.num_hidden_layers;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Llama::new(cfg, vb).unwrap();
        let mut cache = KvCache::new(num_layers);

        let input_ids = Tensor::zeros((1usize, 5usize), DType::U32, &dev).unwrap();
        let logits = model.forward(&input_ids, 0, &mut cache).unwrap();


        assert_eq!(logits.dims(), &[1, 1, vocab_size]);
    }

    #[test]
    fn llama_decode_logits_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vocab_size = cfg.vocab_size;
        let num_layers = cfg.num_hidden_layers;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Llama::new(cfg, vb).unwrap();
        let mut cache = KvCache::new(num_layers);


        let input_ids = Tensor::zeros((1usize, 1usize), DType::U32, &dev).unwrap();
        let logits = model.forward(&input_ids, 0, &mut cache).unwrap();
        assert_eq!(logits.dims(), &[1, 1, vocab_size]);
    }



    #[test]
    fn autoregressive_kv_cache_accumulates() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let num_layers = cfg.num_hidden_layers;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Llama::new(cfg, vb).unwrap();
        let mut cache = KvCache::new(num_layers);


        let prompt = Tensor::new(&[[1u32, 2u32, 3u32, 4u32]], &dev).unwrap();
        model.forward(&prompt, 0, &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 4, "cache should hold 4 tokens after prefill");


        let next = Tensor::new(&[[5u32]], &dev).unwrap();
        model.forward(&next, cache.seq_len(), &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 5, "cache should grow to 5 after one decode step");


        let next = Tensor::new(&[[6u32]], &dev).unwrap();
        model.forward(&next, cache.seq_len(), &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 6);
    }

    #[test]
    fn cache_clear_resets_seq_len() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let num_layers = cfg.num_hidden_layers;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = Llama::new(cfg, vb).unwrap();
        let mut cache = KvCache::new(num_layers);

        let prompt = Tensor::new(&[[1u32, 2u32, 3u32]], &dev).unwrap();
        model.forward(&prompt, 0, &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 3);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }
}
