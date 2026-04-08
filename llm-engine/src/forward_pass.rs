use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, VarBuilder};

use crate::model::Config;

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
        let x = x.to_dtype(DType::F32)?;
        let mean_sq = (&x * &x)?.mean_keepdim(D::Minus1)?;
        let norm_x = x.broadcast_div(&(mean_sq + self.eps)?.sqrt()?)?;
        norm_x.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, dtype: DType, dev: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = 8192usize;


        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?;


        let positions =
            Tensor::arange(0u32, max_seq_len as u32, dev)?.to_dtype(DType::F32)?;
        let freqs = positions
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq.unsqueeze(0)?)?;


        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        Ok(Self {
            cos: emb.cos()?.to_dtype(dtype)?,
            sin: emb.sin()?.to_dtype(dtype)?,
        })
    }





    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;

        let cos = self
            .cos
            .narrow(0, offset, seq)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .narrow(0, offset, seq)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        Ok((apply_rope(q, &cos, &sin)?, apply_rope(k, &cos, &sin)?))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;


    pub fn tiny_cfg() -> Config {
        Config {
            vocab_size:          64,
            hidden_size:         32,
            intermediate_size:   64,
            num_hidden_layers:   2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps:        1e-5,
            rope_theta:          10_000.0,
            bos_token_id:        Some(1),
            eos_token_id:        Some(2),
        }
    }



    #[test]
    fn rms_norm_output_shape() {
        let dev = Device::Cpu;
        let norm = RmsNorm::new(32, 1e-5, VarBuilder::zeros(DType::F32, &dev)).unwrap();
        let x = Tensor::randn(0f32, 1f32, (2usize, 8usize, 32usize), &dev).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 32]);
    }

    #[test]
    fn rms_norm_preserves_dtype() {
        let dev = Device::Cpu;
        let norm = RmsNorm::new(16, 1e-5, VarBuilder::zeros(DType::F32, &dev)).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1usize, 4usize, 16usize), &dev).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dtype(), DType::F32);
    }



    #[test]
    fn rope_preserves_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let rope = RotaryEmbedding::new(&cfg, DType::F32, &dev).unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let q = Tensor::randn(0f32, 1f32, (1usize, cfg.num_attention_heads, 6usize, head_dim), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1usize, cfg.num_key_value_heads, 6usize, head_dim), &dev).unwrap();

        let (qr, kr) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(qr.dims(), q.dims());
        assert_eq!(kr.dims(), k.dims());
    }

    #[test]
    fn rope_with_offset_preserves_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let rope = RotaryEmbedding::new(&cfg, DType::F32, &dev).unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;


        let q = Tensor::randn(0f32, 1f32, (1usize, cfg.num_attention_heads, 1usize, head_dim), &dev).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1usize, cfg.num_key_value_heads, 1usize, head_dim), &dev).unwrap();

        let (qr, kr) = rope.apply(&q, &k, 10).unwrap();
        assert_eq!(qr.dims(), q.dims());
        assert_eq!(kr.dims(), k.dims());
    }



    #[test]
    fn mlp_output_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mlp = Mlp::new(&cfg, VarBuilder::zeros(DType::F32, &dev)).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1usize, 5usize, cfg.hidden_size), &dev).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 5, cfg.hidden_size]);
    }



    #[test]
    fn attention_prefill_shapes() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let attn = Attention::new(&cfg, VarBuilder::zeros(DType::F32, &dev)).unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let x = Tensor::randn(0f32, 1f32, (1usize, 4usize, cfg.hidden_size), &dev).unwrap();
        let (out, k, v) = attn.forward(&x, None, 0, None).unwrap();

        assert_eq!(out.dims(), &[1, 4, cfg.hidden_size]);
        assert_eq!(k.dims(), &[1, cfg.num_key_value_heads, 4, head_dim]);
        assert_eq!(v.dims(), &[1, cfg.num_key_value_heads, 4, head_dim]);
    }

    #[test]
    fn attention_decode_kv_cache_grows() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let attn = Attention::new(&cfg, VarBuilder::zeros(DType::F32, &dev)).unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;


        let k_cache = Tensor::zeros((1usize, cfg.num_key_value_heads, 4usize, head_dim), DType::F32, &dev).unwrap();
        let v_cache = Tensor::zeros((1usize, cfg.num_key_value_heads, 4usize, head_dim), DType::F32, &dev).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1usize, 1usize, cfg.hidden_size), &dev).unwrap();
        let (out, k_new, v_new) = attn.forward(&x, None, 4, Some((&k_cache, &v_cache))).unwrap();

        assert_eq!(out.dims(), &[1, 1, cfg.hidden_size]);

        assert_eq!(k_new.dims(), &[1, cfg.num_key_value_heads, 5, head_dim]);
        assert_eq!(v_new.dims(), &[1, cfg.num_key_value_heads, 5, head_dim]);
    }



    #[test]
    fn block_residual_shape_preserved() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let block = Block::new(&cfg, VarBuilder::zeros(DType::F32, &dev)).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1usize, 6usize, cfg.hidden_size), &dev).unwrap();
        let (out, _, _) = block.forward(&x, None, 0, None).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn block_decode_with_cache() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let block = Block::new(&cfg, VarBuilder::zeros(DType::F32, &dev)).unwrap();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let k_cache = Tensor::zeros((1usize, cfg.num_key_value_heads, 3usize, head_dim), DType::F32, &dev).unwrap();
        let v_cache = k_cache.clone();

        let x = Tensor::randn(0f32, 1f32, (1usize, 1usize, cfg.hidden_size), &dev).unwrap();
        let (out, k_new, _) = block.forward(&x, None, 3, Some((&k_cache, &v_cache))).unwrap();

        assert_eq!(out.dims(), &[1, 1, cfg.hidden_size]);
        assert_eq!(k_new.dim(2).unwrap(), 4);
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[x2.neg()?, x1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?
}

pub struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    rope:       RotaryEmbedding,
    n_heads:    usize,
    n_kv_heads: usize,
    head_dim:   usize,
    n_rep:      usize,
}

impl Attention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_hidden = cfg.num_key_value_heads * head_dim;

        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_hidden, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_hidden, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,
            rope: RotaryEmbedding::new(cfg, DType::F32, vb.device())?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            n_rep: cfg.num_attention_heads / cfg.num_key_value_heads,
        })
    }


    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;


        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;


        let (q, k) = self.rope.apply(&q, &k, offset)?;


        let (k, v) = match kv_cache {
            Some((k_cache, v_cache)) => (
                Tensor::cat(&[k_cache, &k], 2)?,
                Tensor::cat(&[v_cache, &v], 2)?,
            ),
            None => (k, v),
        };


        let k_exp = repeat_kv(&k, self.n_rep)?;
        let v_exp = repeat_kv(&v, self.n_rep)?;


        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k_exp.transpose(2, 3)?)? * scale)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn
            .matmul(&v_exp)?
            .transpose(1, 2)?
            .reshape((b, seq, self.n_heads * self.head_dim))?;

        Ok((self.o_proj.forward(&out)?, k, v))
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, seq, d) = x.dims4()?;


    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, d))?
        .contiguous()?
        .reshape((b, n_kv * n_rep, seq, d))
}

pub struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj:   candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

pub struct Block {
    self_attn:                 Attention,
    mlp:                       Mlp,
    input_layernorm:           RmsNorm,
    post_attention_layernorm:  RmsNorm,
}

impl Block {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }


    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {

        let normed = self.input_layernorm.forward(x)?;
        let (attn_out, k, v) = self.self_attn.forward(&normed, mask, offset, kv_cache)?;
        let x = (x + attn_out)?;


        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        Ok(((x + ffn_out)?, k, v))
    }
}
