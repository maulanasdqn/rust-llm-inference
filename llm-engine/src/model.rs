use candle_core::DTType;

pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<i32>,
    pub eos_token_id: Option<i32>,
}

impl Config {
    pub fn llma_3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.5,
            bos_token_id: Some(128000),
            eos_token_id: Some(128001),
        }
    }
}
