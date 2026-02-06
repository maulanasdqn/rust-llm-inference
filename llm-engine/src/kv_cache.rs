use candle_core::Tensor;

pub struct Cache {
    pub k: Tensor,
    pub v: Tensor,
}

pub struct KvCache {
    layers: Vec<Option<Cache>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
        }
    }
}
