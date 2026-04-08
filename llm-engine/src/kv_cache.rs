use candle_core::{Result, Tensor};

pub struct Cache {
    pub k: Tensor,
    pub v: Tensor,
}

pub struct KvCache {
    layers: Vec<Option<Cache>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self { layers: (0..num_layers).map(|_| None).collect() }
    }


    pub fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)> {
        self.layers.get(layer)?.as_ref().map(|c| (&c.k, &c.v))
    }


    pub fn update(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<()> {
        self.layers[layer] = Some(Cache { k, v });
        Ok(())
    }


    pub fn clear(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
    }


    pub fn seq_len(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| l.as_ref())
            .map(|c| c.k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn zeros(seq: usize, dev: &Device) -> Tensor {

        Tensor::zeros((1usize, 8usize, seq, 128usize), DType::F32, dev).unwrap()
    }

    #[test]
    fn empty_cache_returns_none() {
        let cache = KvCache::new(4);
        assert!(cache.get(0).is_none());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn update_and_get_roundtrip() {
        let dev = Device::Cpu;
        let mut cache = KvCache::new(2);
        let k = zeros(3, &dev);
        let v = zeros(3, &dev);
        cache.update(0, k, v).unwrap();

        let (ck, cv) = cache.get(0).unwrap();
        assert_eq!(ck.dims(), &[1, 8, 3, 128]);
        assert_eq!(cv.dims(), &[1, 8, 3, 128]);
    }

    #[test]
    fn seq_len_reflects_cached_tokens() {
        let dev = Device::Cpu;
        let mut cache = KvCache::new(2);
        cache.update(0, zeros(5, &dev), zeros(5, &dev)).unwrap();
        assert_eq!(cache.seq_len(), 5);
    }

    #[test]
    fn clear_removes_all_layers() {
        let dev = Device::Cpu;
        let mut cache = KvCache::new(2);
        cache.update(0, zeros(1, &dev), zeros(1, &dev)).unwrap();
        cache.update(1, zeros(1, &dev), zeros(1, &dev)).unwrap();
        cache.clear();
        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_none());
        assert_eq!(cache.seq_len(), 0);
    }
}
