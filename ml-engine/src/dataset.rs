use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct IrisSample {
    pub features: [f32; 4],
    pub label:    usize,
}

pub struct IrisDataset {
    records: Vec<IrisSample>,
}

impl IrisDataset {
    pub fn train() -> Self {
        Self { records: iris_records(0, 12) }
    }

    pub fn test() -> Self {
        Self { records: iris_records(12, 3) }
    }
}

impl Dataset<IrisSample> for IrisDataset {
    fn get(&self, index: usize) -> Option<IrisSample> {
        self.records.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}

fn iris_records(offset: usize, count: usize) -> Vec<IrisSample> {

    const FEATURES: [[f32; 4]; 15] = [

        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],

        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],

        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
    ];
    const LABELS: [usize; 15] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

    FEATURES[offset..offset + count]
        .iter()
        .zip(LABELS[offset..offset + count].iter())
        .map(|(f, &l)| IrisSample { features: *f, label: l })
        .collect()
}

#[derive(Clone, Debug)]
pub struct IrisBatch<B: Backend> {

    pub inputs:  Tensor<B, 2>,

    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone)]
pub struct IrisBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> IrisBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<IrisSample, IrisBatch<B>> for IrisBatcher<B> {
    fn batch(&self, items: Vec<IrisSample>) -> IrisBatch<B> {
        let features: Vec<Tensor<B, 1>> = items
            .iter()
            .map(|s| {
                Tensor::<B, 1>::from_floats(s.features, &self.device)
            })
            .collect();

        let targets: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|s| {
                Tensor::<B, 1, Int>::from_ints([s.label as i32], &self.device)
            })
            .collect();

        IrisBatch {
            inputs:  Tensor::stack(features, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
