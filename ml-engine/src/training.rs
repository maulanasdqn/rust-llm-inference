use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

use crate::dataset::{IrisBatch, IrisBatcher, IrisDataset};
use crate::model::MlpClassifier;

impl<B: AutodiffBackend> TrainStep<IrisBatch<B>, ClassificationOutput<B>>
    for MlpClassifier<B>
{
    fn step(&self, batch: IrisBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<IrisBatch<B>, ClassificationOutput<B>> for MlpClassifier<B> {
    fn step(&self, batch: IrisBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,

    #[config(default = 50)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub batch_size: usize,

    #[config(default = 1)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-3)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    B::seed(config.seed);

    let batcher_train = IrisBatcher::<B>::new(device.clone());
    let batcher_valid = IrisBatcher::<B::InnerBackend>::new(device.clone());

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(IrisDataset::train());

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(IrisDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            MlpClassifier::new(4, 64, 3, &device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let trained = learner.fit(train_loader, valid_loader);

    trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("failed to save model");

    println!("Model saved to {artifact_dir}/model");
}
