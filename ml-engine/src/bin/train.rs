use burn::backend::{Autodiff, NdArray};
use burn::optim::AdamConfig;
use ml_engine::training::{train, TrainingConfig};

type B = NdArray<f32>;

fn main() {
    let artifact_dir = "/tmp/ml-engine-artifacts";
    std::fs::create_dir_all(artifact_dir).expect("failed to create artifact directory");

    let config = TrainingConfig::new(AdamConfig::new());

    println!("Training MLP on Iris dataset ({} epochs) …", config.num_epochs);
    train::<Autodiff<B>>(artifact_dir, config, Default::default());
}
