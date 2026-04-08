pub mod dataset;
pub mod model;
pub mod training;

pub use dataset::{IrisBatch, IrisBatcher, IrisDataset, IrisSample};
pub use model::MlpClassifier;
pub use training::{train, TrainingConfig};
