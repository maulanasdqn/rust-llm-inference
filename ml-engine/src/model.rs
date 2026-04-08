use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    train::ClassificationOutput,
};

#[derive(Module, Debug)]
pub struct MlpClassifier<B: Backend> {
    fc1:    Linear<B>,
    fc2:    Linear<B>,
    output: Linear<B>,
    act:    Relu,
}

impl<B: Backend> MlpClassifier<B> {
    pub fn new(
        input_size:  usize,
        hidden_size: usize,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1:    LinearConfig::new(input_size, hidden_size).init(device),
            fc2:    LinearConfig::new(hidden_size, hidden_size).init(device),
            output: LinearConfig::new(hidden_size, num_classes).init(device),
            act:    Relu::new(),
        }
    }


    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.act.forward(self.fc1.forward(x));
        let x = self.act.forward(self.fc2.forward(x));
        self.output.forward(x)
    }


    pub fn forward_classification(
        &self,
        inputs:  Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(inputs);
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        ClassificationOutput::new(loss, logits, targets)
    }
}
