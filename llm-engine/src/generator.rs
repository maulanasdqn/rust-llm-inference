use std::io::Write as _;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::sampler::Sampler;
use crate::{KvCache, Llama};

pub struct Generator {
    pub model:    Llama,
    pub tokenizer: Tokenizer,
    pub device:   Device,
}

impl Generator {
    pub fn new(model: Llama, tokenizer: Tokenizer, device: Device) -> Self {
        Self { model, tokenizer, device }
    }









    pub fn generate_streaming(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampler: &dyn Sampler,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let eos = self.model.config.eos_token_id.unwrap_or(128001);


        let prompt_ids: Vec<u32> = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
            .get_ids()
            .to_vec();

        let mut cache = KvCache::new(self.model.config.num_hidden_layers);
        let mut generated_ids: Vec<u32> = Vec::new();


        let input = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .unsqueeze(0)
            .context("prefill unsqueeze")?;
        let logits = self.model.forward(&input, 0, &mut cache)?;
        let mut next_id = sampler.sample(&logits)?;


        for _ in 0..max_new_tokens {
            if next_id == eos {
                break;
            }
            generated_ids.push(next_id);

            if let Ok(piece) = self.tokenizer.decode(&[next_id], false) {
                on_token(&piece);
            }

            let input = Tensor::new(&[next_id], &self.device)?
                .unsqueeze(0)
                .context("decode unsqueeze")?;
            let logits = self.model.forward(&input, cache.seq_len(), &mut cache)?;
            next_id = sampler.sample(&logits)?;
        }

        self.tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))
    }


    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampler: &dyn Sampler,
    ) -> Result<String> {
        self.generate_streaming(prompt, max_new_tokens, sampler, |_| {})
    }




    pub fn chat(
        &self,
        user_message: &str,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        sampler: &dyn Sampler,
    ) -> Result<String> {
        let sys = system_prompt.unwrap_or("You are a helpful assistant.");
        let prompt = format!(
            "<|begin_of_text|>\
             <|start_header_id|>system<|end_header_id|>\n{sys}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n"
        );
        self.generate(&prompt, max_new_tokens, sampler)
    }


    pub fn print_streaming(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampler: &dyn Sampler,
    ) -> Result<String> {
        self.generate_streaming(prompt, max_new_tokens, sampler, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        })
    }
}
