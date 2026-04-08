use std::collections::HashMap;

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
    fn dim(&self) -> usize;
}

pub struct BagOfWordsEmbedder {
    vocab: HashMap<String, usize>,
    dim:   usize,
}

impl BagOfWordsEmbedder {

    pub fn new(corpus: &[&str], dim: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for text in corpus {
            for word in text.split_whitespace() {
                let w = normalise(word);
                if !w.is_empty() {
                    *freq.entry(w).or_default() += 1;
                }
            }
        }
        let mut words: Vec<(String, usize)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1));
        let vocab = words
            .into_iter()
            .take(dim)
            .enumerate()
            .map(|(i, (w, _))| (w, i))
            .collect();
        Self { vocab, dim }
    }
}

impl Embedder for BagOfWordsEmbedder {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut vec = vec![0.0f32; self.dim];
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Ok(vec);
        }
        let weight = 1.0 / words.len() as f32;
        for word in &words {
            let w = normalise(word);
            if let Some(&idx) = self.vocab.get(&w) {
                vec[idx] += weight;
            }
        }
        Ok(vec)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

fn normalise(word: &str) -> String {
    word.to_lowercase()
        .trim_matches(|c: char| !c.is_alphabetic())
        .to_string()
}
