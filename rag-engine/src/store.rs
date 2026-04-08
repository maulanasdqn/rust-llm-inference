use crate::document::Chunk;

pub struct VectorStore {
    chunks: Vec<Chunk>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    pub fn insert(&mut self, chunk: Chunk) {
        self.chunks.push(chunk);
    }

    pub fn insert_many(&mut self, chunks: Vec<Chunk>) {
        self.chunks.extend(chunks);
    }




    pub fn search(&self, query_emb: &[f32], top_k: usize) -> Vec<&Chunk> {
        let mut scored: Vec<(f32, usize)> = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                c.embedding
                    .as_ref()
                    .map(|e| (cosine_similarity(query_emb, e), i))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.into_iter().take(top_k).map(|(_, i)| &self.chunks[i]).collect()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Chunk;

    fn chunk_with_emb(doc_id: &str, emb: Vec<f32>) -> Chunk {
        let mut c = Chunk::new(doc_id, 0, "");
        c.embedding = Some(emb);
        c
    }

    #[test]
    fn retrieves_nearest_chunk() {
        let mut store = VectorStore::new();
        store.insert(chunk_with_emb("doc-0", vec![1.0, 0.0]));
        store.insert(chunk_with_emb("doc-1", vec![0.0, 1.0]));

        let results = store.search(&[1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc-0");
    }

    #[test]
    fn returns_top_k_in_order() {
        let mut store = VectorStore::new();
        store.insert(chunk_with_emb("a", vec![1.0, 0.0]));
        store.insert(chunk_with_emb("b", vec![0.8, 0.6]));
        store.insert(chunk_with_emb("c", vec![0.0, 1.0]));

        let results = store.search(&[1.0, 0.0], 2);
        assert_eq!(results[0].doc_id, "a");
        assert_eq!(results[1].doc_id, "b");
    }

    #[test]
    fn skips_chunks_without_embedding() {
        let mut store = VectorStore::new();
        store.insert(Chunk::new("no-emb", 0, ""));
        store.insert(chunk_with_emb("has-emb", vec![1.0]));

        let results = store.search(&[1.0], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "has-emb");
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}
