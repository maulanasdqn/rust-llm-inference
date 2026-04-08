use anyhow::Result;

use crate::chunker::TextSplitter;
use crate::document::{Chunk, Document};
use crate::embedder::Embedder;
use crate::store::VectorStore;

pub struct RagPipeline<E: Embedder> {
    embedder: E,
    splitter: TextSplitter,
    store:    VectorStore,
}

impl<E: Embedder> RagPipeline<E> {
    pub fn new(embedder: E, chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            embedder,
            splitter: TextSplitter::new(chunk_size, chunk_overlap),
            store: VectorStore::new(),
        }
    }




    pub fn add_document(&mut self, doc: &Document) -> Result<()> {
        let mut chunks = self.splitter.split(doc);
        for chunk in &mut chunks {
            chunk.embedding = Some(self.embedder.embed(&chunk.content)?);
        }
        self.store.insert_many(chunks);
        Ok(())
    }




    pub fn retrieve<'a>(&'a self, query: &str, top_k: usize) -> Result<Vec<&'a Chunk>> {
        let q_emb = self.embedder.embed(query)?;
        Ok(self.store.search(&q_emb, top_k))
    }




    pub fn build_context(&self, query: &str, top_k: usize) -> Result<String> {
        let chunks = self.retrieve(query, top_k)?;
        let ctx = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] (doc: {}) {}", i + 1, c.doc_id, c.content))
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(ctx)
    }


    pub fn build_prompt(&self, query: &str, top_k: usize) -> Result<String> {
        let context = self.build_context(query, top_k)?;
        Ok(format!(
            "Use the following context to answer the question.\n\n\
             Context:\n{context}\n\n\
             Question: {query}\n\n\
             Answer:"
        ))
    }



    pub fn chunk_count(&self) -> usize {
        self.store.len()
    }
}
