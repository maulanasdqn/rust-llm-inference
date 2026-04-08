use crate::document::{Chunk, Document};

pub struct TextSplitter {
    chunk_size:    usize,
    chunk_overlap: usize,
}

impl TextSplitter {


    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        assert!(
            chunk_overlap < chunk_size,
            "chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        );
        Self { chunk_size, chunk_overlap }
    }


    pub fn split(&self, doc: &Document) -> Vec<Chunk> {
        let words: Vec<&str> = doc.content.split_whitespace().collect();
        if words.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut idx = 0usize;
        let step = self.chunk_size - self.chunk_overlap;

        loop {
            let end = (start + self.chunk_size).min(words.len());
            let content = words[start..end].join(" ");
            chunks.push(Chunk::new(&doc.id, idx, content));
            idx += 1;

            if end == words.len() {
                break;
            }
            start += step;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;

    #[test]
    fn splits_into_chunks_with_overlap() {
        let doc = Document::new("d1", "t", "a b c d e f g h i j");
        let splitter = TextSplitter::new(4, 1);
        let chunks = splitter.split(&doc);
        assert_eq!(chunks[0].content, "a b c d");
        assert_eq!(chunks[1].content, "d e f g");
    }

    #[test]
    fn single_chunk_when_content_fits() {
        let doc = Document::new("d2", "t", "hello world");
        let splitter = TextSplitter::new(10, 2);
        let chunks = splitter.split(&doc);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
    }

    #[test]
    fn empty_document_yields_no_chunks() {
        let doc = Document::new("d3", "t", "");
        let splitter = TextSplitter::new(4, 1);
        assert!(splitter.split(&doc).is_empty());
    }
}
