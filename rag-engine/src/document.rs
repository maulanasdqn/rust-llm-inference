use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id:       String,
    pub title:    String,
    pub content:  String,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(
        id:      impl Into<String>,
        title:   impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id:       id.into(),
            title:    title.into(),
            content:  content.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub doc_id:      String,
    pub chunk_index: usize,
    pub content:     String,

    pub embedding:   Option<Vec<f32>>,
}

impl Chunk {
    pub fn new(
        doc_id:      impl Into<String>,
        chunk_index: usize,
        content:     impl Into<String>,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            chunk_index,
            content: content.into(),
            embedding: None,
        }
    }
}
