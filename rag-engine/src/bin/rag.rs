use rag_engine::{BagOfWordsEmbedder, Document, RagPipeline};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let corpus = [
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "The borrow checker enforces memory safety at compile time without a garbage collector.",
        "Cargo is Rust's package manager and build system.",
        "Candle is a minimalist ML framework for Rust developed by Hugging Face.",
        "Rig is a Rust framework for building LLM-powered applications with a clean API.",
        "Burn is a deep learning framework written entirely in Rust, supporting multiple backends.",
        "Retrieval-augmented generation (RAG) combines dense retrieval with generative language models.",
        "A KV cache stores key and value tensors to avoid recomputation during autoregressive decoding.",
        "Transformers use multi-head self-attention to model long-range dependencies in sequences.",
        "Rotary Position Embeddings (RoPE) encode position by rotating query and key vectors.",
        "Grouped-query attention (GQA) reduces memory bandwidth by sharing KV heads across query heads.",
        "SwiGLU is an activation function used in the feed-forward layers of modern LLMs.",
        "RMSNorm normalises activations by their root-mean-square rather than mean and variance.",
    ];


    let embedder = BagOfWordsEmbedder::new(&corpus, 256);
    let mut pipeline = RagPipeline::new(embedder,  40,  8);


    for (i, text) in corpus.iter().enumerate() {
        let doc = Document::new(format!("doc-{i}"), format!("Snippet {i}"), *text);
        pipeline.add_document(&doc)?;
    }

    println!("Indexed {} chunks.\n", pipeline.chunk_count());


    let queries = [
        "What is the KV cache used for?",
        "How does Rust ensure memory safety?",
        "Which frameworks exist for ML in Rust?",
    ];

    for query in &queries {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Query: {query}");

        let prompt = pipeline.build_prompt(query, 3)?;
        println!("\n{prompt}\n");








    }

    Ok(())
}
