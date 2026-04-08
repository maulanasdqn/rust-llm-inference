# rust-llm-inference

A Cargo workspace of three independent ML/AI engine crates written in Rust. Each crate targets a distinct concern: language model inference, retrieval-augmented generation, and supervised model training.

```
rust-llm-inference/
├── llm-engine/     # Llama 3 transformer inference (candle)
├── rag-engine/     # Retrieval-augmented generation pipeline (rig-core)
└── ml-engine/      # MLP classifier training on CPU (burn)
```

---

## Requirements

- Rust 1.85+ (workspace uses edition 2024 for `llm-engine`, 2021 for the others)
- Cargo

No CUDA or GPU drivers are required. All three engines run on CPU.

---

## crates

### llm-engine

Implements a Llama 3 transformer from scratch using [candle](https://github.com/huggingface/candle). The architecture covers every component needed for autoregressive generation: token embedding, RMSNorm, Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA), SwiGLU MLP, and a per-layer KV cache.

**Key modules:**

| Module | Purpose |
|---|---|
| `model.rs` | `Config`, `Llama`, causal attention mask |
| `forward_pass.rs` | `RmsNorm`, `RotaryEmbedding`, `Attention`, `Mlp`, `Block` |
| `kv_cache.rs` | Layer-indexed key/value tensor store |
| `backward_pass.rs` | Cross-entropy loss, perplexity, causal label shift |
| `sampler.rs` | `Greedy`, `Temperature`, `TopK`, `TopP` sampling strategies |
| `generator.rs` | `Generator` — tokenisation, prefill, decode loop, chat formatting |

**Architecture details (Llama 3 8B defaults):**

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 128 256 |
| Hidden size | 4 096 |
| Intermediate size | 14 336 |
| Layers | 32 |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 8 |
| RoPE theta | 500 000 |

**Tests (26):** causal mask shape and values, KV cache accumulation and clearing, prefill and decode logit shapes, backward pass loss shape, sampler output range, generator streaming output.

---

### rag-engine

A self-contained retrieval-augmented generation pipeline. Documents are split into overlapping word-level chunks, embedded, stored in an in-memory vector store, and retrieved by cosine similarity. The pipeline assembles a numbered context block and formats it into a prompt ready for any LLM.

**Key modules:**

| Module | Purpose |
|---|---|
| `document.rs` | `Document`, `Chunk` types (serde-serialisable) |
| `chunker.rs` | `TextSplitter` — configurable size and overlap |
| `embedder.rs` | `Embedder` trait, `BagOfWordsEmbedder` (frequency-weighted vocab) |
| `store.rs` | `VectorStore` — brute-force cosine similarity search |
| `pipeline.rs` | `RagPipeline` — index documents, retrieve chunks, build prompts |

**Tests (6):** chunk count and overlap, nearest chunk retrieval, top-k ordering, missing-embedding skip.

---

### ml-engine

Trains a three-layer MLP on the Iris dataset using [burn](https://github.com/tracel-ai/burn) with the `NdArray` CPU backend. Demonstrates burn's full training loop including dataloader, metrics, checkpointing, and model serialisation.

**Architecture:**

```
Input (4) -> Linear -> ReLU -> Linear (64) -> ReLU -> Linear -> Output (3)
```

**Training defaults:**

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 4 |
| Learning rate | 1e-3 |
| Optimiser | Adam |
| Seed | 42 |

**Key modules:**

| Module | Purpose |
|---|---|
| `dataset.rs` | `IrisSample`, `IrisDataset` (15 samples, 3 classes), `IrisBatcher` |
| `model.rs` | `MlpClassifier<B>` with `forward` and `forward_classification` |
| `training.rs` | `TrainingConfig`, `train<B: AutodiffBackend>` |

---

## Build

```bash
# build all crates
cargo build

# run all tests
cargo test
```

---

## Running the demos

### LLM inference

The `generate` binary requires real Llama 3 weights. Download them first:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B \
    --local-dir ./weights/llama-3-8b \
    --include '*.safetensors' 'tokenizer.json'
```

Then run one of the five use cases:

```bash
# text completion (greedy)
cargo run -p llm-engine --bin generate -- \
    --model ./weights/llama-3-8b --usecase text

# code completion (greedy)
cargo run -p llm-engine --bin generate -- \
    --model ./weights/llama-3-8b --usecase code

# question answering (temperature = 0.6)
cargo run -p llm-engine --bin generate -- \
    --model ./weights/llama-3-8b --usecase qa

# summarisation (top-p = 0.9)
cargo run -p llm-engine --bin generate -- \
    --model ./weights/llama-3-8b --usecase summarize

# multi-turn chat with system prompt (top-p = 0.9)
cargo run -p llm-engine --bin generate -- \
    --model ./weights/llama-3-8b --usecase chat
```

### RAG pipeline

No external assets required. The demo indexes 13 knowledge snippets and runs three queries against them:

```bash
cargo run -p rag-engine --bin rag
```

### ML training

No external assets required. Trains on the embedded Iris dataset and saves the model to `/tmp/ml-engine-artifacts/`:

```bash
cargo run -p ml-engine --bin train
```

---

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| candle-core | 0.8.2 | Tensor operations and device abstraction |
| candle-nn | 0.8.2 | Neural network layers (linear, embedding, norms) |
| candle-transformers | 0.8.2 | Transformer utilities |
| tokenizers | 0.21.0 | HuggingFace fast tokenizer bindings |
| rand | 0.9 | Stochastic sampling |
| rig-core | 0.6 | LLM provider and RAG abstractions |
| burn | 0.16 | Deep learning framework (ndarray + autodiff) |
| anyhow | 1.0 | Error propagation |
| serde / serde_json | 1.0 | Serialisation |
| tokio | 1.0 | Async runtime (rag-engine) |
