# DECISIONS.md — Sunrise AMC Voice RAG Pipeline

## 1. Model Selection

### Embedding model — `all-MiniLM-L6-v2`
- **Why**: 384-dimensional, MIT-licensed, ships with `sentence-transformers`.  
  Runs entirely on CPU in ~50 ms per batch; no GPU needed.  
  Benchmarks (MTEB) show strong performance on semantic similarity for short Q&A text.
- **Tradeoff**: A larger model (e.g. `BAAI/bge-large-en-v1.5`) would improve retrieval precision ~5–8 points on MTEB, but requires ~3× more memory and is 4× slower on CPU.

### LLM — `llama3` via Ollama (fallback: Groq free tier)
- **Why**: Llama 3 8B is currently the best open-weight model at its size class.  
  It follows instructions reliably, supports system prompts, and fits in ~8 GB RAM with 4-bit quantisation via Ollama.
- **Groq fallback**: Added because the assignment explicitly permits it for hardware-constrained machines.  Only `llama3-8b-8192` (open-source, free) is used — no paid API.
- **Tradeoff**: Mistral 7B is a valid alternative; chosen Llama 3 because it scores higher on instruction-following benchmarks (MT-Bench 8.3 vs 7.6 for Mistral 7B).

---

## 2. Chunking Strategy

**Strategy: FAQ-aware semantic splitting (Q&A boundary chunking)**

The source document is a structured FAQ with 10 numbered Q&A pairs.  
The optimal retrieval unit is one complete Q+A pair because:

1. Splitting mid-question or mid-answer loses the semantic unit the LLM needs to answer correctly.
2. Co-locating the Q-number with its answer enables reliable source citation.
3. Each pair is ~150–400 characters — well within the embedding model's context window and small enough that cosine similarity is precise rather than diluted.

**Parameters**
- `MAX_CHUNK_CHARS = 800` — soft ceiling; sub-splits only if a block exceeds this.
- `OVERLAP_CHARS = 120` — ~1 sentence overlap when sub-splitting, preserving cross-sentence context.
- Chunking falls back to sentence-boundary splitting (not character-count splitting) to avoid mid-sentence cuts.

**What we rejected**
- Naive character-count chunks (e.g. 512 chars, no overlap): splits Q&A pairs unpredictably; kills citation accuracy.
- Recursive character text splitter (LangChain default): designed for prose, not FAQ tables — would produce similarly bad splits here.

---

## 3. Tradeoffs Made

| Decision | Simplicity/Speed Gain | What We Sacrificed |
|---|---|---|
| `int8` quantisation on CPU | Runs on any laptop | ~1–2% accuracy vs float32 |
| `base` Whisper model | Fast transcription (~5s on CPU) | Lower accuracy on accents / noisy audio vs `medium` or `large-v3` |
| No re-ranking step | Simpler pipeline, lower latency | A cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) would improve answer quality on ambiguous queries |
| Single-turn RAG (no conversation history) | No state management needed | Cannot handle follow-up questions |
| Groq fallback instead of llama.cpp direct | Easier setup | External network call; not fully local |

---

## 4. Production Readiness

**What works well at prototype scale:**
- Persistent ChromaDB index: re-indexing only happens on `--rebuild`, so latency after the first run is ~200 ms.
- VAD filter in Whisper: prevents hallucinated transcript text during silence.
- Relevance threshold (score ≥ 0.30): returns a polite "I don't know" instead of a hallucinated answer for out-of-scope queries.

**What would need to change at scale (production deployment):**

| Concern | Current State | Production Fix |
|---|---|---|
| Latency | ~5–15s end-to-end on CPU | GPU inference (A10G); async pipeline; response streaming |
| Whisper model | `base` — low accuracy on accented Hindi-English | Fine-tune `medium` on investor domain audio; or use `large-v3` |
| Vector DB | Local ChromaDB on disk | Managed Qdrant / Weaviate with HNSW tuning and replicas |
| LLM serving | Ollama single-process | vLLM with continuous batching; load balancer across replicas |
| FAQ updates | Manual `--rebuild` | Incremental indexing triggered by document upload webhook |
| Observability | `logging` to stdout | OpenTelemetry traces; LLM input/output logging for audit |
| Security | No auth | mTLS between services; input sanitisation; PII redaction before LLM |
| Fallback | Groq free tier | SLA-backed self-hosted LLM; circuit breaker to human agent queue |
| Eval | Manual ground-truth set | Automated regression with RAGAS (faithfulness + answer relevancy scores) |
