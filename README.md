# Sunrise AMC — Voice-Powered Investor Support Assistant

End-to-end pipeline: **audio → transcript (Faster-Whisper) → RAG answer (ChromaDB + Ollama/Groq)**

---

## Project Structure

```
sunrise_rag/
├── input/                  ← Place investor_sample.mp3 and SunriseAMC_FAQ.pdf here
├── output/                 ← transcript.json, final_output.json, eval_results.json
├── data/
│   └── chroma/             ← Persistent ChromaDB vector index (auto-created, gitignored)
├── src/
│   ├── __init__.py
│   ├── transcriber.py      ← Part 1: Faster-Whisper transcription
│   ├── ingestion.py        ← FAQ-aware PDF chunking
│   ├── rag_engine.py       ← Part 2: ChromaDB + Ollama/Groq RAG
│   └── utils.py            ← Shared helpers
├── main.py                 ← Part 3: Full end-to-end pipeline
├── eval.py                 ← Manual eval against ground-truth Q&A pairs
├── requirements.txt
├── DECISIONS.md
└── .gitignore
```

---

## Prerequisites

- Python 3.10 or 3.11
- [Ollama](https://ollama.com/download) installed and running locally  
  *(or set `GROQ_API_KEY` for the free-tier cloud fallback — see below)*

---

## Setup

### 1. Clone / unzip the repository

```bash
cd sunrise_rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the LLM (Ollama)

```bash
ollama pull llama3
```

> **No GPU / Ollama not available?**  
> Set your Groq free-tier key instead (only Llama 3 / Mistral open-source models used):
> ```bash
> export GROQ_API_KEY=your_key_here
> ```

### 5. Place input files

```
input/
├── investor_sample.mp3
└── SunriseAMC_FAQ.pdf
```

---

## Run

### Full end-to-end pipeline (audio → answer)

```bash
python main.py
```

### Custom audio or PDF paths

```bash
python main.py --audio path/to/audio.mp3 --pdf path/to/faq.pdf
```

### Force rebuild of the vector index

```bash
python main.py --rebuild
```

### Text-only mode (skip audio, test RAG directly)

```bash
python main.py --query "How long does a redemption take?"
```

---

## Evaluate RAG quality

```bash
python eval.py
```

Runs 11 test cases (10 in-scope + 1 out-of-scope) and prints accuracy.  
Results saved to `output/eval_results.json`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Model to use via Ollama |
| `GROQ_API_KEY` | *(unset)* | Groq free-tier key (fallback only) |

---

## Output Files

| File | Contents |
|---|---|
| `output/transcript.json` | Full transcript with word-level confidence + timestamps |
| `output/final_output.json` | Query, answer, sources, and end-to-end latency |
| `output/eval_results.json` | Per-question eval results and overall accuracy |

---

## Latency Benchmarks (MacBook Pro M2, CPU-only)

| Step | Typical Duration |
|---|---|
| Whisper `base` transcription (15s audio) | ~5–8s |
| ChromaDB index build (first run) | ~3–5s |
| Embedding + retrieval | ~200ms |
| Ollama Llama 3 8B generation | ~8–15s |
| **Total (first run)** | **~16–28s** |
| **Total (subsequent runs, index cached)** | **~10–20s** |

---

## .gitignore

```
.venv/
data/
__pycache__/
*.pyc
input/investor_sample.mp3
```
