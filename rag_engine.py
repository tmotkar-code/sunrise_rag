"""
Part 2 — RAG Answer Engine
Embed FAQ chunks → ChromaDB → retrieve → answer via local LLM (Ollama).

Model choices (documented in DECISIONS.md):
 - Embedding : sentence-transformers/all-MiniLM-L6-v2  (fast, 384-dim, MIT)
 - LLM       : llama3 via Ollama (or mistral as fallback)
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

import chromadb
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.ingestion import load_pdf_text, chunk_faq
from src.utils import get_logger

logger = get_logger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "sunrise_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3")          # override via env
N_RESULTS = 3                                              # top-k retrieval

# Groq fallback (free tier, open-source models only)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

_SYSTEM_PROMPT = """You are a helpful investor support agent for Sunrise Asset Management Co. Ltd.
Answer the investor's question strictly based on the FAQ excerpts provided.
- Be concise and factual.
- If the answer is not in the excerpts, say: "I don't have information on that. Please contact support at support@sunriseamc.in."
- Always end your answer with: "Source: <FAQ question number(s)>" (e.g. "Source: Q7").
- Do not make up information."""


def _is_ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama(prompt: str, system: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _call_groq(prompt: str, system: str) -> str:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "Ollama is not available and GROQ_API_KEY is not set. "
            "Either start Ollama or set the GROQ_API_KEY environment variable."
        )
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _generate_answer(context_chunks: list[dict], query: str) -> str:
    context_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in context_chunks
    )
    prompt = (
        f"FAQ Excerpts:\n{context_text}\n\n"
        f"Investor Question: {query}\n\n"
        "Answer:"
    )

    if _is_ollama_available():
        logger.info("Using Ollama (%s) for generation", LLM_MODEL)
        return _call_ollama(prompt, _SYSTEM_PROMPT)
    else:
        logger.warning("Ollama not reachable — falling back to Groq free tier")
        return _call_groq(prompt, _SYSTEM_PROMPT)


class RAGEngine:
    """
    Manages the vector store and orchestrates retrieval + generation.
    """

    def __init__(self, pdf_path: str, rebuild_index: bool = False):
        self._pdf_path = pdf_path
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = self._get_or_build_collection(rebuild_index)

    # ── Index management ────────────────────────────────────────────────────

    def _get_or_build_collection(
        self, rebuild: bool
    ) -> chromadb.Collection:
        existing = [c.name for c in self._client.list_collections()]

        if COLLECTION_NAME in existing and not rebuild:
            logger.info("Loading existing ChromaDB collection '%s'", COLLECTION_NAME)
            return self._client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embed_fn,
            )

        if COLLECTION_NAME in existing:
            logger.info("Rebuilding ChromaDB collection '%s'", COLLECTION_NAME)
            self._client.delete_collection(COLLECTION_NAME)

        logger.info("Building new ChromaDB collection from %s …", self._pdf_path)
        collection = self._client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_pdf(collection)
        return collection

    def _index_pdf(self, collection: chromadb.Collection) -> None:
        raw_text = load_pdf_text(self._pdf_path)
        chunks = chunk_faq(raw_text)

        t0 = time.perf_counter()
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"section": c["section"], "chunk_id": c["chunk_id"]} for c in chunks],
        )
        logger.info(
            "Indexed %d chunks in %.2fs", len(chunks), time.perf_counter() - t0
        )

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Retrieve top-k chunks for the question, then generate a grounded answer.

        Returns
        -------
        {
          "answer": "...",
          "sources": ["Q7", "Q8"],
          "retrieved_chunks": [...],
          "retrieval_scores": [0.91, 0.87, 0.72]
        }
        """
        if not question.strip():
            return {
                "answer": "No question was provided (empty transcript).",
                "sources": [],
                "retrieved_chunks": [],
                "retrieval_scores": [],
            }

        logger.info("Retrieving top-%d chunks for query: %r", N_RESULTS, question)
        t0 = time.perf_counter()

        results = self._collection.query(
            query_texts=[question],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_elapsed = time.perf_counter() - t0

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # Cosine distance → similarity score (1 = identical, 0 = orthogonal)
        scores = [round(1 - d, 4) for d in distances]

        retrieved = [
            {"chunk_id": m["chunk_id"], "text": doc, "score": score}
            for doc, m, score in zip(docs, metas, scores)
        ]

        logger.info(
            "Retrieval done in %.3fs | top scores: %s",
            retrieval_elapsed,
            scores,
        )

        # Filter low-relevance chunks (score < 0.3 → likely out-of-scope query)
        relevant = [c for c in retrieved if c["score"] >= 0.30]
        if not relevant:
            return {
                "answer": (
                    "I don't have information on that. "
                    "Please contact support at support@sunriseamc.in."
                ),
                "sources": [],
                "retrieved_chunks": retrieved,
                "retrieval_scores": scores,
            }

        t1 = time.perf_counter()
        answer = _generate_answer(relevant, question)
        gen_elapsed = time.perf_counter() - t1

        # Extract cited Q-numbers from the answer text  (e.g. "Source: Q7, Q8")
        sources = list(dict.fromkeys(
            m for c in relevant for m in [c["chunk_id"]]
        ))

        logger.info("Generation done in %.2fs", gen_elapsed)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "retrieval_scores": scores,
        }
