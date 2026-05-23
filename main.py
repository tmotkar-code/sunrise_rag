"""
Sunrise AMC — Voice-Powered Investor Support Assistant
End-to-end pipeline: audio → transcript → RAG answer
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.transcriber import transcribe_audio
from src.rag_engine import RAGEngine
from src.utils import save_json, get_logger

logger = get_logger(__name__)


def run_pipeline(audio_path: str, pdf_path: str, rebuild_index: bool = False) -> dict:
    start = time.perf_counter()

    # ── Part 1: Transcribe ──────────────────────────────────────────────────
    logger.info("Step 1/3 — Transcribing audio: %s", audio_path)
    transcript_result = transcribe_audio(audio_path)
    save_json(transcript_result, "output/transcript.json")
    logger.info("Transcript: %s", transcript_result["text"])

    # ── Part 2: RAG ─────────────────────────────────────────────────────────
    logger.info("Step 2/3 — Loading RAG engine (PDF: %s)", pdf_path)
    rag = RAGEngine(pdf_path=pdf_path, rebuild_index=rebuild_index)

    # ── Part 3: Connect ──────────────────────────────────────────────────────
    query = transcript_result["text"]
    logger.info("Step 3/3 — Querying RAG with: %r", query)
    rag_result = rag.query(query)

    elapsed = time.perf_counter() - start

    final_output = {
        "query": query,
        "transcript": transcript_result,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "latency_seconds": round(elapsed, 3),
    }

    save_json(final_output, "output/final_output.json")
    logger.info("Pipeline complete in %.2fs", elapsed)
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Sunrise AMC Voice RAG Pipeline")
    parser.add_argument(
        "--audio", default="input/investor_sample.mp3", help="Path to investor audio file"
    )
    parser.add_argument(
        "--pdf", default="input/SunriseAMC_FAQ.pdf", help="Path to FAQ PDF"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild of the ChromaDB vector index"
    )
    parser.add_argument(
        "--query", default=None, help="Skip audio; run RAG directly with this text query"
    )
    args = parser.parse_args()

    if args.query:
        # Text-only mode — useful for testing RAG without an audio file
        from src.rag_engine import RAGEngine
        rag = RAGEngine(pdf_path=args.pdf, rebuild_index=args.rebuild)
        result = rag.query(args.query)
        print(json.dumps(result, indent=2))
        return

    if not Path(args.audio).exists():
        logger.error("Audio file not found: %s", args.audio)
        sys.exit(1)
    if not Path(args.pdf).exists():
        logger.error("PDF file not found: %s", args.pdf)
        sys.exit(1)

    result = run_pipeline(args.audio, args.pdf, rebuild_index=args.rebuild)
    print("\n" + "=" * 60)
    print("QUERY  :", result["query"])
    print("ANSWER :", result["answer"])
    print("SOURCE :", result["sources"])
    print(f"LATENCY: {result['latency_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
