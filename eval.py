"""
Simple evaluation script — tests the RAG engine against a set of known
investor questions and expected Q-number sources.

Run:
    python eval.py --pdf input/SunriseAMC_FAQ.pdf

Outputs a per-question pass/fail summary and an overall accuracy score.
"""

import argparse
import json
from src.rag_engine import RAGEngine
from src.utils import get_logger, save_json

logger = get_logger("eval")

# Ground-truth test cases: (query, expected_source_q_numbers)
TEST_CASES = [
    ("What documents do I need for KYC?", ["Q1"]),
    ("How long does KYC verification take?", ["Q2"]),
    ("Can a minor invest in mutual funds?", ["Q3"]),
    ("What is the minimum SIP amount?", ["Q4"]),
    ("How do I stop my SIP?", ["Q5"]),
    ("What happens if my SIP payment fails?", ["Q6"]),
    ("How long does a redemption take to process?", ["Q7"]),
    ("Can I redeem only part of my investment?", ["Q8"]),
    ("What is the tax on equity mutual fund gains?", ["Q9"]),
    ("Will TDS be deducted on my redemption?", ["Q10"]),
    # Edge case — out-of-scope
    ("What is the price of gold today?", []),
]


def run_eval(pdf_path: str) -> None:
    rag = RAGEngine(pdf_path=pdf_path)
    results = []
    passed = 0

    for query, expected_sources in TEST_CASES:
        result = rag.query(query)
        retrieved_sources = result["sources"]

        # Pass if any expected source appears in retrieved sources
        hit = any(src in retrieved_sources for src in expected_sources) if expected_sources else True
        if not expected_sources and result["sources"]:
            hit = False  # Should NOT retrieve anything for out-of-scope

        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1

        logger.info("[%s] Q: %r | Expected: %s | Got: %s", status, query, expected_sources, retrieved_sources)
        results.append({
            "query": query,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "answer_preview": result["answer"][:120],
            "status": status,
            "top_score": result["retrieval_scores"][0] if result["retrieval_scores"] else 0,
        })

    accuracy = passed / len(TEST_CASES)
    logger.info("Accuracy: %d/%d (%.0f%%)", passed, len(TEST_CASES), accuracy * 100)

    save_json({"accuracy": accuracy, "results": results}, "output/eval_results.json")
    print(f"\nEval complete: {passed}/{len(TEST_CASES)} passed ({accuracy*100:.0f}%)")
    print("Full results saved to output/eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="input/SunriseAMC_FAQ.pdf")
    args = parser.parse_args()
    run_eval(args.pdf)
