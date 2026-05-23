"""
PDF ingestion and chunking for the Sunrise AMC FAQ.

Chunking strategy: FAQ-aware semantic splitting
─────────────────────────────────────────────────
The FAQ is structured as numbered Q&A pairs (Q1…Q10 across 4 sections).
The ideal chunk is one complete Q+A pair so that retrieval always returns a
self-contained answer with its question number (used as the source citation).

Algorithm:
1. Extract text from the PDF with pypdf.
2. Split on the regex pattern  r"Q\d+\." to isolate each Q&A block.
3. Each block becomes one chunk.  The block is prefixed with its FAQ number
   so that the LLM can cite it (e.g. "Source: Q7").
4. If a block exceeds MAX_CHUNK_CHARS it is split further on sentence
   boundaries with a TOKEN_OVERLAP character overlap to preserve context.

This beats naive character-count chunking because:
 - No Q&A pair is ever split in the middle.
 - The citation anchor (Q-number) is always co-located with the answer text.
 - Overlap is applied only when genuinely needed.
"""

import re
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

from src.utils import get_logger

logger = get_logger(__name__)

MAX_CHUNK_CHARS = 800    # soft ceiling per chunk before sub-splitting
OVERLAP_CHARS = 120      # overlap when a block exceeds MAX_CHUNK_CHARS


def load_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF, page by page."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    logger.info("Extracted %d characters from %d pages", len(text), len(pages))
    return text


def _split_on_sentences(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Fallback splitter: split a long block into sentence-boundary chunks
    with overlap.
    """
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            # keep last <overlap> chars for context continuity
            current = current[-overlap:] + " " + sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_faq(raw_text: str) -> list[dict]:
    """
    Split FAQ text into semantic Q&A chunks.

    Returns a list of dicts:
        {
          "chunk_id": "Q3",
          "section": "KYC & Onboarding",
          "text": "Q3. Can a minor invest…  Yes. A minor can…"
        }
    """
    # Map section headers to their Q-number ranges (from the PDF structure)
    section_map = {
        "KYC & Onboarding": range(1, 4),
        "SIP & Transactions": range(4, 7),
        "Redemption & Payouts": range(7, 9),
        "Taxation": range(9, 11),
    }

    def _section_for(q_num: int) -> str:
        for name, rng in section_map.items():
            if q_num in rng:
                return name
        return "General"

    # Split on Q1. Q2. … Q10. boundaries
    pattern = re.compile(r"(?=Q\d{1,2}\.)", re.MULTILINE)
    raw_blocks = [b.strip() for b in pattern.split(raw_text) if b.strip()]

    chunks: list[dict] = []
    for block in raw_blocks:
        # Extract the Q-number
        m = re.match(r"Q(\d{1,2})\.", block)
        if not m:
            # Preamble / footer text — keep as a general chunk
            if len(block) > 50:
                chunks.append({"chunk_id": "preamble", "section": "General", "text": block})
            continue

        q_num = int(m.group(1))
        section = _section_for(q_num)
        chunk_id = f"Q{q_num}"

        if len(block) <= MAX_CHUNK_CHARS:
            chunks.append({"chunk_id": chunk_id, "section": section, "text": block})
        else:
            # Sub-split large blocks while keeping the Q-number prefix on the first sub-chunk
            sub_chunks = _split_on_sentences(block, MAX_CHUNK_CHARS, OVERLAP_CHARS)
            for i, sc in enumerate(sub_chunks):
                chunks.append({
                    "chunk_id": f"{chunk_id}_part{i+1}",
                    "section": section,
                    "text": sc,
                })

    logger.info("Created %d FAQ chunks", len(chunks))
    return chunks
