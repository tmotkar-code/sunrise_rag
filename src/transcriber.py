"""
Part 1 — Voice Transcription
Uses faster-whisper (CTranslate2 backend) for local, CPU/GPU speech-to-text.

Output schema:
{
  "text": "full transcript",
  "language": "en",
  "duration_seconds": 12.4,
  "words": [
    {"word": "what", "start": 0.0, "end": 0.24, "probability": 0.98},
    ...
  ]
}
"""

import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from src.utils import get_logger

logger = get_logger(__name__)

# "base" balances accuracy vs. speed on a CPU laptop.
# Switch to "small" or "medium" for better accuracy if a GPU is available.
_DEFAULT_MODEL = "base"
_DEVICE = "cpu"          # change to "cuda" if a GPU is available
_COMPUTE_TYPE = "int8"   # int8 is fast on CPU; use "float16" on GPU


def transcribe_audio(
    audio_path: str,
    model_size: str = _DEFAULT_MODEL,
    device: str = _DEVICE,
    compute_type: str = _COMPUTE_TYPE,
) -> dict[str, Any]:
    """
    Transcribe an audio file and return a structured result with word-level
    confidence scores and timestamps.

    Parameters
    ----------
    audio_path   : path to .mp3 / .wav / .m4a file
    model_size   : faster-whisper model size ("tiny", "base", "small", "medium", "large-v3")
    device       : "cpu" or "cuda"
    compute_type : "int8" (CPU) or "float16" / "float32" (GPU)

    Returns
    -------
    dict with keys: text, language, duration_seconds, words
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Loading Whisper model '%s' on %s …", model_size, device)
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    logger.info("Transcribing %s …", path.name)
    t1 = time.perf_counter()

    segments, info = model.transcribe(
        str(path),
        beam_size=5,
        word_timestamps=True,   # enables per-word confidence + timestamps
        vad_filter=True,        # silence / VAD filter reduces hallucinations
        vad_parameters={
            "min_silence_duration_ms": 500,
            "threshold": 0.5,
        },
    )

    # Materialise the lazy generator
    words_out: list[dict] = []
    full_text_parts: list[str] = []

    for segment in segments:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                words_out.append(
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                )

    elapsed = time.perf_counter() - t1
    full_text = " ".join(full_text_parts)

    logger.info("Transcription complete in %.2fs | detected language: %s",
                elapsed, info.language)

    # Edge case: empty audio / all silence
    if not full_text.strip():
        logger.warning("No speech detected in audio file.")
        full_text = ""

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 3),
        "transcription_time_seconds": round(elapsed, 3),
        "words": words_out,
    }
