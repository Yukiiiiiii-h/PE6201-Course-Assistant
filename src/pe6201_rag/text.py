"""Text normalization and chunking helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
    "does", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "you", "your", "use", "instead", "team", "teams",
}


def normalize_text(text: str) -> str:
    # Some older PDFs expose unpaired UTF-16 surrogate code points. SQLite's
    # UTF-8 encoder rejects them, so replace only those invalid code points.
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    # PDF extractors use single newlines for visual line wrapping. Joining
    # those lines reconstructs sentences; true paragraph breaks remain.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def terms(text: str, *, drop_stopwords: bool = True) -> list[str]:
    values = [m.group(0).lower() for m in WORD_RE.finditer(text)]
    if drop_stopwords:
        values = [value for value in values if value not in STOPWORDS and len(value) > 1 and not value.isdigit()]
    return values


def unique_terms(text: str) -> list[str]:
    return list(dict.fromkeys(terms(text)))


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if len(part.strip()) >= 30]


def chunk_page(text: str, target_chars: int = 2400, overlap_chars: int = 300) -> Iterable[str]:
    """Yield readable chunks while retaining limited cross-chunk context."""
    text = normalize_text(text)
    if not text:
        return
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) == 1:
        paragraphs = split_sentences(text) or [text]

    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > target_chars * 2:
            pieces = [paragraph[i : i + target_chars] for i in range(0, len(paragraph), target_chars)]
        else:
            pieces = [paragraph]
        for piece in pieces:
            candidate = f"{current}\n{piece}".strip() if current else piece
            if current and len(candidate) > target_chars:
                yield current
                tail = current[-overlap_chars:].lstrip()
                current = f"{tail}\n{piece}".strip()
            else:
                current = candidate
    if current:
        yield current
