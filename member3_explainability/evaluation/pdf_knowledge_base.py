"""
member3_explainability/evaluation/pdf_knowledge_base.py
=========================================================
Real, document-backed RAG source layered on top of knowledge_base.py.

Where knowledge_base.py holds 8 hand-curated one-line facts, this module
actually opens the PDFs in evaluation/papers/, extracts their text, and
chunks them into short retrievable passages -- the "upload documents, the
system reads them and retrieves from them" flavour of RAG.

Each source PDF carries its citation + retrieval tags in real PDF metadata
(Title / Author / Subject), set when the PDF was authored, which this module
reads back via pypdf rather than re-typing citations in Python. Four of the
six PDFs contain verbatim quoted excerpts from the actual published paper
(fetched from an open-access copy); the remaining two (Cao et al. 2014,
Russell 1980) contain a clearly-labelled curated summary because no
freely-accessible full text could be found -- see the "kind" label printed
on each PDF's own first page.

combined_retriever() merges these PDF-derived passages with the curated
Facts from knowledge_base.py into a single Retriever, so a query can surface
either a hand-verified one-liner or a real excerpt from the source paper,
whichever matches best.


"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

from member3_explainability.evaluation.knowledge_base import Fact, Retriever, KNOWLEDGE_BASE

PAPERS_DIR = Path(__file__).parent / "papers"

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?"“”\'])\s+(?=[A-Z"“‘\'])')
_CHUNK_SIZE = 2          # sentences per retrievable chunk
_MIN_CHUNK_CHARS = 40    # drop near-empty fragments


def _extract_text_and_meta(pdf_path: Path) -> Tuple[str, object]:
    reader = PdfReader(str(pdf_path))
    text = " ".join((page.extract_text() or "") for page in reader.pages)
    return re.sub(r"\s+", " ", text).strip(), reader.metadata


def _split_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) >= 20]


def _chunk(sentences: List[str], size: int = _CHUNK_SIZE) -> List[str]:
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), size)]


@lru_cache(maxsize=1)
def load_pdf_facts() -> List[Fact]:
    """Extract + chunk every PDF in evaluation/papers/ into retrievable Facts.

    Returns [] (not an error) if the papers/ folder is empty or missing, so
    the rest of the app degrades gracefully if no PDFs have been added yet.
    Malformed/unreadable PDFs are skipped rather than crashing the app.
    """
    facts: List[Fact] = []
    if not PAPERS_DIR.exists():
        return facts

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        try:
            text, meta = _extract_text_and_meta(pdf_path)
        except Exception:
            continue

        citation = (meta.author if meta and meta.author else pdf_path.stem)
        tags = (
            [t.strip() for t in meta.subject.split(",") if t.strip()]
            if meta and meta.subject else []
        )

        for i, chunk_text in enumerate(_chunk(_split_sentences(text))):
            if len(chunk_text) < _MIN_CHUNK_CHARS:
                continue
            facts.append(Fact(
                id=f"{pdf_path.stem}_{i}",
                text=chunk_text,
                tags=tags,
                citation=citation,
            ))
    return facts


def combined_retriever() -> Retriever:
    """A Retriever over BOTH the curated one-liner facts (knowledge_base.py)
    AND real chunked excerpts extracted from the PDFs in evaluation/papers/."""
    return Retriever(facts=KNOWLEDGE_BASE + load_pdf_facts())


# ---------------------------------------------------------------------------
# Smoke test:  python -m member3_explainability.evaluation.pdf_knowledge_base
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    facts = load_pdf_facts()
    print(f"Extracted {len(facts)} chunks from {len(list(PAPERS_DIR.glob('*.pdf')))} PDFs\n")
    for f in facts:
        print(f"[{f.id}]  ({f.citation})")
        print(f"  tags: {f.tags}")
        print(f"  {f.text[:160]}{'...' if len(f.text) > 160 else ''}")
        print()

    r = combined_retriever()
    print("=== sample retrieval: 'why is GSR contributing to stress' ===")
    for f in r.retrieve("why is GSR contributing to stress signal quality", k=3):
        print(f" -> [{f.id}] {f.citation}")
