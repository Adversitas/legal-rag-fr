"""Structure-aware chunking.

Splits on headings first so a chunk rarely straddles two sections, then packs
paragraphs up to `chunk_size` characters. Each chunk is prefixed with the fiche
title and its current section heading so it stays self-explanatory once retrieved
out of context.
"""

import re

from .documents import Chunk, Document


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split on markdown headings; each section is labelled with its full heading path
    (e.g. "50 salariés ou plus › Le CSE a-t-il un budget ?") so conditions are never lost."""
    path: list[str] = []
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(#+)\s+(.*)", line)
        if match:
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2)]
            sections.append((" › ".join(path), []))
        else:
            sections[-1][1].append(line)
    return [(h, paras) for h, paras in sections if paras]


def _split_long(para: str, size: int) -> list[str]:
    """Hard-split a single paragraph that is longer than the chunk size, on sentence ends."""
    if len(para) <= size:
        return [para]
    pieces, current = [], ""
    for sentence in para.replace(". ", ".\n").splitlines():
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: Document, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(heading: str, body: list[str]) -> None:
        header = doc.title if not heading else f"{doc.title} — {heading}"
        chunks.append(
            Chunk(
                id=f"{doc.id}#{len(chunks)}",
                doc_id=doc.id,
                title=doc.title,
                url=doc.url,
                text=f"{header}\n\n" + "\n".join(body),
            )
        )

    for heading, paras in _sections(doc.text):
        paras = [p for para in paras for p in _split_long(para, chunk_size)]
        current: list[str] = []
        length = 0
        for para in paras:
            if current and length + len(para) > chunk_size:
                emit(heading, current)
                # carry trailing paragraphs forward as overlap
                tail: list[str] = []
                tail_len = 0
                for p in reversed(current):
                    if tail_len + len(p) > overlap:
                        break
                    tail.insert(0, p)
                    tail_len += len(p)
                current, length = tail, tail_len
            current.append(para)
            length += len(para)
        if current:
            emit(heading, current)
    return chunks


def chunk_corpus(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
