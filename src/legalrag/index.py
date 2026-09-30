"""Hybrid lexical + dense index.

BM25 catches exact legal vocabulary ("préavis", "RSA", article numbers) that small
embedding models blur; dense vectors catch paraphrases ("virée" -> "licenciement").
Rankings are merged with Reciprocal Rank Fusion, which needs no score calibration.

At a few thousand fiches a brute-force numpy dot product is fast enough; swap in
FAISS / a vector DB only if the corpus grows by orders of magnitude.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from .documents import Chunk, ScoredChunk
from .embeddings import Embedder

Mode = Literal["bm25", "dense", "hybrid"]

_STOPWORDS = set(
    """au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me
    même mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes
    toi ton tu un une vos votre vous c d j l m n s t y est sont été être a ai as avons avez ont
    cette cet si comment quel quelle quels quelles quoi dois doit peut puis faut""".split()
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS and len(t) > 1]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked lists of item indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "HybridIndex":
        vectors = embedder.embed_documents([c.text for c in chunks])
        return cls(chunks, vectors, embedder)

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def _dense_ranking(self, query: str, n: int) -> list[int]:
        q = self.embedder.embed_query(query)
        scores = self.vectors @ q
        return list(np.argsort(-scores)[:n])

    def search(self, query: str, k: int = 10, mode: Mode = "hybrid") -> list[ScoredChunk]:
        if mode == "bm25":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._bm25_ranking(query, k))]
        elif mode == "dense":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self._dense_ranking(query, k))]
        else:
            pool = max(k, 50)
            ranked = rrf([self._bm25_ranking(query, pool), self._dense_ranking(query, pool)])[:k]
        return [ScoredChunk(self.chunks[i], float(s)) for i, s in ranked]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps({"embedding_model": self.embedder.name, "n_chunks": len(self.chunks)})
        )

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "HybridIndex":
        meta = json.loads((directory / "meta.json").read_text())
        if meta["embedding_model"] != embedder.name:
            raise ValueError(
                f"Index was built with {meta['embedding_model']!r}, "
                f"but the configured embedder is {embedder.name!r}. Rebuild the index."
            )
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(directory / "vectors.npy"), embedder)
