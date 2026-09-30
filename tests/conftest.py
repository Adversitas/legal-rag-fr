import hashlib

import numpy as np
import pytest

from legalrag.documents import Document
from legalrag.index import tokenize


class HashEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""

    name = "hash-test"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def docs():
    return [
        Document("F1", "Préavis de démission", "u1", "Le salarié qui démissionne respecte un préavis d'un mois."),
        Document("F2", "Congés payés", "u2", "Le salarié acquiert 2,5 jours de congés payés par mois travaillé."),
        Document("F3", "Carte d'identité", "u3", "La carte nationale d'identité est valable 15 ans pour un majeur."),
    ]
