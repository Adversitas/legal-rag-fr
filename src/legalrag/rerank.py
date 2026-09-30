from .documents import ScoredChunk
from .embeddings import best_device


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs jointly. Slower than bi-encoder retrieval but more precise,
    so it is only applied to the small candidate pool."""

    def __init__(self, model_name: str, batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name
        self.device = best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                enc = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
        return scores

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self.score(query, [c.chunk.text for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)[:k]
        return [ScoredChunk(c.chunk, float(s)) for c, s in ranked]
