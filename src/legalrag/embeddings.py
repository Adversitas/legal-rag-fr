from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def best_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class TransformerEmbedder:
    """Mean-pooled, L2-normalised embeddings from a Hugging Face encoder (e.g. multilingual-e5).

    E5 models are trained with "query: " / "passage: " prefixes; omitting them
    measurably hurts retrieval, so they are added automatically for e5 models.
    """

    def __init__(self, model_name: str, batch_size: int = 64, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self.device = best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.batch_size = batch_size
        self.max_length = max_length
        self._e5 = "e5" in model_name.lower()
        self._torch = torch

    def _encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        torch = self._torch
        # sort by length so batches need little padding, then restore the original order
        order = np.argsort([-len(t) for t in texts])
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        batches = range(0, len(texts), self.batch_size)
        if progress:
            from tqdm import tqdm

            batches = tqdm(batches, desc="Embedding", unit="batch")
        with torch.inference_mode():
            for start in batches:
                idx = order[start : start + self.batch_size]
                enc = self.tokenizer(
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                out[idx] = pooled.cpu().numpy()
        return out

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [f"passage: {t}" if self._e5 else t for t in texts]
        return self._encode(texts, progress=len(texts) > 1000)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}" if self._e5 else text])[0]
