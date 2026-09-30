import re
from dataclasses import dataclass, field

from .config import Settings
from .documents import ScoredChunk
from .index import HybridIndex, Mode
from .llm import LLM
from .rerank import CrossEncoderReranker

SYSTEM_PROMPT = """Tu es un assistant qui répond aux questions sur les droits et démarches \
administratives en France, en t'appuyant uniquement sur les extraits de fiches service-public.fr \
fournis.

Règles :
- Réponds en français, de façon claire et concise.
- Appuie chaque affirmation sur les extraits et cite-les avec leur numéro entre crochets, \
par exemple [1] ou [2][3].
- Si les extraits ne permettent pas de répondre, dis-le explicitement plutôt que de deviner, \
et indique quelle information manque.
- Les règles dépendent souvent de la situation (salarié du privé ou fonctionnaire, \
ancienneté, région...). Si c'est le cas, précise-le.
- Tu ne remplaces pas un conseil juridique personnalisé ; ne le rappelle que si la question \
porte sur un litige ou une situation individuelle complexe."""

NO_ANSWER_MARKERS = ("ne permettent pas", "ne contiennent pas", "pas d'information", "je ne sais pas")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[ScoredChunk]
    cited: list[int] = field(default_factory=list)  # 1-based indices into sources

    @property
    def abstained(self) -> bool:
        lowered = self.text.lower()
        return any(m in lowered for m in NO_ANSWER_MARKERS)


def format_context(sources: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{i}] {s.chunk.title} ({s.chunk.url})\n{s.chunk.text}" for i, s in enumerate(sources, 1)
    )


def parse_citations(text: str, n_sources: int) -> list[int]:
    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in found if 1 <= i <= n_sources)


class RAGPipeline:
    def __init__(
        self,
        index: HybridIndex,
        llm: LLM | None,
        reranker: CrossEncoderReranker | None = None,
        mode: Mode = "hybrid",
        candidates: int = 30,
        top_k: int = 5,
    ):
        self.index = index
        self.llm = llm
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates
        self.top_k = top_k

    @classmethod
    def from_settings(cls, settings: Settings, index: HybridIndex, llm: LLM | None) -> "RAGPipeline":
        reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
        return cls(index, llm, reranker, settings.retrieval_mode, settings.candidates, settings.top_k)

    def retrieve(self, question: str) -> list[ScoredChunk]:
        if self.reranker:
            pool = self.index.search(question, self.candidates, self.mode)
            return self.reranker.rerank(question, pool, self.top_k)
        return self.index.search(question, self.top_k, self.mode)

    def answer(self, question: str) -> Answer:
        if self.llm is None:
            raise RuntimeError("No LLM configured; use retrieve() for retrieval-only runs.")
        sources = self.retrieve(question)
        user = f"Extraits :\n\n{format_context(sources)}\n\nQuestion : {question}"
        text = self.llm.generate(SYSTEM_PROMPT, user)
        return Answer(question, text, sources, parse_citations(text, len(sources)))
