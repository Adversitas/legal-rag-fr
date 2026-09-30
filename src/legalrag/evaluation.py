"""Evaluation harness.

Retrieval is scored at the *document* level (a fiche counts as found if any of its
chunks is retrieved), which is what matters to a user following the source link.

Generation is scored by an LLM judge against the retrieved sources (faithfulness) and
a reference answer (correctness), plus two deterministic checks: citation validity and
abstention on questions the corpus cannot answer.
"""

import json
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .documents import Document, ScoredChunk
from .llm import LLM
from .pipeline import Answer, format_context


@dataclass
class EvalItem:
    id: str
    question: str
    relevant_docs: list[str]  # empty => the corpus cannot answer it; the system should abstain
    reference: str = ""
    kind: str = "factual"  # factual | situational | unanswerable
    source: str = "manual"  # manual | synthetic


def load_eval_set(path: Path) -> list[EvalItem]:
    with path.open(encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def save_eval_set(items: list[EvalItem], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- retrieval


def ranked_doc_ids(results: list[ScoredChunk]) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r.chunk.doc_id not in seen:
            seen.append(r.chunk.doc_id)
    return seen


def retrieval_metrics(ranked: list[str], relevant: list[str], k: int) -> dict[str, float]:
    top = ranked[:k]
    rel = set(relevant)
    hits = [d for d in top if d in rel]
    first = next((i for i, d in enumerate(top) if d in rel), None)
    return {
        f"hit@{k}": float(bool(hits)),
        f"recall@{k}": len(set(hits)) / len(rel),
        f"mrr@{k}": 0.0 if first is None else 1.0 / (first + 1),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(r[key] for r in rows) for key in rows[0]} if rows else {}


# --------------------------------------------------------------------------- generation

JUDGE_SYSTEM = """Tu évalues les réponses d'un assistant juridique RAG. Sois strict et factuel.
Réponds uniquement avec un objet JSON, sans texte autour."""

JUDGE_TEMPLATE = """Extraits fournis à l'assistant :
{context}

Question : {question}

Réponse de référence (peut être vide) : {reference}

Réponse de l'assistant :
{answer}

Évalue la réponse de l'assistant et renvoie ce JSON :
{{
  "claims_total": <nombre d'affirmations factuelles dans la réponse>,
  "claims_supported": <nombre de ces affirmations directement soutenues par les extraits>,
  "correctness": <1 à 5 : accord avec la réponse de référence sur le fond ; 3 si pas de référence>,
  "abstained": <true si l'assistant dit ne pas pouvoir répondre>
}}"""


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]!r}")
    return json.loads(match.group(0))


@dataclass
class GenerationResult:
    id: str
    question: str
    answer: str
    faithfulness: float | None
    correctness: float | None
    abstained: bool
    should_abstain: bool
    citation_valid: float  # share of cited sources that belong to a relevant fiche
    retrieved_docs: list[str] = field(default_factory=list)


def judge_answer(judge: LLM, item: EvalItem, answer: Answer) -> GenerationResult:
    raw = judge.generate(
        JUDGE_SYSTEM,
        JUDGE_TEMPLATE.format(
            context=format_context(answer.sources),
            question=item.question,
            reference=item.reference or "(aucune)",
            answer=answer.text,
        ),
    )
    verdict = _parse_json(raw)
    total = int(verdict.get("claims_total", 0))
    faithfulness = int(verdict.get("claims_supported", 0)) / total if total else None

    cited_docs = [answer.sources[i - 1].chunk.doc_id for i in answer.cited]
    rel = set(item.relevant_docs)
    citation_valid = (
        sum(d in rel for d in cited_docs) / len(cited_docs) if cited_docs and rel else float(not cited_docs)
    )
    return GenerationResult(
        id=item.id,
        question=item.question,
        answer=answer.text,
        faithfulness=faithfulness,
        correctness=float(verdict["correctness"]) if item.reference else None,
        abstained=bool(verdict.get("abstained", answer.abstained)),
        should_abstain=not item.relevant_docs,
        citation_valid=citation_valid,
        retrieved_docs=ranked_doc_ids(answer.sources),
    )


def summarize_generation(results: list[GenerationResult]) -> dict[str, float]:
    def avg(values):
        values = [v for v in values if v is not None]
        return statistics.fmean(values) if values else float("nan")

    answerable = [r for r in results if not r.should_abstain]
    unanswerable = [r for r in results if r.should_abstain]
    return {
        "faithfulness": avg(r.faithfulness for r in answerable),
        "correctness_1_5": avg(r.correctness for r in answerable),
        "citation_precision": avg(r.citation_valid for r in answerable if not r.abstained),
        "false_abstention_rate": avg(float(r.abstained) for r in answerable),
        "correct_abstention_rate": avg(float(r.abstained) for r in unanswerable),
    }


# --------------------------------------------------------------------------- synthetic data

QGEN_SYSTEM = """Tu crées un jeu d'évaluation pour un assistant sur les démarches administratives \
françaises. Réponds uniquement avec un objet JSON."""

QGEN_TEMPLATE = """Voici une fiche service-public.fr :

Titre : {title}
{text}

Écris UNE question réaliste qu'un particulier pourrait poser et à laquelle cette fiche répond.
Contraintes :
- formule-la comme un vrai usager (langage courant, éventuellement sa situation personnelle) ;
- n'emploie PAS les mots exacts du titre, pour ne pas favoriser la recherche par mots-clés ;
- la réponse doit se trouver dans la fiche.

JSON attendu : {{"question": "...", "reference": "<réponse courte et exacte d'après la fiche>", \
"kind": "factual" ou "situational"}}"""


def generate_questions(
    llm: LLM, docs: list[Document], n: int, seed: int = 0, max_chars: int = 6000
) -> list[EvalItem]:
    rng = random.Random(seed)
    sample = rng.sample(docs, min(n, len(docs)))
    items = []
    for i, doc in enumerate(sample):
        raw = llm.generate(QGEN_SYSTEM, QGEN_TEMPLATE.format(title=doc.title, text=doc.text[:max_chars]))
        try:
            out = _parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        items.append(
            EvalItem(
                id=f"syn{i:04d}",
                question=out["question"],
                relevant_docs=[doc.id],
                reference=out.get("reference", ""),
                kind=out.get("kind", "factual"),
                source="synthetic",
            )
        )
    return items
