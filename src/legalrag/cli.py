import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import typer

from .config import Settings

app = typer.Typer(help="French legal-guide RAG: ingest, index, ask, evaluate.", no_args_is_help=True)


def _settings() -> Settings:
    return Settings()


def _embedder(s: Settings):
    from .embeddings import TransformerEmbedder

    return TransformerEmbedder(s.embedding_model)


def _load_index(s: Settings):
    from .index import HybridIndex

    return HybridIndex.load(s.index_dir, _embedder(s))


@app.command()
def download():
    """Download the service-public.fr open-data export and parse fiches to data/documents.jsonl."""
    from .ingest import download as dl
    from .ingest import load_corpus, save_documents

    s = _settings()
    xml_dir = dl(s.corpus_url, s.raw_dir)
    docs = load_corpus(xml_dir)
    save_documents(docs, s.data_dir / "documents.jsonl")
    typer.echo(f"Parsed {len(docs)} fiches -> {s.data_dir / 'documents.jsonl'}")


@app.command("build-index")
def build_index():
    """Chunk documents, embed them, and save the hybrid index."""
    from .chunking import chunk_corpus
    from .index import HybridIndex
    from .ingest import read_documents

    s = _settings()
    docs = read_documents(s.data_dir / "documents.jsonl")
    chunks = chunk_corpus(docs, s.chunk_size, s.chunk_overlap)
    typer.echo(f"{len(docs)} documents -> {len(chunks)} chunks; embedding with {s.embedding_model}")
    HybridIndex.build(chunks, _embedder(s)).save(s.index_dir)
    typer.echo(f"Index saved to {s.index_dir}")


@app.command()
def search(
    question: str,
    k: int = 5,
    mode: str = typer.Option("hybrid", help="bm25 | dense | hybrid"),
):
    """Show the passages retrieval finds for a question. No LLM needed."""
    s = _settings()
    for i, res in enumerate(_load_index(s).search(question, k, mode), 1):
        header, _, body = res.chunk.text.partition("\n\n")
        typer.echo(f"\n[{i}] {header}\n    {res.chunk.url}")
        excerpt = " ".join(body.split())
        typer.echo(f"    {excerpt[:300]}{'…' if len(excerpt) > 300 else ''}")


@app.command()
def ask(question: str, show_sources: bool = True):
    """Answer a question with citations."""
    from .llm import make_llm
    from .pipeline import RAGPipeline

    s = _settings()
    pipe = RAGPipeline.from_settings(s, _load_index(s), make_llm(s))
    ans = pipe.answer(question)
    typer.echo(ans.text)
    if show_sources:
        typer.echo("\nSources :")
        for i, src in enumerate(ans.sources, 1):
            mark = "*" if i in ans.cited else " "
            typer.echo(f" {mark}[{i}] {src.chunk.title} - {src.chunk.url}")


@app.command("gen-questions")
def gen_questions(n: int = 150, out: Path = Path("eval/questions_synthetic.jsonl"), seed: int = 0):
    """Draft synthetic eval questions from random fiches (review them by hand before trusting them)."""
    from .evaluation import generate_questions, save_eval_set
    from .ingest import read_documents
    from .llm import make_llm

    s = _settings()
    docs = read_documents(s.data_dir / "documents.jsonl")
    items = generate_questions(make_llm(s), docs, n, seed)
    save_eval_set(items, out)
    typer.echo(f"Wrote {len(items)} questions to {out}")


@app.command("eval-retrieval")
def eval_retrieval(
    eval_set: Path = Path("eval/questions.jsonl"),
    k: list[int] = typer.Option([1, 3, 5, 10]),
    rerank: bool = False,
):
    """Compare BM25 / dense / hybrid (/ +rerank) retrieval on the eval set. No LLM calls."""
    from .evaluation import load_eval_set, mean_metrics, ranked_doc_ids, retrieval_metrics
    from .rerank import CrossEncoderReranker

    s = _settings()
    index = _load_index(s)
    items = [it for it in load_eval_set(eval_set) if it.relevant_docs]
    configs: list[tuple[str, object]] = [("bm25", None), ("dense", None), ("hybrid", None)]
    if rerank:
        configs.append(("hybrid+rerank", CrossEncoderReranker(s.reranker_model)))

    depth = max(k)
    report = {}
    for name, reranker in configs:
        mode = name.split("+")[0]
        rows = []
        for it in items:
            if reranker:
                results = reranker.rerank(it.question, index.search(it.question, s.candidates, mode), depth)
            else:
                results = index.search(it.question, depth * 3, mode)  # several chunks can share a doc
            ranked = ranked_doc_ids(results)
            row = {}
            for kk in k:
                row.update(retrieval_metrics(ranked, it.relevant_docs, kk))
            rows.append(row)
        report[name] = mean_metrics(rows)

    cols = list(next(iter(report.values())).keys())
    typer.echo(f"{len(items)} answerable questions\n")
    typer.echo(f"{'config':<15}" + "".join(f"{c:>11}" for c in cols))
    for name, m in report.items():
        typer.echo(f"{name:<15}" + "".join(f"{m[c]:>11.3f}" for c in cols))
    _save_run("retrieval", {"eval_set": str(eval_set), "n": len(items), "results": report})


@app.command("eval-generation")
def eval_generation(
    eval_set: Path = Path("eval/questions.jsonl"),
    judge_model: str = typer.Option("claude-opus-5-5", help="Judge runs on the Anthropic API."),
    limit: int = 0,
):
    """Answer every eval question and score it with an LLM judge. Costs API calls."""
    from .evaluation import judge_answer, load_eval_set, summarize_generation
    from .llm import AnthropicLLM, make_llm
    from .pipeline import RAGPipeline

    s = _settings()
    pipe = RAGPipeline.from_settings(s, _load_index(s), make_llm(s))
    judge = AnthropicLLM(judge_model, effort="medium")
    items = load_eval_set(eval_set)
    if limit:
        items = items[:limit]

    results = []
    with typer.progressbar(items, label="Answering + judging") as bar:
        for it in bar:
            results.append(judge_answer(judge, it, pipe.answer(it.question)))

    summary = summarize_generation(results)
    for key, val in summary.items():
        typer.echo(f"{key:<26}{val:.3f}")
    _save_run(
        "generation",
        {
            "eval_set": str(eval_set),
            "generator": f"{s.llm_provider}:{s.llm_model}",
            "judge": judge_model,
            "retrieval_mode": s.retrieval_mode,
            "reranker": s.use_reranker,
            "summary": summary,
            "items": [asdict(r) for r in results],
        },
    )


def _save_run(kind: str, payload: dict) -> None:
    out_dir = Path("eval/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"\nSaved {path}")


if __name__ == "__main__":
    app()
