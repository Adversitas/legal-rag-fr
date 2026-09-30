# Legal RAG FR — measured question answering over French public-service guides

Ask a question about French rights and procedures (*"Mon patron peut-il refuser mes congés en août ?"*)
and get a concise answer grounded in the official [service-public.fr](https://www.service-public.fr)
fiches, with numbered citations linking back to the source.

The point of this project is not the chatbot. It's the **evaluation**: every design choice
(chunking, BM25 vs dense vs hybrid retrieval, reranking, Claude vs a local model) is backed by numbers
from a reproducible eval harness.

> Status: work in progress. Results tables below are filled in as experiments run.

## Architecture

```
service-public.fr XML export ──► parse fiches ──► structure-aware chunking
                                                  (title + section header on every chunk)
                                                          │
                                   ┌──────────────────────┴──────────────────┐
                                   ▼                                         ▼
                           BM25 (accent-folded,                  multilingual-e5 embeddings
                           French stopwords)                      (numpy, cosine)
                                   └──────────► Reciprocal Rank Fusion ◄──────┘
                                                          │
                                            optional cross-encoder rerank
                                                          │
                                        top-k chunks, numbered [1]..[k]
                                                          │
                              LLM (Claude via Anthropic SDK  |  local model via LM Studio/Ollama)
                                                          │
                                        answer with [n] citations ──► FastAPI /ask
```

## Data

The *Fiches pratiques et ressources de Service-Public.fr* open-data export (Licence Ouverte / Etalab 2.0),
published by DILA. Only the "Particuliers" fiches (`F*.xml`) are indexed.

## Evaluation

| Layer | Metric | How |
|---|---|---|
| Retrieval | hit@k, recall@k, MRR@k (document level) | eval set with gold fiche IDs, no LLM calls |
| Generation | faithfulness | LLM judge: share of claims supported by the retrieved extracts |
| Generation | correctness (1–5) | LLM judge vs. reference answer |
| Generation | citation precision | deterministic: cited sources that belong to a gold fiche |
| Generation | abstention | the system should say "I don't know" on out-of-scope questions, and only then |

**Eval set.** Questions are drafted by an LLM from randomly sampled fiches, with instructions to use
everyday language and avoid the title's wording (otherwise the set would unfairly favour keyword search).
Drafts are then reviewed by hand, and a set of manually written and out-of-scope questions is added.
Synthetic and manual questions are reported separately.

### Results

**Retrieval, hand-written set (25 answerable questions, document level).** Embeddings:
`multilingual-e5-small`, 53,625 chunks from 2,918 fiches.

| Method | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.40 | 0.68 | 0.84 | 0.56 |
| Dense | 0.60 | 0.72 | 0.80 | 0.68 |
| **Hybrid (RRF)** | **0.64** | **0.84** | **0.92** | **0.76** |

With 25 questions each one is worth 4 points, so small gaps are noise. The larger synthetic set
will tighten these numbers. Hybrid still wins at every cutoff: BM25 ranks the right fiche low
when users paraphrase, and dense misses exact legal terms that BM25 catches.

**Where it fails: everyday vs. legal vocabulary.** All four misses are vocabulary gaps:
"vacances" for *congés payés*, "carte grise" for *certificat d'immatriculation*, and "caution"
for *dépôt de garantie*. The last one is a trap: in legal French *caution* means a guarantor, so
the system confidently retrieves the guarantor fiche. Candidate fixes, to be measured: a
cross-encoder reranker, LLM query rewriting into administrative vocabulary, and a larger
embedding model.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # source .venv/bin/activate on Linux/macOS
pip install -e ".[ml,dev]"
cp .env.example .env                             # set the LLM provider / keys

legalrag download          # fetch + parse the open-data export -> data/documents.jsonl
legalrag build-index       # chunk + embed -> data/index/
legalrag ask "Combien de temps dure le préavis quand on démissionne ?"

legalrag eval-retrieval --rerank     # retrieval comparison table, free
legalrag eval-generation --limit 50  # end-to-end, uses API calls

uvicorn legalrag.api:app --reload    # POST /ask {"question": "..."}
```

Using a local model instead of Claude: start LM Studio's server, then set
`RAG_LLM_PROVIDER=openai_compat` and `RAG_LLM_MODEL=<model id>` in `.env`.

GPU jobs in Docker (embedding ~54k chunks takes minutes on a GPU vs. ~1.5 h on CPU).
The project folder is mounted, so code changes don't need a rebuild:

```bash
docker build -f Dockerfile.gpu -t legal-rag-fr:gpu .
docker run --rm --gpus all -v .:/app -v hf-cache:/root/.cache/huggingface legal-rag-fr:gpu legalrag build-index
docker run --rm --gpus all -v .:/app -v hf-cache:/root/.cache/huggingface legal-rag-fr:gpu legalrag eval-retrieval --eval-set eval/questions_manual.jsonl --rerank
```

Serving the API in Docker:

```bash
docker build -t legal-rag-fr .
docker run -p 8000:8000 -v ./data:/app/data --env-file .env legal-rag-fr
```

## Project layout

```
src/legalrag/
  ingest.py       XML export -> Documents
  chunking.py     heading-aware chunking with overlap
  index.py        BM25 + dense + RRF hybrid index
  rerank.py       cross-encoder reranker
  llm.py          Anthropic / OpenAI-compatible backends
  pipeline.py     retrieve -> prompt -> answer with citations
  evaluation.py   metrics, LLM judge, synthetic question generation
  api.py, cli.py
eval/             eval sets and saved runs
tests/            unit tests (no network, no model downloads)
```

## Limitations

- Legal information, not legal advice. Fiches are updated regularly; re-run `download` to refresh.
- The LLM judge is itself a model, so its scores are spot-checked by hand.
