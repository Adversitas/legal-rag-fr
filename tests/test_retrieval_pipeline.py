import pytest
from conftest import FakeLLM

from legalrag.chunking import chunk_corpus
from legalrag.evaluation import EvalItem, judge_answer, retrieval_metrics, summarize_generation
from legalrag.index import HybridIndex, rrf, tokenize
from legalrag.pipeline import RAGPipeline, parse_citations


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("Quelle est la durée du préavis ?") == ["duree", "preavis"]


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 2]])
    assert fused[0][0] == 1


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_finds_relevant_doc(docs, embedder, mode):
    index = HybridIndex.build(chunk_corpus(docs, 500, 0), embedder)
    top = index.search("combien de jours de congés payés par mois", k=2, mode=mode)
    assert top[0].chunk.doc_id == "F2"


def test_index_roundtrip(tmp_path, docs, embedder):
    index = HybridIndex.build(chunk_corpus(docs, 500, 0), embedder)
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, embedder)
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert loaded.search("carte d'identité", 1)[0].chunk.doc_id == "F3"


def test_index_load_rejects_mismatched_embedder(tmp_path, docs, embedder):
    HybridIndex.build(chunk_corpus(docs, 500, 0), embedder).save(tmp_path)
    embedder.name = "other-model"
    with pytest.raises(ValueError, match="Rebuild the index"):
        HybridIndex.load(tmp_path, embedder)


def test_parse_citations_ignores_out_of_range():
    assert parse_citations("Voir [1] et [3][2], pas [9].", n_sources=3) == [1, 2, 3]


def test_pipeline_passes_numbered_sources_and_parses_citations(docs, embedder):
    index = HybridIndex.build(chunk_corpus(docs, 500, 0), embedder)
    llm = FakeLLM("Le préavis est d'un mois [1].")
    pipe = RAGPipeline(index, llm, top_k=2)
    ans = pipe.answer("Quel préavis pour une démission ?")
    _, user_prompt = llm.calls[0]
    assert "[1] Préavis de démission" in user_prompt
    assert ans.cited == [1]
    assert ans.sources[0].chunk.doc_id == "F1"


def test_retrieval_metrics():
    m = retrieval_metrics(["F9", "F1", "F2"], ["F1", "F7"], k=3)
    assert m == {"hit@3": 1.0, "recall@3": 0.5, "mrr@3": 0.5}


def test_judge_and_summary(docs, embedder):
    index = HybridIndex.build(chunk_corpus(docs, 500, 0), embedder)
    pipe = RAGPipeline(index, FakeLLM("Un mois [1]."), top_k=2)
    judge = FakeLLM('Voici : {"claims_total": 2, "claims_supported": 1, "correctness": 4, "abstained": false}')
    item = EvalItem("q1", "Quel préavis pour une démission ?", ["F1"], reference="Un mois.")
    res = judge_answer(judge, item, pipe.answer(item.question))
    assert res.faithfulness == 0.5
    assert res.citation_valid == 1.0
    summary = summarize_generation([res])
    assert summary["correctness_1_5"] == 4
    assert summary["false_abstention_rate"] == 0.0
