"""HTTP API. Run with: uvicorn legalrag.api:app --reload"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .embeddings import TransformerEmbedder
from .index import HybridIndex
from .llm import LLMError, make_llm
from .pipeline import RAGPipeline

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = Settings()
    index = HybridIndex.load(s.index_dir, TransformerEmbedder(s.embedding_model))
    state["pipeline"] = RAGPipeline.from_settings(s, index, make_llm(s))
    state["settings"] = s
    yield
    state.clear()


app = FastAPI(title="Legal RAG FR", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    url: str
    cited: bool
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    s: Settings = state["settings"]
    return {"status": "ok", "llm": f"{s.llm_provider}:{s.llm_model}", "mode": s.retrieval_mode}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        ans = state["pipeline"].answer(req.question)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return AskResponse(
        answer=ans.text,
        sources=[
            Source(
                n=i,
                doc_id=s.chunk.doc_id,
                title=s.chunk.title,
                url=s.chunk.url,
                cited=i in ans.cited,
                excerpt=s.chunk.text[:400],
            )
            for i, s in enumerate(ans.sources, 1)
        ],
    )
