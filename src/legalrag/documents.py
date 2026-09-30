from dataclasses import asdict, dataclass


@dataclass
class Document:
    id: str  # service-public fiche id, e.g. "F2258"
    title: str
    url: str
    text: str
    audience: str = ""  # e.g. "Particuliers"
    breadcrumb: str = ""  # theme path, e.g. "Travail > Congés"


@dataclass
class Chunk:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    url: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
