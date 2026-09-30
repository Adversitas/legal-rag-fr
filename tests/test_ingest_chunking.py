from itertools import pairwise
from pathlib import Path

from legalrag.chunking import chunk_document
from legalrag.documents import Document
from legalrag.ingest import parse_fiche

FIXTURE = Path(__file__).parent / "fixtures" / "F0001.xml"


def test_parse_fiche_extracts_metadata():
    doc = parse_fiche(FIXTURE)
    assert doc.id == "F0001"
    assert doc.title == "Préavis de démission d'un salarié du secteur privé"
    assert doc.url == "https://www.service-public.gouv.fr/particuliers/vosdroits/F0001"
    assert doc.audience == "Particuliers"
    assert doc.breadcrumb == "Travail - Formation > Démission"


def test_parse_fiche_keeps_conditional_structure():
    lines = parse_fiche(FIXTURE).text.splitlines()
    assert lines[0] == "Le salarié qui démissionne doit en principe respecter un préavis."
    assert "# Salarié en CDI" in lines
    assert "## Quelle est la durée du préavis ?" in lines
    assert "### Cadre" in lines
    assert "Le préavis est souvent d'1 mois." in lines  # inline markup flattened
    assert "Attention : Un préavis non effectué peut donner lieu à une indemnité." in lines
    assert "- Premier point de liste." in lines
    assert "Moins de 6 mois | 1 semaine" in lines
    assert not any("Ne doit pas apparaître" in line for line in lines)


def test_chunks_carry_full_condition_path():
    chunks = chunk_document(parse_fiche(FIXTURE), chunk_size=1000, overlap=0)
    headers = [c.text.split("\n\n", 1)[0] for c in chunks]
    title = "Préavis de démission d'un salarié du secteur privé"
    assert f"{title} — Salarié en CDI › Quelle est la durée du préavis ? › Cadre" in headers
    assert f"{title} — Salarié en CDD › Peut-on démissionner ?" in headers
    # the callout after the BlocCas belongs to the chapter again, not to the last Cas
    attention = next(c for c in chunks if "Attention" in c.text)
    assert attention.text.startswith(f"{title} — Salarié en CDI › Quelle est la durée du préavis ?\n")
    assert [c.id for c in chunks] == [f"F0001#{i}" for i in range(len(chunks))]


def test_chunk_size_is_respected_and_overlap_carries_context():
    paras = [f"Paragraphe numéro {i} " + "x" * 80 for i in range(20)]
    doc = Document("F9", "Titre", "u", "\n".join(paras))
    chunks = chunk_document(doc, chunk_size=400, overlap=120)
    bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
    assert all(len(b) <= 400 + 120 for b in bodies)
    # the last paragraph of chunk n reappears at the start of chunk n+1
    for a, b in pairwise(bodies):
        assert a.splitlines()[-1] == b.splitlines()[0]
    joined = "\n".join(bodies)
    assert all(p in joined for p in paras)


def test_very_long_paragraph_is_split():
    long_para = ". ".join(f"Phrase {i} assez longue pour remplir" for i in range(100))
    chunks = chunk_document(Document("F8", "T", "u", long_para), chunk_size=300, overlap=0)
    assert len(chunks) > 1
