"""Download and parse the service-public.fr "Vos droits et démarches" open-data export.

The export (Licence Ouverte / Etalab) is a zip of XML files. Files named F*.xml are
"fiches" - the practical guides we index. N*.xml (theme nodes) and R*.xml (resources)
are skipped, as are "Recherche guidée" fiches, which are interactive decision trees.

Most fiches are *conditional*: the same question has different answers depending on
the reader's situation (Situation > Chapitre > SousChapitre > Cas). The extractor keeps
that nesting as markdown heading levels so the chunker can attach the full condition
path to every chunk.
"""

import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import httpx
from lxml import etree

from .documents import Document

BASE_URL = "https://www.service-public.gouv.fr/particuliers/vosdroits/"

# Containers whose <Titre> is a section heading; each one nests one level deeper
_STRUCTURAL = {"Situation", "Chapitre", "SousChapitre", "Cas"}
# Callout boxes: their <Titre> ("Attention", "À noter", "Exemple") becomes an inline label
_CALLOUTS = {"Attention", "ANoter", "ASavoir", "Exemple"}
# Subtrees that are links, online services or addresses rather than explanatory content
_SKIP = {"ServiceEnLigne", "OuSAdresser", "Source", "RessourceWeb", "PivotLocal", "Complement"}
_SKIP_TYPES = {"Recherche guidée"}


def download(url: str, dest_dir: Path) -> Path:
    """Download and unzip the export into dest_dir. Returns the directory of XML files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "vosdroits.zip"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for block in r.iter_bytes():
                f.write(block)
    xml_dir = dest_dir / "xml"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(xml_dir)
    return xml_dir


def _local(tag) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _text(elem) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def _extract(elem, out: list[str], level: int = 0, prefix: str = "") -> None:
    """Walk the content tree, emitting one line per paragraph and '#'-headings for sections."""
    name = _local(elem.tag)
    if name in _SKIP:
        return

    if name in _STRUCTURAL:
        level += 1
        heading = ""
        for child in elem:
            if _local(child.tag) == "Titre":
                title = _text(child)
                if title:
                    heading = f"{'#' * min(level, 6)} {title}"
                    out.append(heading)
                continue
            start = len(out)
            _extract(child, out, level)
            # A nested section (e.g. a Cas) just closed: re-open ours so the text that
            # follows is not attributed to the last nested case. Empty re-openings are
            # dropped by the chunker.
            if heading and any(line.startswith("#") for line in out[start:]):
                out.append(heading)
        return

    if name in _CALLOUTS:
        label = next((_text(c) for c in elem if _local(c.tag) == "Titre"), name)
        start = len(out)
        for child in elem:
            if _local(child.tag) != "Titre":
                _extract(child, out, level)
        if len(out) > start:
            out[start] = f"{label} : {out[start]}"
        return

    if name == "Tableau":
        for row in elem.iter("{*}Rangée", "Rangée"):
            cells = [_text(c) for c in row if _local(c.tag) == "Cellule"]
            if any(cells):
                out.append(" | ".join(cells))
        return

    if name in {"Paragraphe", "TitreFlottant"}:
        text = _text(elem)
        if text:
            out.append(prefix + text)
        return

    if name == "Item":
        start = len(out)
        for child in elem:
            _extract(child, out, level)
        if len(out) > start:
            out[start] = "- " + out[start]
        return

    if name == "Titre":  # a title outside any known container: keep it as plain text
        text = _text(elem)
        if text:
            out.append(text)
        return

    for child in elem:
        _extract(child, out, level)


def parse_fiche(path: Path) -> Document | None:
    root = etree.parse(str(path)).getroot()
    doc_id = root.get("ID") or path.stem
    if not doc_id.startswith("F") or root.get("type") in _SKIP_TYPES:
        return None

    title = description = ""
    audiences: list[str] = []
    crumbs: list[str] = []
    body: list[str] = []
    for el in root:
        name = _local(el.tag)
        if name == "title":
            title = _text(el)
        elif name == "description":
            description = _text(el)
        elif name == "Audience":
            audiences.append(_text(el))
        elif name == "FilDAriane":
            # drop the "Accueil particuliers" root and the fiche itself
            crumbs = [_text(n) for n in el][1:-1]
        elif name in {"Introduction", "Texte", "ListeSituations"}:
            _extract(el, body)

    if not title or not body:
        return None
    text = "\n".join(([description] if description else []) + body)
    return Document(
        id=doc_id,
        title=title,
        url=root.get("spUrl") or BASE_URL + doc_id,
        text=text,
        audience=", ".join(audiences),
        breadcrumb=" > ".join(crumbs),
    )


def load_corpus(xml_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(xml_dir.rglob("F*.xml")):
        try:
            doc = parse_fiche(path)
        except etree.XMLSyntaxError:
            continue
        if doc:
            docs.append(doc)
    return docs


def save_documents(docs: Iterable[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")


def read_documents(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f]
