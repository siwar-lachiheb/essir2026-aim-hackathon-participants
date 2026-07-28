"""Load a PDF into the vector store.

    parse PDF (data/in) -> pages -> [chunk] -> embeddings -> Qdrant

By default there is no chunking (one vector per page) and embeddings come from a local
sentence-transformers model. Both are yours to improve (see chunking.py and embeddings.py).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import models

from ..config import get_settings
from ..models import IngestResponse
from ..vectorstore.qdrant_store import get_store
from .chunking import chunk_pages
from .embeddings import get_embedder

# A fixed namespace so re-ingesting the same document overwrites its points
# (idempotent ids) instead of duplicating them.
_NAMESPACE = uuid.UUID("6f0d9b1e-3b7a-4c2e-9a1d-000000000000")

ExtractorFn = Callable[[Path], list[str]]


def _find_pdf(filename: str | None) -> Path:
    in_dir = Path(get_settings().in_dir)
    if filename:
        path = in_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"no such PDF: {path}")
        return path
    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no *.pdf found in {in_dir}/ — put your document there first")
    return pdfs[0]


# ── pypdf (shipped by default; fine for clean digital PDFs) ──────────────────

def extract_pypdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


# ── pdfplumber (layout-aware, good for tables and columns) ────────────────────
#   pip install pdfplumber

def extract_pdfplumber(path: Path) -> list[str]:
    import pdfplumber
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page_obj in pdf.pages:
            pages.append(page_obj.extract_text() or "")
    return pages


# ── PyMuPDF / fitz (fast, good balance of speed and quality) ──────────────────
#   pip install PyMuPDF

def extract_pymupdf(path: Path) -> list[str]:
    import fitz
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


# ── Docling (IBM Research; heavy but very good on complex layout) ─────────────
#   pip install docling

def extract_docling(path: Path) -> list[str]:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document
    pages: list[str] = []
    for page_num in range(len(result.pages) if hasattr(result, "pages") else 1):
        text = doc.export_to_markdown(page_no=page_num) if hasattr(doc, "export_to_markdown") else doc.export_to_text()
        pages.append(text)
    return pages


# ── Marker (GPU-accelerated; excellent for scans and complex layout) ──────────
#   pip install marker-pdf

def extract_marker(path: Path) -> list[str]:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(path))
    return [rendered.markdown]


# ── GROBID (REST API; needs docker run -p 8070:8070 lfoppiano/grobid:0.8.0) ──
#   pip install httpx

def extract_grobid(path: Path) -> list[str]:
    import xml.etree.ElementTree as ET

    import httpx

    grobid_url = get_settings().grobid_url or "http://localhost:8070"
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    with open(path, "rb") as f:
        pdf_bytes = f.read()

    resp = httpx.post(
        f"{grobid_url}/api/processFulltextDocument",
        files={"input": (path.name, pdf_bytes, "application/pdf")},
        data={
            "consolidateHeader": "1",
            "segmentSentences": "1",
        },
        timeout=300,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    pages: list[str] = []
    for div in root.findall(".//tei:text//tei:div", ns):
        pb = div.find(".//tei:pb", ns)
        if pb is not None:
            text = "\n".join(
                el.text or ""
                for el in div.iter()
                if el.tag == f"{{{ns['tei']}}}p" and el.text
            )
            pages.append(text)
    return pages or [_tei_all_text(root, ns)]


def _tei_all_text(root: object, ns: dict[str, str]) -> str:
    return "\n".join(
        el.text or "" for el in root.iter() if el.tag == f"{{{ns['tei']}}}p" and el.text
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────

_EXTRACTORS: dict[str, ExtractorFn] = {
    "pypdf": extract_pypdf,
    "pdfplumber": extract_pdfplumber,
    "pymupdf": extract_pymupdf,
    "docling": extract_docling,
    "marker": extract_marker,
    "grobid": extract_grobid,
}


def extract_pages(path: Path, *, backend: str | None = None) -> list[str]:
    backend = backend or get_settings().extractor
    fn = _EXTRACTORS.get(backend)
    if fn is None:
        raise ValueError(f"unknown extractor {backend!r}; choose from {sorted(_EXTRACTORS)}")
    return fn(path)


def ingest(filename: str | None = None, reset: bool = False) -> IngestResponse:
    settings = get_settings()
    embedder = get_embedder()
    store = get_store()

    path = _find_pdf(filename)
    pages = extract_pages(path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"{path.name} produced no text — is it a scanned/image PDF?")

    # Embed in batches. is_query=False marks these as documents ("passage:" for e5).
    vectors: list[list[float]] = []
    batch = 32
    for i in range(0, len(chunks), batch):
        texts = [c.text for c in chunks[i : i + batch]]
        vectors.extend(embedder.embed(texts, is_query=False))

    store.ensure_collection(dim=len(vectors[0]), reset=reset)

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_NAMESPACE, f"{path.name}:{c.index}")),
            vector=vec,
            payload={"text": c.text, "page": c.page, "source": path.name},
        )
        for c, vec in zip(chunks, vectors)
    ]
    store.upsert(points)

    return IngestResponse(
        document=path.name,
        pages=len(pages),
        chunks=len(chunks),
        collection=settings.qdrant_collection,
    )
