"""Compare PDF text extractors side-by-side on a given PDF.

Usage:
    python scripts/compare_extractors.py path/to/document.pdf

For each installed extractor this prints:
  1. Summary stats (chars, lines, words) per page
  2. A side-by-side preview of page 1 (first 500 chars)
  3. Full output saved to data/out/extractor-comparison/<name>.txt

Install extractors you want to try:
    uv sync --group pdfplumber
    uv sync --group pymupdf
    uv sync --group docling
    uv sync --group marker-pdf
    uv sync --group all-extractors
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


def _install_error(name: str, pip_pkg: str) -> str:
    return f"{name}: missing — pip install {pip_pkg} (or uv sync --group {name})"


def _try_extractor(name: str, path: Path, pip_pkg: str, group: str) -> list[str] | str:
    try:
        if name == "pypdf":
            from app.rag.ingest import extract_pypdf as fn
        elif name == "pdfplumber":
            from app.rag.ingest import extract_pdfplumber as fn
        elif name == "pymupdf":
            from app.rag.ingest import extract_pymupdf as fn
        elif name == "docling":
            from app.rag.ingest import extract_docling as fn
        elif name == "marker":
            from app.rag.ingest import extract_marker as fn
        elif name == "grobid":
            from app.rag.ingest import extract_grobid as fn
        else:
            return f"{name}: unknown extractor"

        pages = fn(path)
        if not any(p.strip() for p in pages):
            return f"{name}: extracted 0 characters (empty output)"
        return pages
    except ImportError:
        return _install_error(name, pip_pkg)
    except Exception as exc:  # noqa: BLE001
        return f"{name}: {type(exc).__name__}: {exc}"


def _show_preview(pages: list[str], label: str, chars: int = 500) -> None:
    text = pages[0] if pages else ""
    preview = text[:chars].replace("\n", "\\n").replace("\t", "\\t")
    print(f"  [{label:>12}] {preview}{'…' if len(text) > chars else ''}")


def _save_output(out_dir: Path, name: str, pages: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    out_path.write_text(
        "\n\n".join(pages),
        encoding="utf-8",
    )
    print(f"  [{name:>12}] full output → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main(pdf_path: Path) -> None:
    if not pdf_path.is_file():
        print(f"error: not a file — {pdf_path}")
        sys.exit(1)

    extractors: list[tuple[str, str, str]] = [
        ("pypdf", "pypdf", "pypdf"),
        ("pdfplumber", "pdfplumber", "pdfplumber"),
        ("pymupdf", "PyMuPDF", "pymupdf"),
        ("docling", "docling", "docling"),
        ("marker", "marker-pdf", "marker-pdf"),
        ("grobid", "httpx", "grobid"),
    ]

    out_dir = Path("data/out/extractor-comparison")

    print(f"\n{'='*70}")
    print(f"PDF: {pdf_path.name}")
    print(f"{'='*70}\n")

    for name, pip_pkg, group in extractors:
        print(f"── {name} ──")
        result = _try_extractor(name, pdf_path, pip_pkg, group)

        if isinstance(result, str):
            print(f"  {result}\n")
            continue

        pages = result
        total_chars = sum(len(p) for p in pages)
        total_lines = sum(p.count("\n") + 1 for p in pages)
        total_words = sum(len(p.split()) for p in pages)

        print(f"  pages: {len(pages):>4}  chars: {total_chars:>8,}  "
              f"lines: {total_lines:>6,}  words: {total_words:>6,}")

        _show_preview(pages, name)
        _save_output(out_dir, name, pages)
        print()

    print("Tip: compare the files in data/out/extractor-comparison/")
    print("     to find which extractor reads your document best.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: python {sys.argv[0]} path/to/document.pdf")
        sys.exit(1)
    main(Path(sys.argv[1]))
