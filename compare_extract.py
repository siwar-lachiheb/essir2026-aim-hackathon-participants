import sys, pathlib

PDF = next(pathlib.Path("data/in").glob("*.pdf"))
PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # 0-indexed

out = pathlib.Path("scratch/extract"); out.mkdir(parents=True, exist_ok=True)

# pypdf (the baseline)
from pypdf import PdfReader
(out / "pypdf.txt").write_text(
    PdfReader(str(PDF)).pages[PAGE].extract_text() or "", encoding="utf-8")

# PyMuPDF
import fitz
(out / "pymupdf.txt").write_text(
    fitz.open(str(PDF))[PAGE].get_text("text"), encoding="utf-8")

# pdfplumber
import pdfplumber
with pdfplumber.open(str(PDF)) as pdf:
    (out / "pdfplumber.txt").write_text(
        pdf.pages[PAGE].extract_text() or "", encoding="utf-8")

for f in sorted(out.glob("*.txt")):
    print(f"\n{'='*20} {f.name} {'='*20}")
    print(f.read_text(encoding="utf-8")[:1200])