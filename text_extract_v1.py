from __future__ import annotations
from pathlib import Path

def extract_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    ext = p.suffix.lower()
    if ext == ".pdf":
        import pdfplumber
        chunks = []
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    chunks.append(t)
        return "\n".join(chunks).strip()

    if ext in (".docx",):
        from docx import Document
        doc = Document(str(p))
        paras = [par.text for par in doc.paragraphs if par.text and par.text.strip()]
        return "\n".join(paras).strip()

    if ext in (".txt",):
        return p.read_text(encoding="utf-8").strip()

    raise ValueError(f"Unsupported file type: {ext}. Use pdf/docx/txt")
