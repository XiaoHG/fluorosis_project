#!/usr/bin/env python3
"""PDF text extraction pipeline: pdftotext → OCR fallback."""
import subprocess
import sys
import os
from pathlib import Path

MIN_TEXT_CHARS = 500  # Below this, fall back to OCR


def try_pdftotext(pdf_path: str) -> str | None:
    """Attempt fast text extraction via pdftotext."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
        )
        text = result.stdout.strip()
        if len(text) >= MIN_TEXT_CHARS:
            return text
    except Exception:
        pass
    return None


def try_ocr(pdf_path: str, lang: str = "eng+chi_sim") -> str:
    """OCR fallback using pdf2image + tesseract."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(pdf_path, dpi=300)
    pages_text = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang=lang)
        pages_text.append(text)
    return "\n\n".join(pages_text)


def extract_text(pdf_path: str, ocr_lang: str = "eng+chi_sim") -> tuple[str, str]:
    """
    Extract text from a PDF. Returns (text, method_used).
    method_used is 'pdftotext' or 'ocr'.
    """
    text = try_pdftotext(pdf_path)
    if text:
        return text, "pdftotext"

    print(f"  pdftotext insufficient, running OCR on {os.path.basename(pdf_path)}...")
    text = try_ocr(pdf_path, lang=ocr_lang)
    return text, "ocr"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_ocr.py <pdf_path> [output_path] [lang]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    lang = sys.argv[3] if len(sys.argv) > 3 else "eng+chi_sim"

    text, method = extract_text(pdf_path, lang)

    if output_path:
        Path(output_path).write_text(text)
        print(f"  -> {output_path} ({method}, {len(text)} chars)")
    else:
        print(text)
