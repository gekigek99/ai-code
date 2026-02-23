"""
lib.pdf — PDF detection and text extraction.

Public API:
    is_file_pdf(path) -> bool
        Check whether *path* is a PDF file (by extension + magic bytes).

    extract_text_from_pdf(path) -> str
        Extract all text content from a PDF using PyMuPDF (fitz).

Depends on the ``PyMuPDF`` (``fitz``) package.  If full image-based PDF
analysis is needed, use the Anthropic PDF-support API with citations instead.
"""

import os
import re

import fitz  # PyMuPDF


def is_file_pdf(path: str) -> bool:
    """Check if a file is a PDF based on extension and file-header magic bytes."""
    if not path.lower().endswith(".pdf"):
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(5)
            return header == b"%PDF-"
    except Exception:
        return False


def extract_text_from_pdf(path: str) -> str:
    """Extract text content from a PDF file using PyMuPDF.

    Returns a string containing all extracted text, or an error / info message
    if extraction fails or yields no content.
    """
    try:
        pdf_document = fitz.open(path)

        text_content = []

        # Header metadata
        text_content.append(f"[PDF Document: {os.path.basename(path)}]")
        text_content.append(f"[Pages: {pdf_document.page_count}]")

        metadata = pdf_document.metadata
        if metadata:
            if metadata.get("title"):
                text_content.append(f"[Title: {metadata['title']}]")
            if metadata.get("author"):
                text_content.append(f"[Author: {metadata['author']}]")

        text_content.append("[Content:]")
        text_content.append("")

        # Extract text page by page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            if pdf_document.page_count > 1:
                text_content.append(f"\n└─ Page {page_num + 1}")

            page_text = page.get_text()
            # Limit consecutive newlines for readability
            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()

            if page_text:
                text_content.append(page_text)
            else:
                text_content.append("[No text content on this page]")

        pdf_document.close()

        full_text = "\n".join(text_content)

        # If no meaningful (non-header) text was extracted, flag it
        if not any(
            line.strip() and not line.startswith("[") for line in text_content
        ):
            return "[PDF contains no extractable text content - may be scanned/image-based]"

        return full_text

    except Exception as e:
        return f"[Error extracting PDF text: {type(e).__name__}: {e}]"
