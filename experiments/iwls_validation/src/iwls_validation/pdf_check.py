from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np


def verify_pdf_readability(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        variance = float(arr.var())
        ok = pix.width >= 1200 and pix.height >= 700 and variance > 10.0
        return {
            "path": str(pdf_path),
            "width": pix.width,
            "height": pix.height,
            "pixel_variance": variance,
            "readable": bool(ok),
        }
    finally:
        doc.close()
