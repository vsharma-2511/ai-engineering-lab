"""Decide how each PDF page must be read, before any OCR runs.

A PDF page can carry a text layer (typed or exported documents), only
pictures of text (scanner output), or both (a typed report with a
pasted scan or screenshot). Reading the text layer is exact and fast;
OCR is slower and can misread, so it is used only where needed:

  text     enough text layer and no large text-free images: read as is
  scanned  no usable text layer, but images or vector drawings (scans,
           text converted to outlines): OCR the whole page
  mixed    a text layer plus large images without text over them:
           read the text layer and OCR just those image regions
  blank    no text and nothing drawn: nothing to read

A whole document is "digital" (only text pages), "scanned" (every
non-blank page scanned) or "mixed" (anything in between).
"""
from dataclasses import dataclass, field

import pymupdf

from .. import config

# Pages with less text than this have no usable text layer.
MIN_USEFUL_CHARACTERS = 20
# Text turned into vector outlines draws many paths; a few lines or
# boxes do not make a page worth OCR.
MIN_DRAWINGS_FOR_OCR = 50


@dataclass
class PageAnalysis:
    kind: str                      # text | scanned | mixed | blank
    text_characters: int
    image_coverage: float          # share of the page covered by images
    ocr_regions: list = field(default_factory=list)  # pymupdf.Rect, mixed
    reason: str = ""


def _image_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = []
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if not rect.is_empty:
            rects.append(rect)
    return rects


def analyze_page(
    page: pymupdf.Page,
    min_region_fraction: float | None = None,
) -> PageAnalysis:
    if min_region_fraction is None:
        min_region_fraction = config.OCR_MIN_REGION_FRACTION
    page_area = page.rect.get_area() or 1.0

    text_characters = len(page.get_text("text").strip())
    images = _image_rects(page)
    coverage = min(1.0, sum(r.get_area() for r in images) / page_area)

    if text_characters < MIN_USEFUL_CHARACTERS:
        if images:
            return PageAnalysis("scanned", text_characters, coverage,
                                reason="images but no text layer")
        if len(page.get_drawings()) >= MIN_DRAWINGS_FOR_OCR:
            return PageAnalysis("scanned", text_characters, coverage,
                                reason="vector drawings but no text layer")
        return PageAnalysis("blank", text_characters, coverage,
                            reason="no text and nothing drawn")

    regions = [
        rect for rect in images
        if rect.get_area() / page_area >= min_region_fraction
        # A text layer over the image (an already-OCR'd scan) is read
        # directly; OCR'ing it again would duplicate the text.
        and len(page.get_text("text", clip=rect).strip())
        < MIN_USEFUL_CHARACTERS
    ]
    if regions:
        return PageAnalysis("mixed", text_characters, coverage, regions,
                            reason=f"{len(regions)} image region(s) "
                                   "without a text layer")
    return PageAnalysis("text", text_characters, coverage,
                        reason="text layer")


def document_type(kinds: list[str]) -> str:
    """digital, scanned or mixed, from the page kinds."""
    content = [kind for kind in kinds if kind != "blank"]
    if not content or all(kind == "text" for kind in content):
        return "digital"
    if all(kind == "scanned" for kind in content):
        return "scanned"
    return "mixed"
