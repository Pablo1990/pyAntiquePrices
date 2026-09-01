from __future__ import annotations

from pyantique_prices.vision.marks import (
    MarkAnalysisService,
    classify_mark_type,
    lookup_manufacturer_candidates,
    normalize_mark_text,
)


def test_normalize_mark_text():
    assert normalize_mark_text("  Japy Frères #12 ") == "JAPY FRRES 12"


def test_lookup_manufacturer_candidates():
    candidates = lookup_manufacturer_candidates("JAPY FRERES")
    assert "Japy Freres" in candidates


def test_classify_mark_type_uses_keywords():
    mark_type = classify_mark_type({"text": "Signed on base"})
    assert mark_type == "signature"


def test_mark_analysis_service_enriches_marks():
    service = MarkAnalysisService()
    marks = service.analyze(
        {"marks": [{"text": "Japy Freres", "confidence": 0.8, "location": "base"}]}
    )
    assert marks[0]["normalized_text"] == "JAPY FRERES"
    assert marks[0]["mark_type"] == "text"
    assert "Japy Freres" in marks[0]["manufacturer_candidates"]
