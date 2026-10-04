from __future__ import annotations

from pyantique_prices.vision.marks import (
    MarkAnalysisService,
    classify_mark_type,
    detect_mark_regions,
    extract_mark_text,
    generate_mark_candidates,
    lookup_manufacturer_candidates,
    normalize_mark,
    normalize_mark_text,
)
from pyantique_prices.vision.schemas import AntiqueImageSet, ImageRole


def test_normalize_mark_text():
    assert normalize_mark_text("  Japy Frères #12 ") == "JAPY FRRES 12"
    assert normalize_mark("Wedgwood") == "WEDGWOOD"


def test_lookup_manufacturer_candidates():
    candidates = lookup_manufacturer_candidates("JAPY FRERES")
    assert "Japy Freres" in candidates


def test_generate_mark_candidates_returns_evidence_backed_candidates():
    candidates = generate_mark_candidates("JAPY FRERES")
    assert candidates[0].name == "Japy Freres"
    assert candidates[0].confidence > 0
    assert candidates[0].evidence


def test_classify_mark_type_uses_keywords():
    mark_type = classify_mark_type({"text": "Signed on base"})
    assert mark_type == "signature"


def test_detect_mark_regions_prefers_base_and_mark_images():
    image_set = AntiqueImageSet.from_paths(
        ["front.jpg", "base.jpg", "maker_mark.jpg"],
        roles=[ImageRole.front, ImageRole.base, ImageRole.maker_mark],
    )
    regions = detect_mark_regions(image_set)
    assert [region["image_name"] for region in regions] == ["base.jpg", "maker_mark.jpg"]


def test_extract_mark_text_does_not_fabricate_unreadable_marks():
    image_set = AntiqueImageSet.from_paths(["base.jpg"], roles=[ImageRole.base])
    marks = extract_mark_text(
        image_set,
        regions=[{"image_name": "base.jpg", "location": "base", "confidence": 0.4}],
    )
    assert marks[0]["text"] is None
    assert "could not be read" in marks[0]["evidence"].lower()


def test_mark_analysis_service_enriches_marks():
    service = MarkAnalysisService()
    marks = service.analyze(
        {"marks": [{"text": "Japy Freres", "confidence": 0.8, "location": "base"}]}
    )
    assert marks[0]["normalized_text"] == "JAPY FRERES"
    assert marks[0]["mark_type"] == "text"
    assert marks[0]["manufacturer_candidates"][0]["name"] == "Japy Freres"
