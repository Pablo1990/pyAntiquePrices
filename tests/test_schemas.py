from __future__ import annotations

from pyantique_prices.vision.schemas import (
    AntiqueIdentification,
    AntiqueImageSet,
    Candidate,
    EvidenceValue,
    ImageRole,
    MarkEvidence,
)


def test_evidence_value_defaults():
    value = EvidenceValue()

    assert value.value is None
    assert value.confidence == 0.0
    assert value.evidence_images == []


def test_antique_identification_coerces_candidates_and_period_alias():
    identification = AntiqueIdentification(
        object_type={"value": "vase"},
        period="late 19th century",
        materials=["porcelain"],
        manufacturer_candidates=["Meissen"],
        marks=[MarkEvidence(text="Meissen", confidence=0.9)],
    )

    assert identification.object_type == "vase"
    assert identification.likely_period == "late 19th century"
    assert identification.materials == ["porcelain"]
    assert identification.marks[0].text == "Meissen"
    assert identification.model_dump()["contradictions"] == []
    assert identification.manufacturer_candidates[0] == Candidate(
        name="Meissen",
        confidence=0.5,
        evidence=None,
    )


def test_antique_image_set_tracks_roles():
    image_set = AntiqueImageSet.from_paths(
        ["front.jpg", "mark.jpg"],
        roles=[ImageRole.front, ImageRole.maker_mark],
    )

    assert image_set.images[0].name == "front.jpg"
    assert image_set.roles == [ImageRole.front, ImageRole.maker_mark]
