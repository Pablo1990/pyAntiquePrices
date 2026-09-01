from __future__ import annotations

from pyantique_prices.vision.schemas import AntiqueIdentification, EvidenceValue, MarkEvidence


def test_evidence_value_defaults():
    value = EvidenceValue()

    assert value.value is None
    assert value.confidence == 0.0
    assert value.evidence_images == []


def test_antique_identification_nested_defaults():
    identification = AntiqueIdentification(
        materials=["porcelain"],
        marks=[MarkEvidence(text="Meissen", confidence=0.9)],
    )

    assert identification.object_type.value is None
    assert identification.materials == ["porcelain"]
    assert identification.marks[0].text == "Meissen"
    assert identification.model_dump()["contradictions"] == []
