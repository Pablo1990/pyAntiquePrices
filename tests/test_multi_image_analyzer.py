from __future__ import annotations

from pathlib import Path

from pyantique_prices.vision.analyzer import MultiImageAnalyzer, validate_images
from pyantique_prices.vision.ollama import OllamaClient, is_context_overflow_error


class _FakeClient:
    def analyze_images(self, images, prompt: str, system: str | None = None):  # noqa: ARG002
        assert len(images) == 3
        return """{
  "object_type": "mantel clock",
  "country": "France",
  "condition": "good",
  "marks": [{"text": "Japy Freres", "confidence": 0.9}]
}"""


class _FakeTextClient:
    def analyze_images(self, images, prompt: str, system: str | None = None):  # noqa: ARG002
        assert len(images) == 3
        return """Object: Oil painting
Period: Late 19th century
Country: France
Materials: Oil on canvas
Condition: fair"""


class _OverflowThenCompactClient:
    def __init__(self):
        self.calls = []

    def analyze_images(self, images, prompt: str, system: str | None = None):  # noqa: ARG002
        self.calls.append({"prompt": prompt, "system": system})
        if len(self.calls) == 1:
            raise Exception(
                '{"error":{"code":400,"message":"request (6923 tokens) exceeds the available context size (4096 tokens), try increasing it","type":"exceed_context_size_error"}}'
            )
        return """{
  "object_type": "lithograph",
  "country": "France"
}"""


def _touch_images(tmp_path: Path) -> list[Path]:
    files = []
    for idx in range(3):
        path = tmp_path / f"img{idx}.jpg"
        path.write_bytes(b"test")
        files.append(path)
    return files


def test_validate_images_enforces_range_and_extensions(tmp_path):
    files = _touch_images(tmp_path)
    valid = validate_images(files)
    assert len(valid) == 3


def test_multi_image_analyzer_returns_structured_identification(tmp_path):
    files = _touch_images(tmp_path)
    analyzer = MultiImageAnalyzer(client=_FakeClient())
    result = analyzer.analyze(files, context="French family estate")
    assert result["object_type"]["value"] == "mantel clock"
    assert result["country"]["value"] == "France"
    assert result["marks"][0]["normalized_text"] == "JAPY FRERES"
    assert result["manufacturer_candidates"][0]["name"] == "Japy Freres"


def test_multi_image_analyzer_falls_back_to_text_fields(tmp_path):
    files = _touch_images(tmp_path)
    analyzer = MultiImageAnalyzer(client=_FakeTextClient())
    result = analyzer.analyze(files, context="")
    assert result["object_type"]["value"] == "Oil painting"
    assert result["likely_period"]["value"] == "Late 19th century"
    assert result["country"]["value"] == "France"
    assert "Oil on canvas" in result["materials"]


def test_multi_image_analyzer_retries_with_compact_prompt_after_context_overflow(tmp_path):
    files = _touch_images(tmp_path)
    client = _OverflowThenCompactClient()
    analyzer = MultiImageAnalyzer(client=client)

    result = analyzer.analyze(files, context="x" * 2000)

    assert result["object_type"]["value"] == "lithograph"
    assert len(client.calls) == 2
    assert client.calls[0]["system"] is not None
    assert client.calls[1]["system"] is None


def test_context_overflow_error_detection():
    assert is_context_overflow_error(
        Exception("request exceeds the available context size (4096 tokens)")
    )


def test_ollama_client_passes_num_ctx_option(tmp_path):
    files = _touch_images(tmp_path)

    class _FakeOllamaSdkClient:
        def __init__(self):
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"message": {"content": "{}"}}

    client = OllamaClient(num_ctx=16384)
    fake_sdk_client = _FakeOllamaSdkClient()
    client._client = fake_sdk_client

    client.analyze_images(files, "prompt", system="system")

    assert fake_sdk_client.kwargs["options"]["num_ctx"] == 16384
