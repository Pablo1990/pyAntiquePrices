"""Tests for AntiqueAnalyzer."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_IMAGE = Path(__file__).parent / "assets" / "dummy.jpg"


@pytest.fixture(autouse=True, scope="session")
def create_dummy_image(tmp_path_factory):
    """Create a tiny valid JPEG for tests that need a real file."""
    assets = Path(__file__).parent / "assets"
    assets.mkdir(exist_ok=True)
    if not DUMMY_IMAGE.exists():
        # Minimal 1x1 white JPEG
        jpeg_bytes = bytes(
            [
                0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
                0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06,
                0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C,
                0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F,
                0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
                0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28,
                0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27,
                0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF,
                0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10,
                0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05,
                0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00,
                0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
                0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
                0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33,
                0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A,
                0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
                0xFB, 0xD2, 0xCA, 0xED, 0x90, 0xFF, 0xD9,
            ]
        )
        DUMMY_IMAGE.write_bytes(jpeg_bytes)


# ---------------------------------------------------------------------------
# AntiqueAnalyzer tests
# ---------------------------------------------------------------------------

from pyantique_prices.analyzer import AntiqueAnalyzer


class TestEncodeImage:
    def test_encodes_existing_file(self):
        encoded = AntiqueAnalyzer._encode_image(DUMMY_IMAGE)
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert len(decoded) > 0

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            AntiqueAnalyzer._encode_image("/nonexistent/path/image.jpg")


class TestCollectImages:
    def test_single_file_returns_list_of_one(self):
        result = AntiqueAnalyzer.collect_images(DUMMY_IMAGE)
        assert result == [DUMMY_IMAGE]

    def test_directory_returns_images(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        (tmp_path / "readme.txt").write_bytes(b"x")
        result = AntiqueAnalyzer.collect_images(tmp_path)
        names = {p.name for p in result}
        assert names == {"a.jpg", "b.png"}

    def test_directory_sorted(self, tmp_path):
        for name in ("c.jpg", "a.jpg", "b.jpg"):
            (tmp_path / name).write_bytes(b"x")
        result = AntiqueAnalyzer.collect_images(tmp_path)
        assert [p.name for p in result] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_empty_directory_returns_empty(self, tmp_path):
        result = AntiqueAnalyzer.collect_images(tmp_path)
        assert result == []

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            AntiqueAnalyzer.collect_images("/no/such/path")


class TestParsePriceRange:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Estimated price: 500 – 800 EUR", (500.0, 800.0)),
            ("Value range: €1,200-€1,500", (1200.0, 1500.0)),
            ("Between 300 to 450 EUR", (300.0, 450.0)),
            ("No price mentioned here.", None),
        ],
    )
    def test_parse(self, text, expected):
        result = AntiqueAnalyzer.parse_price_range(text)
        assert result == expected


class TestAnalyse:
    def _make_mock_ollama(self, content="This is a fine 19th century vase worth €400-€600.", model="minicpm-v"):
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": content}}
        # Simulate model already present locally (new SDK: object with .models attribute)
        model_obj = MagicMock()
        model_obj.model = model
        list_response = MagicMock()
        list_response.models = [model_obj]
        mock_ollama.list.return_value = list_response
        return mock_ollama

    def test_analyse_calls_ollama(self):
        mock_ollama = self._make_mock_ollama()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="minicpm-v")
            result = analyzer.analyse(DUMMY_IMAGE, context="Blue vase")

        # chat is called at least twice: once for keyword generation, once for appraisal
        assert mock_ollama.chat.call_count >= 2
        assert "vase" in result.lower()

    def test_analyse_uses_context_and_prices(self):
        mock_ollama = self._make_mock_ollama(content="appraisal text")

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer()
            analyzer.analyse(
                DUMMY_IMAGE,
                context="My grandmother's clock",
                reference_prices="Similar clocks: 300-500 EUR",
            )

        # When reference_prices is pre-supplied, only the appraisal chat call is made
        mock_ollama.chat.assert_called_once()
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0].get("messages", []) if call_args.args else call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "grandmother" in user_message["content"]
        assert "300-500" in user_message["content"]

    def test_deep_thinking_prompt_includes_thinking_section(self):
        mock_ollama = self._make_mock_ollama(content="deep result")

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(deep_thinking=True)
            analyzer.analyse(DUMMY_IMAGE)

        # The last chat call is the appraisal call
        call_args = mock_ollama.chat.call_args_list[-1]
        messages = call_args.kwargs.get("messages") or call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "<thinking>" in user_message["content"]

    def test_standard_prompt_no_thinking_section(self):
        mock_ollama = self._make_mock_ollama(content="standard result")

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(deep_thinking=False)
            analyzer.analyse(DUMMY_IMAGE)

        # The last chat call is the appraisal call
        call_args = mock_ollama.chat.call_args_list[-1]
        messages = call_args.kwargs.get("messages") or call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "<thinking>" not in user_message["content"]

    def test_auto_pulls_missing_model(self):
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "appraisal"}}
        # Model is NOT in the local list (new SDK object style)
        list_response = MagicMock()
        list_response.models = []
        mock_ollama.list.return_value = list_response
        # pull returns an iterable of progress objects
        prog = MagicMock()
        prog.status = "success"
        mock_ollama.pull.return_value = iter([prog])

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="minicpm-v")
            analyzer.analyse(DUMMY_IMAGE)

        mock_ollama.pull.assert_called_once_with("minicpm-v", stream=True)
        assert mock_ollama.chat.call_count >= 1

    def test_pull_failure_raises_runtime_error(self):
        mock_ollama = MagicMock()
        list_response = MagicMock()
        list_response.models = []
        mock_ollama.list.return_value = list_response
        mock_ollama.pull.side_effect = Exception("connection refused")

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="minicpm-v")
            with pytest.raises(RuntimeError, match="Failed to download model"):
                analyzer.analyse(DUMMY_IMAGE)

    def test_skips_pull_when_model_present(self):
        mock_ollama = self._make_mock_ollama()

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="minicpm-v")
            analyzer.analyse(DUMMY_IMAGE)

        mock_ollama.pull.assert_not_called()

    def test_skips_pull_when_model_present_old_sdk_format(self):
        """Ensure old SDK dict-style list response also works."""
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "ok"}}
        # Old SDK: plain dict
        mock_ollama.list.return_value = {"models": [{"name": "minicpm-v"}]}

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="minicpm-v")
            analyzer.analyse(DUMMY_IMAGE)

        mock_ollama.pull.assert_not_called()

    def test_non_vision_model_raises_value_error(self):
        """400 multimodal error from Ollama is re-raised as a friendly ValueError."""
        mock_ollama = self._make_mock_ollama()
        mock_ollama.chat.side_effect = Exception(
            '{"error":{"code":400,"message":"Multimodal data provided, but model does not support multimodal requests.","type":"invalid_request_error"}}'
        )

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="gpt-oss:20b")
            with pytest.raises(ValueError, match="does not support image"):
                analyzer.analyse(DUMMY_IMAGE)

    def test_other_error_reraises_unchanged(self):
        """Non-multimodal errors are re-raised as-is (not wrapped in ValueError)."""
        mock_ollama = self._make_mock_ollama()
        mock_ollama.chat.side_effect = Exception("connection timeout")

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            analyzer = AntiqueAnalyzer(model="minicpm-v")
            with pytest.raises(Exception, match="connection timeout"):
                analyzer.analyse(DUMMY_IMAGE)
