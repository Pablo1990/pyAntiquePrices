# pyAntiquePrices

A Python package to estimate the **price and age** of antiques (art, furniture, ceramics, jewellery, …) from an image, using a local LLM via [Ollama](https://ollama.com/).

## Features

* 🖼️ **Multimodal LLM appraisal** – submits your image to any Ollama vision model (default: `llama3.2-vision`) with a domain-specific expert prompt and receives a structured appraisal: description, estimated period, condition, EUR price range, key value drivers and confidence level.
* 🧠 **Deep thinking mode** – chain-of-thought reasoning prompt that walks through object ID, style analysis, material assessment, condition check, market comparables and synthesis before giving a final answer. Significantly improves accuracy on ambiguous items.
* 🌐 **Web reference prices** – optionally queries DuckDuckGo (no API key) scoped to major auction sites (Catawiki, LiveAuctioneers, Invaluable) to find comparable listings and ground the price estimate. Fully respects `robots.txt` with a polite crawl delay.
* 🖥️ **Minimal GUI** – a lightweight Tkinter interface with model selector, deep-thinking toggle, folder batch mode; no web server or cloud service required.
* 💻 **Headless CLI mode** – for scripting or batch pipelines.

## Requirements

| Dependency | Notes |
|---|---|
| Python ≥ 3.9 | |
| [Ollama](https://ollama.com/) | Must be running locally. A vision model will be auto-downloaded on first use. |
| `ollama` Python SDK | installed automatically |
| `requests`, `beautifulsoup4` | web scraper |
| `Pillow` | image utilities |
| `tkinter` | GUI (usually ships with Python) |

## Recommended models

| Model | Notes |
|---|---|
| `llama3.2-vision` | **Default** – Meta's best locally-runnable vision model |
| `llava:34b` | Large LLaVA – very capable, requires ~20 GB VRAM |
| `gemma3` | Google Gemma 3 – strong general reasoning |
| `mistral-small3.1` | Mistral vision model |
| `llava` | Original LLaVA – lighter fallback |

Any Ollama-compatible model name can be typed in the GUI or passed via `--model`.

## Installation

```bash
pip install -e .
```

## Usage

### GUI (default)

```bash
pyantique-prices
```

1. Select an **Image…** or an entire **Folder…** of images.
2. Choose a model from the dropdown (or type any Ollama model name).
3. *(Optional)* Enable **Deep thinking** for more accurate chain-of-thought reasoning.
4. *(Optional)* Enter search keywords to look up reference prices from auction sites.
5. *(Optional)* Add any additional context about the item.
6. Click **Analyse antique**.

### CLI (headless)

```bash
# Single image
pyantique-prices --cli path/to/item.jpg \
    --keywords "silver pocket watch 19th century" \
    --context "Found in an English estate, no hallmarks visible" \
    --model llama3.2-vision

# Entire folder
pyantique-prices --cli path/to/folder/ --model llama3.2-vision

# Disable deep thinking for a faster pass
pyantique-prices --cli item.jpg --no-deep-thinking
```

### Python API

```python
from pyantique_prices import AntiqueAnalyzer, MultiSourceScraper

scraper = MultiSourceScraper()
prices = scraper.get_reference_prices("Chinese blue and white porcelain vase")

analyzer = AntiqueAnalyzer(model="llama3.2-vision", deep_thinking=True)
appraisal = analyzer.analyse(
    "vase.jpg",
    context="No markings on base.",
    reference_prices=prices,
)
print(appraisal)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Legal / Ethical notes

* The web scraper uses the DuckDuckGo HTML endpoint, respects its `robots.txt`, and includes a configurable crawl delay (default 3 s).
* The LLM runs entirely **locally** via Ollama; no image data is sent to any external service.

