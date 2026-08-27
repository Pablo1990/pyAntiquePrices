# pyAntiquePrices

A Python package to estimate the **price and age** of antiques (art, furniture, ceramics, jewellery, …) from an image, using a local LLM via [Ollama](https://ollama.com/).

## Features

* 🖼️ **Multimodal LLM appraisal** – submits your image to any Ollama vision model (default: `llava`) and receives a structured expert-style appraisal including description, estimated age, condition, price range and confidence.
* 🌐 **Web reference prices** – optionally scrapes [todocoleccion.net](https://www.todocoleccion.net) for real-world comparable listings to ground the appraisal. The scraper fully respects `robots.txt` and includes a polite crawl delay.
* 🖥️ **Minimal GUI** – a lightweight Tkinter interface; no web server or cloud service required.
* 💻 **Headless CLI mode** – for scripting or CI pipelines.

## Requirements

| Dependency | Notes |
|---|---|
| Python ≥ 3.9 | |
| [Ollama](https://ollama.com/) | Must be running locally with a vision model pulled, e.g. `ollama pull llava` |
| `ollama` Python SDK | installed automatically |
| `requests`, `beautifulsoup4` | web scraper |
| `Pillow` | image utilities |
| `tkinter` | GUI (usually ships with Python) |

## Installation

```bash
pip install -e .
```

## Usage

### GUI (default)

```bash
pyantique-prices
```

1. Click **Browse…** to select an image.
2. *(Optional)* Enter search keywords to look up reference prices on todocoleccion.net.
3. *(Optional)* Add any additional context about the item.
4. Click **Analyse antique**.

### CLI (headless)

```bash
pyantique-prices --cli path/to/item.jpg \
    --keywords "reloj de bolsillo plata siglo XIX" \
    --context "Found in a Spanish estate, no hallmarks visible" \
    --model llava
```

### Python API

```python
from pyantique_prices import AntiqueAnalyzer, TodoColeccionScraper

scraper = TodoColeccionScraper()
prices = scraper.get_reference_prices("jarrón chino porcelana azul blanco")

analyzer = AntiqueAnalyzer(model="llava")
appraisal = analyzer.analyse(
    "vase.jpg",
    context="Blue and white Chinese porcelain vase, no markings on base.",
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

* The web scraper honours `robots.txt` on todocoleccion.net and includes a configurable crawl delay (default 3 s) to avoid overloading the server.
* The LLM runs entirely **locally** via Ollama; no image data is sent to any external service.

