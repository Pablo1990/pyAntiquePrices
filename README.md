# pyAntiquePrices

> **Local-first antique appraisal powered by a vision LLM.**  
> Point it at an image (or a whole folder), pick a model, and get an expert-quality appraisal with estimated age, condition, and price range — all running on your own machine.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick start – GUI](#quick-start--gui)
5. [Quick start – CLI](#quick-start--cli)
6. [Python API](#python-api)
7. [Recommended models](#recommended-models)
8. [Model compatibility notes](#model-compatibility-notes)
9. [Reference price search](#reference-price-search)
10. [Deep thinking mode](#deep-thinking-mode)
11. [Batch processing](#batch-processing)
12. [Troubleshooting](#troubleshooting)
13. [Development](#development)
14. [Legal & ethical notes](#legal--ethical-notes)

---

## How it works

```
Image file
    │
    ▼
AntiqueAnalyzer
    ├─ Step 1 – Auto keyword generation
    │       └─ Quick LLM call: identifies the object and produces 5-7 search terms
    ├─ Step 2 – Automatic price scraping
    │       └─ MultiSourceScraper queries DuckDuckGo → Catawiki / LiveAuctioneers / Invaluable
    │          (user-supplied extra keywords are merged with the auto-generated ones)
    └─ Step 3 – Full appraisal
            └─ Expert appraiser system prompt
            └─ (Optional) Deep-thinking chain-of-thought prompt
                    │
                    ▼
            Structured appraisal:
              • Description (type, style, materials)
              • Estimated age / period
              • Condition grade
              • Price range (auction + retail, EUR)
              • Key value factors
              • Confidence level
```

No data leaves your machine. The LLM runs locally via [Ollama](https://ollama.com/).
Scraping is fully automatic — you don't need to supply any keywords unless you want to refine the search.

---

## Requirements

| Requirement | Notes |
|---|---|
| **Python ≥ 3.9** | |
| **[Ollama](https://ollama.com/)** | Must be installed and running. Download from [ollama.com](https://ollama.com/download). |
| **A vision-capable model** | Auto-downloaded on first use, or run `ollama pull minicpm-v` in advance. |
| `ollama` Python SDK | Installed automatically via `pip`. |
| `requests`, `beautifulsoup4` | Web scraper (installed automatically). |
| `Pillow` | Image utilities (installed automatically). |
| `tkinter` | GUI — ships with most Python distributions. On Linux: `sudo apt install python3-tk`. |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Pablo1990/pyAntiquePrices.git
cd pyAntiquePrices

# 2. Install the package (editable mode recommended for development)
pip install -e .

# 3. Start Ollama (if not already running)
ollama serve          # macOS/Linux background daemon
# On Windows: Ollama runs as a system tray application after installation
```

The first time you analyse an image, the selected model will be automatically downloaded if it is not already present. To pre-download the default model:

```bash
ollama pull minicpm-v
```

---

## Quick start – GUI

```bash
pyantique-prices
```

The graphical interface opens immediately.

### Step-by-step

1. **Select an image or folder**  
   Click **Image…** to pick a single photo, or **Folder…** to select a directory.  
   Supported formats: JPEG, PNG, WEBP, BMP, TIFF, GIF.

2. **Choose a model**  
   The dropdown is pre-populated with recommended vision models.  
   You can also type any Ollama model name directly.  
   If you enter a model that is not in the recommended list, a warning dialog will ask you to confirm — text-only models will fail because they cannot process images.

3. **Enable Deep thinking** *(recommended)*  
   When checked, the model explicitly reasons step-by-step before producing its final answer. This takes longer but is significantly more accurate, especially for ambiguous items.

4. **Enter extra search keywords** *(optional)*  
   Keywords are **auto-generated from the image** — the model first identifies the object and produces auction-specialist search terms automatically. If you want to refine the search (e.g. you know the origin or have the maker's name), add extra keywords here and they will be merged with the auto-generated ones.

5. **Add context** *(optional)*  
   Any extra information you have: provenance, size, markings, origin, family history.

6. **Click Analyse antique**  
   A progress bar tracks each image. Results appear in the scrollable appraisal panel below.

---

## Quick start – CLI

```bash
# Single image, default model (minicpm-v), deep thinking on
pyantique-prices --cli path/to/photo.jpg

# With search keywords and additional context
pyantique-prices --cli photo.jpg \
    --keywords "bronze Buddha Meiji period" \
    --context "Found in a Japanese estate, base has red lacquer seal"

# Specify a different model
pyantique-prices --cli photo.jpg --model llava:13b

# Disable deep thinking for a faster pass
pyantique-prices --cli photo.jpg --no-deep-thinking

# Analyse an entire folder of images
pyantique-prices --cli /path/to/antiques/

# Combine folder + keywords + model
pyantique-prices --cli /path/to/antiques/ \
    --model llava \
    --keywords "Chinese porcelain blue white" \
    --context "Items from a French chateau clearance"
```

### All CLI options

| Option | Default | Description |
|---|---|---|
| `--cli IMAGE_OR_FOLDER` | — | Path to an image file or a directory of images. |
| `--model MODEL` | `minicpm-v` | Ollama model name. Must support vision. |
| `--keywords KEYWORDS` | *(empty)* | Extra search keywords merged with auto-generated ones. |
| `--context CONTEXT` | *(empty)* | Free-text context about the item(s). |
| `--deep-thinking` | on | Enable chain-of-thought reasoning (default). |
| `--no-deep-thinking` | — | Disable chain-of-thought for faster results. |

Running without `--cli` launches the GUI.

---

## Python API

```python
from pyantique_prices import AntiqueAnalyzer, MultiSourceScraper

# 1. (Optional) fetch web reference prices
scraper = MultiSourceScraper()
prices = scraper.get_reference_prices("Chinese blue and white porcelain vase 18th century")

# 2. Analyse the image
analyzer = AntiqueAnalyzer(
    model="minicpm-v",     # any Ollama vision model
    deep_thinking=True,    # chain-of-thought reasoning
)
appraisal = analyzer.analyse(
    "vase.jpg",
    context="No markings on base. Approx 35 cm tall.",
    reference_prices=prices,
)

print(appraisal)

# 3. (Optional) extract the numeric price range
low, high = analyzer.parse_price_range(appraisal) or (None, None)
if low:
    print(f"Estimated range: €{low:.0f} – €{high:.0f}")
```

### Batch processing via API

```python
from pyantique_prices import AntiqueAnalyzer

images = AntiqueAnalyzer.collect_images("/path/to/folder")
analyzer = AntiqueAnalyzer()

for img in images:
    result = analyzer.analyse(img, context="Estate sale items, France")
    print(f"\n=== {img.name} ===\n{result}")
```

---

## Recommended models

These models are confirmed to work with **Ollama ≥ 0.30**.

| Model | RAM / VRAM | Notes |
|---|---|---|
| **`minicpm-v`** *(default)* | ~5 GB | Best balance of accuracy and speed. Works on all Ollama versions. |
| `llava:13b` | ~10 GB | Larger LLaVA — stronger visual understanding. |
| `llava` | ~4 GB | Original LLaVA 7B — lightweight fallback. |
| `moondream` | ~2 GB | Very small; good for low-memory machines. |
| `gemma3` | ~6 GB | Google Gemma 3 vision variant — strong reasoning. |
| `mistral-small3.1` | ~6 GB | Mistral vision model. |

To install a model manually:

```bash
ollama pull minicpm-v
ollama pull llava:13b
```

---

## Model compatibility notes

> ⚠️ **`llama3.2-vision` is NOT supported** by Ollama ≥ 0.30.  
> It uses the `mllama` architecture which was removed when Ollama replaced its backend.  
> Use `minicpm-v` or any model from the table above instead.

> ⚠️ **Text-only models** (e.g. `mistral`, `deepseek-r1`, `gemma`) will fail with:  
> `Multimodal data provided, but model does not support multimodal requests.`  
> The app will show a clear error message listing the recommended vision models.

---

## Reference price search

Price scraping is **fully automatic** — no configuration required.

For every image analysed, the app:

1. Makes a quick LLM call to identify the object and produce 5-7 auction-specialist search keywords.
2. Merges those with any extra keywords you supplied.
3. Queries **DuckDuckGo** (HTML endpoint, no API key required) scoped to major auction sites to find comparable listings:

- [Catawiki](https://www.catawiki.com)
- [LiveAuctioneers](https://www.liveauctioneers.com)
- [Invaluable](https://www.invaluable.com)

The scraped snippets are injected into the LLM prompt so the model can anchor its price estimate to real comparable sales.

The scraper fully respects each site's `robots.txt` and uses a polite 3-second crawl delay between requests. If a site disallows scraping, that source is silently skipped and the appraisal continues without it.

**When to supply extra keywords:**
- You know the maker or manufacturer.
- You have information about the origin not visible in the image.
- The auto-search is returning irrelevant results (e.g. add `"19th century"` to narrow results).

---

## Deep thinking mode

When **Deep thinking** is enabled (the default), the model is asked to work through six explicit reasoning steps before producing the final appraisal:

1. **Object identification** — What is it? Rule out alternatives.
2. **Style and period analysis** — What decorative features narrow the date?
3. **Material and technique assessment** — How was it made?
4. **Condition and authenticity** — What wear is visible? Is ageing consistent?
5. **Market comparables** — What comparable pieces or sales come to mind?
6. **Synthesis** — Combine all factors into a probability-weighted estimate.

The thinking section is shown in the output so you can review the reasoning.  
Disable it with `--no-deep-thinking` (CLI) or uncheck the box (GUI) when you need faster results.

---

## Batch processing

### GUI

Click **Folder…** to select a directory. The app will find all image files inside and analyse them in order. A progress bar advances after each image. Results are shown with a clear separator:

```
============================================================
Image 1/5: vase_01.jpg
============================================================
1. **Description**: Blue and white porcelain vase...
...
============================================================
Image 2/5: clock_01.jpg
============================================================
...
```

### CLI

Pass a directory path to `--cli`:

```bash
pyantique-prices --cli /path/to/images/ --keywords "estate sale" --model minicpm-v
```

Per-image errors are reported to `stderr` without stopping the batch.

---

## Troubleshooting

### `Connection refused` / Ollama not running

Start Ollama first:

```bash
ollama serve
```

Or on macOS/Windows, launch the Ollama application from your Applications folder / system tray.

---

### `unknown model architecture: 'mllama'`

You are using `llama3.2-vision` with Ollama ≥ 0.30. Switch to a supported model:

```bash
pyantique-prices --model minicpm-v --cli photo.jpg
```

---

### `Multimodal data provided, but model does not support multimodal requests`

The selected model is text-only and cannot process images. Use a vision model from the [Recommended models](#recommended-models) table.

---

### GUI does not open (`_tkinter` not found)

Install Tkinter for your Python distribution:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (Homebrew Python)
brew install python-tk
```

---

### Model download is slow

Large models can be several gigabytes. Pre-download before running the app:

```bash
ollama pull minicpm-v      # ~5 GB
ollama pull llava:13b       # ~10 GB
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_analyzer.py -v
```

### Project structure

```
pyAntiquePrices/
├── pyantique_prices/
│   ├── __init__.py        # Public API exports
│   ├── __main__.py        # CLI entry point
│   ├── analyzer.py        # AntiqueAnalyzer – LLM appraisal logic
│   ├── scraper.py         # DuckDuckGoScraper, MultiSourceScraper
│   └── gui.py             # Tkinter GUI
├── tests/
│   ├── test_analyzer.py
│   └── test_scraper.py
├── pyproject.toml
└── README.md
```

### Adding a new scraper source

1. Subclass `_BaseScraper` in `scraper.py`.
2. Set `base_url` and implement `get_reference_prices()`.
3. Check robots.txt via `self._is_allowed(path)` before fetching.
4. Register the new scraper in `MultiSourceScraper.__init__()`.

---

## Legal & ethical notes

- **Privacy**: The LLM runs entirely locally via Ollama. No image data is sent to any external service.
- **Web scraping**: The DuckDuckGo scraper respects `robots.txt` and applies a configurable crawl delay (default 3 s). It identifies itself with a descriptive `User-Agent`.
- **Accuracy**: Appraisals are AI-generated estimates based on visual information only. They should be treated as a starting point, not a professional valuation. For high-value items, consult a certified appraiser.

---

## License

See [LICENSE](LICENSE).
