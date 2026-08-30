"""pyAntiquePrices – public API."""

from .analyzer import AntiqueAnalyzer, RECOMMENDED_MODELS
from .scraper import DuckDuckGoScraper, MultiSourceScraper

__all__ = [
    "AntiqueAnalyzer",
    "RECOMMENDED_MODELS",
    "DuckDuckGoScraper",
    "MultiSourceScraper",
]
__version__ = "0.1.0"
