#!/usr/bin/env python3
"""Evaluate the pricing model on held-out data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("Model evaluation requires trained model artifacts in models/")
    schema_path = Path("models/feature_schema.json")
    if schema_path.exists():
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        print(f"Model trained on {schema.get('num_training_samples', 0)} samples")
    else:
        print("No model found. Run scripts/train_price_model.py first.")


if __name__ == "__main__":
    main()
