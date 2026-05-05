"""Quick scratch script to peek at the open-journeys submission template.

This is a script, not a module: importing it should not run any IO.
Run with `python -m src.models.legacy_model`.
"""

from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    test_journeys = pl.read_csv(DATA_DIR / "open_journeys1_flattened_all0.csv")
    print(test_journeys.columns)
    print(test_journeys.schema)
    print(test_journeys.head())


if __name__ == "__main__":
    main()
