"""
Create Kaggle-format submissions from the CTMC models.

Expected Kaggle format:

    id,order_shipped
    -1000001271 551641434,0.1

The script looks for open/test journeys in data/open_journeys1.csv by default.
If data/open_journeys1_flattened_all0.csv is present, it is used only to match
Kaggle's requested row ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

try:
    from .ctmc import (
        CTMCData,
        ClusteredCTMC,
        GlobalCTMC,
        JourneyFeatureBuilder,
    )
    from .tabular_submission import (
        brier_report,
        calibrate_prevalence,
        TEST_PREVALENCE,
    )
except ImportError:
    from ctmc import (
        CTMCData,
        ClusteredCTMC,
        GlobalCTMC,
        JourneyFeatureBuilder,
    )
    from tabular_submission import (
        brier_report,
        calibrate_prevalence,
        TEST_PREVALENCE,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_DIR = PROJECT_ROOT / "results" / "submissions"

DEFAULT_TEST_EVENTS = DATA_DIR / "open_journeys1.csv"
DEFAULT_SAMPLE = DATA_DIR / "open_journeys1_flattened_all0.csv"

SUCCESS_STATE = 28
HORIZON_SECONDS = 60 * 24 * 60 * 60


def load_test_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing test event file: {path}. Add open_journeys1.csv to data/ "
            "to create real Kaggle CTMC submissions."
        )

    df = pl.read_csv(path)
    required = {"id", "event_timestamp"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{path} must contain at least columns {sorted(required)}")

    if "ed_id" not in df.columns:
        raise ValueError(
            f"{path} must contain ed_id for CTMC state prediction. "
            "The sample submission file alone is not enough."
        )

    events = df.select(["id", "event_timestamp", "ed_id"]).to_pandas()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
    events["ed_id"] = events["ed_id"].astype(int)
    return events.sort_values(["id", "event_timestamp", "ed_id"])


def write_flattened_all0_template(events: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Create the Kaggle-style all-zero ID template from open journey events.

    The historical notebooks use open_journeys1_flattened_all0.csv only to
    enforce row order and column names. It is not a feature table.
    """
    template = (
        events[["id"]]
        .drop_duplicates()
        .assign(order_shipped=0.0)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return template


def test_features_from_events(events: pd.DataFrame, builder: JourneyFeatureBuilder) -> pd.DataFrame:
    features = builder.features_from_events(events)
    features["state"] = features["current_state"]
    return features


def write_submission(
    ids: pd.Series,
    probabilities: np.ndarray,
    output_path: Path,
    sample_path: Path | None = DEFAULT_SAMPLE,
) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "id": ids.astype(str).to_numpy(),
            "order_shipped": np.clip(probabilities, 0.0, 1.0),
        }
    )

    if sample_path is not None and sample_path.exists():
        sample = pd.read_csv(sample_path)
        if "id" in sample.columns:
            sample["id"] = sample["id"].astype(str)
            submission = sample[["id"]].merge(submission, on="id", how="left")
            submission["order_shipped"] = submission["order_shipped"].fillna(0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def fit_training_ctmcs(
    max_train_journeys: int | None,
    n_clusters: int,
) -> tuple[pd.DataFrame, GlobalCTMC, ClusteredCTMC, JourneyFeatureBuilder]:
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_train_journeys)

    global_ctmc = GlobalCTMC().fit(transitions)
    clustered_ctmc = ClusteredCTMC(n_clusters=n_clusters).fit(transitions)
    builder = clustered_ctmc.feature_builder

    return transitions, global_ctmc, clustered_ctmc, builder


def create_ctmc_submissions(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = SUBMISSION_DIR,
    max_train_journeys: int | None = 100_000,
    n_clusters: int = 3,
) -> dict[str, pd.DataFrame]:
    events = load_test_events(test_events_path)
    if sample_path is not None and not sample_path.exists():
        write_flattened_all0_template(events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    _, global_ctmc, clustered_ctmc, builder = fit_training_ctmcs(
        max_train_journeys=max_train_journeys,
        n_clusters=n_clusters,
    )

    features = test_features_from_events(events, builder)

    print(f"Test prevalence target: {TEST_PREVALENCE:.4f}  (all-zeros Brier baseline={TEST_PREVALENCE:.5f})")

    global_probs_raw = global_ctmc.absorption_probability(
        features["current_state"],
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
    )
    clustered_probs_raw = clustered_ctmc.predict_success_probability(
        features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_model=global_ctmc,
    )

    blended_raw = 0.5 * global_probs_raw + 0.5 * clustered_probs_raw

    global_probs    = calibrate_prevalence(global_probs_raw)
    clustered_probs = calibrate_prevalence(clustered_probs_raw)
    blended_probs   = calibrate_prevalence(blended_raw)

    brier_report("global_ctmc",    global_probs)
    brier_report("clustered_ctmc", clustered_probs)
    brier_report("blend_ctmc",     blended_probs)

    outputs = {
        "global": write_submission(
            features["id"],
            global_probs,
            output_dir / "ctmc_global_submission.csv",
            sample_path=sample_path,
        ),
        "clustered": write_submission(
            features["id"],
            clustered_probs,
            output_dir / "ctmc_clustered_submission.csv",
            sample_path=sample_path,
        ),
        "blend": write_submission(
            features["id"],
            blended_probs,
            output_dir / "ctmc_blend_submission.csv",
            sample_path=None,
        ),
    }

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CTMC Kaggle submissions.")
    parser.add_argument("--test-events", type=Path, default=DEFAULT_TEST_EVENTS)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--max-train-journeys", type=int, default=100_000)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=SUBMISSION_DIR)
    args = parser.parse_args()

    outputs = create_ctmc_submissions(
        test_events_path=args.test_events,
        sample_path=args.sample,
        output_dir=args.output_dir,
        max_train_journeys=args.max_train_journeys,
        n_clusters=args.n_clusters,
    )
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")
        print(df.head())


if __name__ == "__main__":
    main()
