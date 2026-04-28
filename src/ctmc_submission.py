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

from ctmc import (
    CTMCData,
    ClusteredCTMC,
    GlobalCTMC,
    JourneyFeatureBuilder,
    NeuralRateCTMC,
)
from tabular_submission import build_open_journey_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_DIR = DATA_DIR / "submissions"

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
    neural_transition_limit: int,
) -> tuple[pd.DataFrame, GlobalCTMC, ClusteredCTMC, JourneyFeatureBuilder, NeuralRateCTMC]:
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_train_journeys)

    global_ctmc = GlobalCTMC().fit(transitions)
    clustered_ctmc = ClusteredCTMC(n_clusters=n_clusters).fit(transitions)
    builder = clustered_ctmc.feature_builder

    neural_training = data.load_neural_rate_training_features(max_rows=neural_transition_limit)
    neural_ctmc = NeuralRateCTMC(hidden_layer_sizes=(64, 32), random_state=42)
    neural_ctmc.fit(neural_training)

    return transitions, global_ctmc, clustered_ctmc, builder, neural_ctmc


def create_ctmc_submissions(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = SUBMISSION_DIR,
    max_train_journeys: int | None = 100_000,
    n_clusters: int = 4,
    neural_transition_limit: int = 75_000,
) -> dict[str, pd.DataFrame]:
    events = load_test_events(test_events_path)
    if sample_path is not None and not sample_path.exists():
        write_flattened_all0_template(events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    _, global_ctmc, clustered_ctmc, builder, neural_ctmc = fit_training_ctmcs(
        max_train_journeys=max_train_journeys,
        n_clusters=n_clusters,
        neural_transition_limit=neural_transition_limit,
    )

    features = test_features_from_events(events, builder)
    neural_features = build_open_journey_features(test_events_path)

    global_probs = global_ctmc.absorption_probability(
        features["current_state"],
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
    )
    clustered_probs = clustered_ctmc.predict_success_probability(
        features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_model=global_ctmc,
    )
    neural_probs = neural_ctmc.predict_success_probability(
        neural_features,
        horizon_seconds=HORIZON_SECONDS,
    )

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
        "neural": write_submission(
            neural_features["id"],
            neural_probs,
            output_dir / "ctmc_neural_rate_submission.csv",
            sample_path=sample_path,
        ),
    }

    blended = (
        0.4 * outputs["global"]["order_shipped"].to_numpy()
        + 0.4 * outputs["clustered"]["order_shipped"].to_numpy()
        + 0.2 * outputs["neural"]["order_shipped"].to_numpy()
    )
    outputs["blend"] = write_submission(
        outputs["global"]["id"],
        blended,
        output_dir / "ctmc_blend_submission.csv",
        sample_path=None,
    )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CTMC Kaggle submissions.")
    parser.add_argument("--test-events", type=Path, default=DEFAULT_TEST_EVENTS)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--max-train-journeys", type=int, default=100_000)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--neural-transition-limit", type=int, default=75_000)
    parser.add_argument("--output-dir", type=Path, default=SUBMISSION_DIR)
    args = parser.parse_args()

    outputs = create_ctmc_submissions(
        test_events_path=args.test_events,
        sample_path=args.sample,
        output_dir=args.output_dir,
        max_train_journeys=args.max_train_journeys,
        n_clusters=args.n_clusters,
        neural_transition_limit=args.neural_transition_limit,
    )
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")
        print(df.head())


if __name__ == "__main__":
    main()
