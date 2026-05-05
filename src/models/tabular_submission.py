"""
Train tabular baselines and write Kaggle-format submissions.

This mirrors the notebook RF approach in the repo, but makes it reusable from
the CTMC exploration notebook and writes outputs to results/ by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSION_DIR = RESULTS_DIR / "submissions"

TRAINING_PATH = DATA_DIR / "journey_training_optionA.parquet"
DEFAULT_TEST_EVENTS = DATA_DIR / "open_journeys1.csv"
DEFAULT_SAMPLE = DATA_DIR / "open_journeys1_flattened_all0.csv"
TUNING_SUMMARY = RESULTS_DIR / "optuna_tuning_summary.csv"


def load_tuned_params(model_name: str) -> dict:
    if not TUNING_SUMMARY.exists():
        return {}
    summary = pd.read_csv(TUNING_SUMMARY)
    row = summary[summary["model"] == model_name]
    if row.empty:
        return {}
    params = row.iloc[0].get("best_params", "{}")
    if not isinstance(params, str) or not params:
        return {}
    return json.loads(params)


def load_training(max_rows: int | None = None) -> pd.DataFrame:
    q = pl.scan_parquet(TRAINING_PATH)
    if max_rows is not None:
        q = q.head(max_rows)
    return q.collect().drop("prefix_actions").to_pandas()


def build_open_journey_features(test_events_path: Path = DEFAULT_TEST_EVENTS) -> pd.DataFrame:
    if not test_events_path.exists():
        raise FileNotFoundError(
            f"Missing test event file: {test_events_path}. Add open_journeys1.csv to data/."
        )

    df = pl.read_csv(test_events_path)
    required = {"id", "event_timestamp", "ed_id"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{test_events_path} must contain {sorted(required)}")

    df = df.with_columns(pl.col("event_timestamp").str.to_datetime(time_zone="UTC"))
    journeys = (
        df.sort(["id", "event_timestamp", "ed_id"])
        .group_by("id")
        .agg(
            [
                pl.col("ed_id").alias("actions"),
                pl.col("event_timestamp").alias("timestamps"),
                pl.len().alias("full_num_actions"),
            ]
        )
    )

    rows = []
    for row in journeys.iter_rows(named=True):
        actions = row["actions"]
        timestamps = row["timestamps"]
        n = len(actions)

        if n > 1:
            prefix_duration = int((timestamps[-1] - timestamps[0]).total_seconds())
            avg_gap = prefix_duration / (n - 1)
            time_since_prev = int((timestamps[-1] - timestamps[-2]).total_seconds())
        else:
            prefix_duration = 0
            avg_gap = 0.0
            time_since_prev = 0

        out = {
            "id": row["id"],
            "full_num_actions": n,
            "snapshot_num_actions": n,
            "snapshot_frac_of_journey": 1.0,
            "first_action_so_far": actions[0],
            "current_last_action": actions[-1],
            "num_unique_actions_so_far": len(set(actions)),
            "prefix_duration_seconds": prefix_duration,
            "avg_gap_seconds": avg_gap,
            "time_since_prev_action_seconds": time_since_prev,
        }
        counts = pd.Series(actions).value_counts()
        for action_id, count in counts.items():
            out[f"action_count_{int(action_id)}"] = int(count)
        rows.append(out)

    return pd.DataFrame(rows).fillna(0)


def align_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_cols = [c for c in train_df.columns if c not in {"id", "label", "final_outcome"}]
    x_train = train_df[feature_cols].copy()
    y_train = train_df["label"].astype(int)
    x_test = test_df.copy()

    for col in feature_cols:
        if col not in x_test.columns:
            x_test[col] = 0
    x_test = x_test[feature_cols].copy()
    return x_train, y_train, x_test


def write_submission(ids: pd.Series, probs: np.ndarray, output_path: Path, sample_path: Path = DEFAULT_SAMPLE) -> pd.DataFrame:
    submission = pd.DataFrame(
        {
            "id": ids.astype(str).to_numpy(),
            "order_shipped": np.clip(probs, 0.0, 1.0),
        }
    )
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if "id" in sample.columns:
            sample["id"] = sample["id"].astype(str)
            submission = sample[["id"]].merge(submission, on="id", how="left")
            submission["order_shipped"] = submission["order_shipped"].fillna(0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def create_tabular_submissions(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = SUBMISSION_DIR,
    max_train_rows: int | None = 300_000,
) -> dict[str, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    train_df = load_training(max_rows=max_train_rows)
    test_df = build_open_journey_features(test_events_path)
    x_train, y_train, x_test = align_features(train_df, test_df)

    outputs: dict[str, pd.DataFrame] = {}

    rf_params = load_tuned_params("random_forest")
    rf = RandomForestClassifier(
        n_estimators=int(rf_params.get("n_estimators", 250)),
        max_depth=rf_params.get("max_depth"),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 1)),
        max_features=rf_params.get("max_features", "sqrt"),
        class_weight=rf_params.get("class_weight", "balanced_subsample"),
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    outputs["random_forest"] = write_submission(
        test_df["id"],
        rf.predict_proba(x_test)[:, 1],
        output_dir / "random_forest_submission.csv",
        sample_path=sample_path,
    )

    hgb_params = load_tuned_params("hist_gradient_boosting")
    hgb = HistGradientBoostingClassifier(
        learning_rate=hgb_params.get("learning_rate", 0.1),
        max_iter=int(hgb_params.get("max_iter", 100)),
        max_leaf_nodes=int(hgb_params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(hgb_params.get("min_samples_leaf", 20)),
        l2_regularization=hgb_params.get("l2_regularization", 0.0),
        random_state=42,
    )
    hgb.fit(x_train, y_train)
    outputs["hist_gradient_boosting"] = write_submission(
        test_df["id"],
        hgb.predict_proba(x_test)[:, 1],
        output_dir / "hist_gradient_boosting_submission.csv",
        sample_path=sample_path,
    )

    try:
        from xgboost import XGBClassifier

        xgb_params = load_tuned_params("xgboost")
        xgb = XGBClassifier(
            n_estimators=int(xgb_params.get("n_estimators", 350)),
            max_depth=int(xgb_params.get("max_depth", 5)),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            subsample=xgb_params.get("subsample", 0.9),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.9),
            min_child_weight=xgb_params.get("min_child_weight", 1.0),
            reg_alpha=xgb_params.get("reg_alpha", 0.0),
            reg_lambda=xgb_params.get("reg_lambda", 1.0),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(x_train, y_train)
        outputs["xgboost"] = write_submission(
            test_df["id"],
            xgb.predict_proba(x_test)[:, 1],
            output_dir / "xgboost_submission.csv",
            sample_path=sample_path,
        )
    except ImportError:
        print("xgboost is not installed; skipped xgboost_submission.csv")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tabular Kaggle submissions.")
    parser.add_argument("--test-events", type=Path, default=DEFAULT_TEST_EVENTS)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=SUBMISSION_DIR)
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    args = parser.parse_args()

    outputs = create_tabular_submissions(
        test_events_path=args.test_events,
        sample_path=args.sample,
        output_dir=args.output_dir,
        max_train_rows=args.max_train_rows,
    )
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")
        print(df.head())


if __name__ == "__main__":
    main()
