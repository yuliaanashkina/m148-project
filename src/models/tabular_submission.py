"""
Train tabular baselines and write Kaggle-format submissions.

This mirrors the notebook RF approach in the repo, but makes it reusable from
the CTMC exploration notebook and writes outputs to results/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
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

# Prevalence constants --------------------------------------------------------
# Training set: successful / (successful + incomplete) from journeys_labeled.parquet
TRAIN_PREVALENCE: float = 0.2194
# Test set: estimated from the all-zeros Kaggle Brier score (Brier = prevalence when p̂=0)
TEST_PREVALENCE: float = 0.05018

# Key funnel milestones — must match preprocess.MILESTONE_ACTIONS
MILESTONE_ACTIONS: dict[str, int] = {
    "has_add_to_cart":        11,
    "has_begin_checkout":      6,
    "has_application_submit":  3,
    "has_place_downpayment":   7,
    "has_place_order":         8,
    "has_account_activation": 29,
}


def calibrate_prevalence(
    probs: np.ndarray,
    train_prev: float = TRAIN_PREVALENCE,
    test_prev: float = TEST_PREVALENCE,
) -> np.ndarray:
    """Shift probabilities from training prevalence to test prevalence.

    Training data has ~22% successful journeys; the Kaggle test set has ~5%.
    Without adjustment every model over-predicts by ~4×. Uses the Bayes
    prior-probability (log-odds) adjustment:

        p_adj = p * r / (p * r + 1 - p)
        r = odds(test_prev) / odds(train_prev)  ≈ 0.188
    """
    r = (test_prev / (1.0 - test_prev)) / (train_prev / (1.0 - train_prev))
    odds = probs / np.maximum(1.0 - probs, 1e-10)
    return (odds * r) / (1.0 + odds * r)


def brier_report(name: str, probs: np.ndarray, test_prev: float = TEST_PREVALENCE) -> None:
    """Print predicted-probability diagnostics against the all-zeros Brier baseline."""
    mean_p = float(np.mean(probs))
    pct_above = float(np.mean(probs > 0.5)) * 100
    print(
        f"  [{name}] mean P(success)={mean_p:.4f}  target~{test_prev:.4f}  "
        f"frac>0.5={pct_above:.1f}%  "
        f"all-zeros Brier baseline={test_prev:.5f}"
    )


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

        # action count features
        counts = pd.Series(actions).value_counts()
        for action_id, count in counts.items():
            out[f"action_count_{int(action_id)}"] = int(count)

        # derived features — mirrors preprocess.add_derived_features()
        action_set = set(actions)
        if actions:
            total_act = len(actions)
            freq = np.array(list(counts.values), dtype=float)
            p = freq / total_act
            out["action_entropy"] = float(-np.sum(p * np.log2(p)))
            out["max_action_repeat"] = int(freq.max())
        else:
            out["action_entropy"] = 0.0
            out["max_action_repeat"] = 0

        for feat_name, ed_id in MILESTONE_ACTIONS.items():
            out[feat_name] = 1 if ed_id in action_set else 0

        rows.append(out)

    df = pd.DataFrame(rows).fillna(0)

    # action proportion features: action_count_X / snapshot_num_actions
    n_acts = df["snapshot_num_actions"].replace(0, np.nan)
    for col in list(df.columns):
        if col.startswith("action_count_"):
            ed_str = col.removeprefix("action_count_")
            df[f"action_prop_{ed_str}"] = (df[col] / n_acts).fillna(0.0)

    return df


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
    from sklearn.utils.class_weight import compute_sample_weight

    train_df = load_training(max_rows=max_train_rows)
    test_df = build_open_journey_features(test_events_path)
    x_train, y_train, x_test = align_features(train_df, test_df)

    print(f"Train prevalence: {y_train.mean():.4f}  |  Test target: {TEST_PREVALENCE:.4f}")
    print(f"Calibration odds ratio: {(TEST_PREVALENCE/(1-TEST_PREVALENCE))/(TRAIN_PREVALENCE/(1-TRAIN_PREVALENCE)):.4f}")

    outputs: dict[str, pd.DataFrame] = {}

    # --- Random Forest -------------------------------------------------------
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
    rf_probs = calibrate_prevalence(rf.predict_proba(x_test)[:, 1])
    brier_report("random_forest", rf_probs)
    outputs["random_forest"] = write_submission(
        test_df["id"], rf_probs,
        output_dir / "random_forest_submission.csv",
        sample_path=sample_path,
    )

    # --- HistGradientBoosting ------------------------------------------------
    # sample_weight balances classes since HGB has no class_weight parameter
    sample_weights = compute_sample_weight("balanced", y_train)
    hgb_params = load_tuned_params("hist_gradient_boosting")
    hgb = HistGradientBoostingClassifier(
        learning_rate=hgb_params.get("learning_rate", 0.1),
        max_iter=int(hgb_params.get("max_iter", 100)),
        max_leaf_nodes=int(hgb_params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(hgb_params.get("min_samples_leaf", 20)),
        l2_regularization=hgb_params.get("l2_regularization", 0.0),
        random_state=42,
    )
    hgb.fit(x_train, y_train, sample_weight=sample_weights)
    hgb_probs = calibrate_prevalence(hgb.predict_proba(x_test)[:, 1])
    brier_report("hist_gradient_boosting", hgb_probs)
    outputs["hist_gradient_boosting"] = write_submission(
        test_df["id"], hgb_probs,
        output_dir / "hist_gradient_boosting_submission.csv",
        sample_path=sample_path,
    )

    # --- XGBoost -------------------------------------------------------------
    try:
        from xgboost import XGBClassifier

        # scale_pos_weight compensates for class imbalance in training data
        scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
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
            scale_pos_weight=xgb_params.get("scale_pos_weight", scale_pos_weight),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(x_train, y_train)
        xgb_probs = calibrate_prevalence(xgb.predict_proba(x_test)[:, 1])
        brier_report("xgboost", xgb_probs)
        outputs["xgboost"] = write_submission(
            test_df["id"], xgb_probs,
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
