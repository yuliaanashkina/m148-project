"""
Compare CTMC success-probability predictions with an expected-time rule.

This is intentionally separate from the submission pipeline. It evaluates on
truncated journey snapshots from journey_training_optionA.parquet so that the
model does not get to see completed successful journeys ending in state 28.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ctmc import CTMCData, ClusteredCTMC, GlobalCTMC, NeuralRateCTMC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUCCESS_STATE = 28
HORIZON_SECONDS = 60 * 24 * 60 * 60


def load_snapshots(max_rows: int | None = 100_000) -> pd.DataFrame:
    q = pl.scan_parquet(DATA_DIR / "journey_training_optionA.parquet")
    if max_rows is not None:
        q = q.head(max_rows)
    return q.collect().to_pandas()


def split_ids(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> tuple[set[str], set[str]]:
    from sklearn.model_selection import train_test_split

    labels = df[["id", "label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        labels["id"],
        test_size=test_size,
        random_state=random_state,
        stratify=labels["label"],
    )
    return set(train_ids), set(test_ids)


def transition_table_for_ids(ids: set[str]) -> pd.DataFrame:
    journeys = (
        pl.scan_parquet(DATA_DIR / "journeys_flattened.parquet")
        .filter(pl.col("id").is_in(list(ids)))
        .select(["id", "journey"])
        .explode("journey")
        .unnest("journey")
        .sort(["id", "event_timestamp", "ed_id"])
        .collect()
        .to_pandas()
    )
    journeys["event_timestamp"] = pd.to_datetime(journeys["event_timestamp"], utc=True)
    journeys["next_state"] = journeys.groupby("id")["ed_id"].shift(-1)
    journeys["next_timestamp"] = journeys.groupby("id")["event_timestamp"].shift(-1)
    transitions = journeys.dropna(subset=["next_state", "next_timestamp"]).copy()
    transitions["state"] = transitions["ed_id"].astype(int)
    transitions["next_state"] = transitions["next_state"].astype(int)
    transitions["dt_seconds"] = (
        transitions["next_timestamp"] - transitions["event_timestamp"]
    ).dt.total_seconds().clip(lower=0).fillna(0.0)
    return transitions[["id", "state", "next_state", "dt_seconds"]]


def snapshot_cluster_features(snapshots: pd.DataFrame, top_actions: list[int]) -> pd.DataFrame:
    rows = []
    for row in snapshots.itertuples(index=False):
        actions = list(getattr(row, "prefix_actions"))
        n = len(actions)
        duration = float(getattr(row, "prefix_duration_seconds"))
        counts = pd.Series(actions).value_counts() if actions else pd.Series(dtype=int)
        out = {
            "id": getattr(row, "id"),
            "num_transitions": max(n - 1, 0),
            "total_observed_time": duration,
            "avg_gap_seconds": float(getattr(row, "avg_gap_seconds")),
            "first_state": int(getattr(row, "first_action_so_far")),
            "current_state": int(getattr(row, "current_last_action")),
            "state": int(getattr(row, "current_last_action")),
        }
        for action in top_actions:
            count = int(counts.get(action, 0))
            out[f"action_count_{action}"] = count
            out[f"action_prop_{action}"] = count / max(n, 1)
            # Without per-action timestamps in the snapshot parquet, allocate
            # observed time proportionally. This is only for cluster assignment.
            out[f"time_in_state_{action}"] = duration * count / max(n, 1)
        out[f"time_to_action_{SUCCESS_STATE}"] = -1.0
        out[f"seen_action_{SUCCESS_STATE}"] = int(SUCCESS_STATE in actions)
        rows.append(out)
    return pd.DataFrame(rows).fillna(-1)


def metrics(y_true: np.ndarray, scores: np.ndarray, predictions: np.ndarray, name: str) -> dict[str, float | str]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    scores = np.clip(scores, 1e-6, 1 - 1e-6)
    return {
        "method": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "log_loss": log_loss(y_true, scores, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, scores),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def run_experiment(max_rows: int = 100_000, n_clusters: int = 4, neural_transition_limit: int = 50_000) -> pd.DataFrame:
    data = CTMCData()
    snapshots = load_snapshots(max_rows=max_rows)
    train_ids, test_ids = split_ids(snapshots)

    eval_snapshots = snapshots[snapshots["id"].isin(test_ids)].copy()
    y_true = eval_snapshots["label"].astype(int).to_numpy()

    train_transitions = transition_table_for_ids(train_ids)

    global_ctmc = GlobalCTMC().fit(train_transitions)
    clustered_ctmc = ClusteredCTMC(n_clusters=n_clusters, random_state=42).fit(train_transitions)

    neural_training = data.load_neural_rate_training_features(max_rows=max_rows)
    neural_training = neural_training[neural_training["id"].isin(train_ids)].head(neural_transition_limit).copy()
    neural_ctmc = NeuralRateCTMC(hidden_layer_sizes=(64, 32), random_state=42)
    neural_ctmc.fit(neural_training)

    eval_features = snapshot_cluster_features(eval_snapshots, clustered_ctmc.feature_builder.top_actions_)

    rows = []

    global_probs = global_ctmc.absorption_probability(eval_features["current_state"], success_state=SUCCESS_STATE)
    global_prob_pred = (global_probs >= 0.5).astype(int)
    rows.append(metrics(y_true, global_probs, global_prob_pred, "global_absorption_probability"))

    global_times = global_ctmc.expected_hitting_time(eval_features["current_state"], success_state=SUCCESS_STATE)
    global_time_pred = (global_times <= HORIZON_SECONDS).astype(int)
    global_time_score = np.exp(-np.nan_to_num(global_times, nan=np.inf, posinf=1e12) / HORIZON_SECONDS)
    rows.append(metrics(y_true, global_time_score, global_time_pred, "global_expected_time_under_60d"))

    clustered_probs = clustered_ctmc.predict_success_probability(
        eval_features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_model=global_ctmc,
    )
    clustered_prob_pred = (clustered_probs >= 0.5).astype(int)
    rows.append(metrics(y_true, clustered_probs, clustered_prob_pred, "clustered_absorption_probability"))

    clustered_times = clustered_ctmc.predict_expected_hitting_time(
        eval_features,
        success_state=SUCCESS_STATE,
        fallback_model=global_ctmc,
    )
    clustered_time_pred = (clustered_times <= HORIZON_SECONDS).astype(int)
    clustered_time_score = np.exp(-np.nan_to_num(clustered_times, nan=np.inf, posinf=1e12) / HORIZON_SECONDS)
    rows.append(metrics(y_true, clustered_time_score, clustered_time_pred, "clustered_expected_time_under_60d"))

    neural_eval = eval_snapshots.drop(columns=["prefix_actions"], errors="ignore").copy()
    neural_probs = neural_ctmc.predict_success_probability(
        neural_eval,
        horizon_seconds=HORIZON_SECONDS,
    )
    neural_prob_pred = (neural_probs >= 0.5).astype(int)
    rows.append(metrics(y_true, neural_probs, neural_prob_pred, "neural_exponential_rate_probability"))

    neural_times = neural_ctmc.predict_expected_success_time(neural_eval)
    neural_time_pred = (neural_times <= HORIZON_SECONDS).astype(int)
    neural_time_score = np.exp(-np.nan_to_num(neural_times, nan=np.inf, posinf=1e12) / HORIZON_SECONDS)
    rows.append(metrics(y_true, neural_time_score, neural_time_pred, "neural_exponential_time_under_60d"))

    results = pd.DataFrame(rows).sort_values(["log_loss", "brier_score"])
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "ctmc_waiting_time_experiment.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CTMC probability vs waiting-time rules.")
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--neural-transition-limit", type=int, default=50_000)
    args = parser.parse_args()

    results = run_experiment(
        max_rows=args.max_rows,
        n_clusters=args.n_clusters,
        neural_transition_limit=args.neural_transition_limit,
    )
    print(results)


if __name__ == "__main__":
    main()
