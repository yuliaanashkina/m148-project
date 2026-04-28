"""
Continuous-time Markov chain utilities for the M148 capstone project.

The module is intentionally organized around the planned modeling progression:

1. Global CTMC: estimate one generator matrix Q from all journeys.
2. Clustered CTMC: cluster journeys without using the outcome label, then fit
   one Q per segment.
3. Personalized neural-style CTMC: learn customer-specific transition
   probabilities and holding rates, then combine them as q_ij(x).
4. Benchmarks: compare against standard tabular classifiers.

The code uses the checkpointed parquet files produced by run_data_pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FLATTENED_PATH = DATA_DIR / "journeys_flattened.parquet"
DEFAULT_TRAINING_PATH = DATA_DIR / "journey_training_optionA.parquet"
DEFAULT_LABELED_PATH = DATA_DIR / "journeys_labeled.parquet"


def _as_seconds(values: pd.Series) -> pd.Series:
    """Convert pandas timedeltas to nonnegative seconds."""
    seconds = values.dt.total_seconds()
    return seconds.clip(lower=0).fillna(0.0)


@dataclass
class CTMCData:
    """Loads flattened journeys and returns event-level or journey-level views."""

    flattened_path: Path = DEFAULT_FLATTENED_PATH
    training_path: Path = DEFAULT_TRAINING_PATH
    labeled_path: Path = DEFAULT_LABELED_PATH

    def load_journeys(self, max_journeys: int | None = None) -> pl.DataFrame:
        q = pl.scan_parquet(self.flattened_path)
        if max_journeys is not None:
            q = q.head(max_journeys)
        return q.collect()

    def load_training_features(self, max_rows: int | None = None) -> pd.DataFrame:
        q = pl.scan_parquet(self.training_path)
        if max_rows is not None:
            q = q.head(max_rows)
        return q.collect().drop("prefix_actions").to_pandas()

    def load_neural_rate_training_features(self, max_rows: int | None = None) -> pd.DataFrame:
        """
        Load model features plus remaining time to success for positive rows.

        The remaining-time target is computed at the same snapshot position used
        by journey_training_optionA.parquet.
        """
        training = pl.scan_parquet(self.training_path)
        if max_rows is not None:
            training = training.head(max_rows)

        base = training.select(
            [
                "id",
                "label",
                "final_outcome",
                "full_num_actions",
                "snapshot_num_actions",
                "first_action_so_far",
                "current_last_action",
                "num_unique_actions_so_far",
                "prefix_duration_seconds",
                "snapshot_frac_of_journey",
                "avg_gap_seconds",
                *[c for c in training.collect_schema().names() if c.startswith("action_count_")],
            ]
        )

        snapshot_times = (
            pl.scan_parquet(self.labeled_path)
            .select(["id", "journey", "end_time"])
            .explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
            .with_columns(pl.int_range(1, pl.len() + 1).over("id").alias("action_num"))
            .join(base.select(["id", "snapshot_num_actions"]), on="id", how="inner")
            .filter(pl.col("action_num") == pl.col("snapshot_num_actions"))
            .select(["id", pl.col("event_timestamp").alias("snapshot_time"), "end_time"])
        )

        enriched = (
            base.join(snapshot_times, on="id", how="left")
            .with_columns(
                pl.when(pl.col("label") == 1)
                .then((pl.col("end_time") - pl.col("snapshot_time")).dt.total_seconds())
                .otherwise(None)
                .alias("remaining_time_to_success_seconds")
            )
            .drop(["snapshot_time", "end_time"])
        )
        return enriched.collect().to_pandas()

    def load_binary_labels(self) -> pd.DataFrame:
        """Return resolved journey labels: successful=1, incomplete=0."""
        return (
            pl.scan_parquet(self.labeled_path)
            .filter(pl.col("journey_status").is_in(["successful", "incomplete"]))
            .select(
                [
                    "id",
                    (pl.col("journey_status") == "successful").cast(pl.Int8).alias("label"),
                    "journey_status",
                ]
            )
            .collect()
            .to_pandas()
        )

    def events(self, max_journeys: int | None = None) -> pd.DataFrame:
        journeys = self.load_journeys(max_journeys=max_journeys)
        events = (
            journeys.select(["id", "journey"])
            .explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
        )
        pdf = events.to_pandas()
        pdf["event_timestamp"] = pd.to_datetime(pdf["event_timestamp"], utc=True)
        return pdf

    def transition_table(self, max_journeys: int | None = None) -> pd.DataFrame:
        events = self.events(max_journeys=max_journeys)
        events["next_state"] = events.groupby("id")["ed_id"].shift(-1)
        events["next_timestamp"] = events.groupby("id")["event_timestamp"].shift(-1)
        transitions = events.dropna(subset=["next_state", "next_timestamp"]).copy()
        transitions["state"] = transitions["ed_id"].astype(int)
        transitions["next_state"] = transitions["next_state"].astype(int)
        transitions["dt_seconds"] = _as_seconds(
            transitions["next_timestamp"] - transitions["event_timestamp"]
        )
        return transitions[["id", "state", "next_state", "dt_seconds"]]


class GlobalCTMC:
    """Estimate a single global CTMC generator matrix Q."""

    def __init__(self, min_time: float = 1e-9) -> None:
        self.min_time = min_time
        self.states_: list[int] = []
        self.transition_counts_: pd.DataFrame | None = None
        self.time_in_state_: pd.Series | None = None
        self.Q_: pd.DataFrame | None = None

    def fit(self, transitions: pd.DataFrame) -> "GlobalCTMC":
        transitions = transitions.copy()
        transitions = transitions[transitions["state"] != transitions["next_state"]]

        states = sorted(
            set(transitions["state"].astype(int)).union(
                set(transitions["next_state"].astype(int))
            )
        )
        self.states_ = states

        counts = pd.crosstab(transitions["state"], transitions["next_state"])
        counts = counts.reindex(index=states, columns=states, fill_value=0)

        time_in_state = transitions.groupby("state")["dt_seconds"].sum()
        time_in_state = time_in_state.reindex(states, fill_value=0.0).clip(
            lower=self.min_time
        )

        q_values = counts.div(time_in_state, axis=0).astype(float).to_numpy(copy=True)
        np.fill_diagonal(q_values, 0.0)
        q_values[np.diag_indices_from(q_values)] = -q_values.sum(axis=1)
        q = pd.DataFrame(q_values, index=states, columns=states)

        self.transition_counts_ = counts
        self.time_in_state_ = time_in_state
        self.Q_ = q
        return self

    def top_rates(self, n: int = 15) -> pd.DataFrame:
        self._check_fit()
        q = self.Q_.copy()
        rows = []
        for i in self.states_:
            for j in self.states_:
                if i != j and q.loc[i, j] > 0:
                    rows.append({"from_state": i, "to_state": j, "rate": q.loc[i, j]})
        return pd.DataFrame(rows).sort_values("rate", ascending=False).head(n)

    def transition_probability(self, horizon_seconds: float) -> pd.DataFrame:
        self._check_fit()
        try:
            from scipy.linalg import expm
        except ImportError as exc:
            raise ImportError("scipy is required for matrix exponential P(t)=exp(Qt)") from exc

        p = expm(self.Q_.to_numpy() * horizon_seconds)
        return pd.DataFrame(p, index=self.states_, columns=self.states_)

    def absorption_probability(
        self,
        current_states: Iterable[int],
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> np.ndarray:
        """
        Estimate P(hit success_state by horizon | current state).

        The success state is made absorbing by zeroing its generator row before
        computing exp(Qt). Unknown states fall back to the global empirical
        success prior implied by the transition count table.
        """
        self._check_fit()
        if success_state not in self.states_:
            return np.zeros(len(list(current_states)), dtype=float)

        try:
            from scipy.linalg import expm
        except ImportError as exc:
            raise ImportError("scipy is required for CTMC absorption probabilities") from exc

        q_absorbing = self.Q_.copy()
        q_absorbing.loc[success_state, :] = 0.0
        p = expm(q_absorbing.to_numpy() * horizon_seconds)

        state_to_idx = {state: idx for idx, state in enumerate(self.states_)}
        success_idx = state_to_idx[success_state]
        fallback = self._empirical_success_prior(success_state)

        probs = []
        for state in current_states:
            idx = state_to_idx.get(int(state))
            probs.append(fallback if idx is None else float(p[idx, success_idx]))
        return np.clip(np.asarray(probs), 0.0, 1.0)

    def expected_hitting_time(
        self,
        current_states: Iterable[int],
        success_state: int = 28,
    ) -> np.ndarray:
        """
        Expected seconds to hit success_state from each current state.

        Uses the standard CTMC linear system on transient states:
            Q_T m = -1.
        States with no finite solution fall back to infinity.
        """
        self._check_fit()
        current_states = list(current_states)
        if success_state not in self.states_:
            return np.full(len(current_states), np.inf)

        transient_states = [state for state in self.states_ if state != success_state]
        if not transient_states:
            return np.zeros(len(current_states), dtype=float)

        q_t = self.Q_.loc[transient_states, transient_states].to_numpy(dtype=float)
        try:
            times = np.linalg.solve(q_t, -np.ones(len(transient_states)))
        except np.linalg.LinAlgError:
            times = np.linalg.lstsq(q_t, -np.ones(len(transient_states)), rcond=None)[0]

        time_by_state = {
            state: max(float(time), 0.0) if np.isfinite(time) else np.inf
            for state, time in zip(transient_states, times)
        }
        time_by_state[success_state] = 0.0

        return np.asarray([time_by_state.get(int(state), np.inf) for state in current_states])

    def _empirical_success_prior(self, success_state: int) -> float:
        if self.transition_counts_ is None or success_state not in self.transition_counts_.columns:
            return 0.0
        total = float(self.transition_counts_.to_numpy().sum())
        if total <= 0:
            return 0.0
        return float(self.transition_counts_[success_state].sum() / total)

    def _check_fit(self) -> None:
        if self.Q_ is None:
            raise RuntimeError("Call fit() before reading CTMC outputs.")


class JourneyFeatureBuilder:
    """Build leakage-safe clustering features from observed journey prefixes."""

    def __init__(self, top_n_actions: int = 20, key_actions: Iterable[int] = (28,)) -> None:
        self.top_n_actions = top_n_actions
        self.key_actions = list(key_actions)
        self.top_actions_: list[int] = []

    def fit_transform(self, transitions: pd.DataFrame) -> pd.DataFrame:
        self.top_actions_ = (
            transitions["state"].value_counts().head(self.top_n_actions).index.astype(int).tolist()
        )
        return self.transform(transitions)

    def transform(self, transitions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        grouped = transitions.sort_values(["id"]).groupby("id", sort=False)
        for user_id, g in grouped:
            total_actions = len(g) + 1
            total_time = float(g["dt_seconds"].sum())
            row = {
                "id": user_id,
                "num_transitions": len(g),
                "total_observed_time": total_time,
                "avg_gap_seconds": total_time / max(len(g), 1),
                "first_state": int(g["state"].iloc[0]),
                "current_state": int(g["next_state"].iloc[-1]),
            }
            counts = g["state"].value_counts()
            for action in self.top_actions_:
                count = int(counts.get(action, 0))
                row[f"action_count_{action}"] = count
                row[f"action_prop_{action}"] = count / max(total_actions, 1)
                row[f"time_in_state_{action}"] = float(
                    g.loc[g["state"] == action, "dt_seconds"].sum()
                )
            for action in self.key_actions:
                hit = g[g["next_state"] == action]
                row[f"time_to_action_{action}"] = (
                    float(g.loc[: hit.index[0], "dt_seconds"].sum()) if len(hit) else np.nan
                )
                row[f"seen_action_{action}"] = int(len(hit) > 0)
            rows.append(row)

        features = pd.DataFrame(rows).fillna(-1)
        return features

    def features_from_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Build the same leakage-safe feature surface from open/test journeys.

        Test journeys have no next observed state after their final action, so
        gap-based summaries use within-prefix time differences only.
        """
        rows = []
        for user_id, g in events.sort_values(["id", "event_timestamp"]).groupby("id", sort=False):
            g = g.copy()
            g["gap_seconds"] = (
                g["event_timestamp"].diff().dt.total_seconds().clip(lower=0).fillna(0.0)
            )
            total_actions = len(g)
            total_time = float(g["gap_seconds"].sum())
            row = {
                "id": user_id,
                "num_transitions": max(total_actions - 1, 0),
                "total_observed_time": total_time,
                "avg_gap_seconds": total_time / max(total_actions - 1, 1),
                "first_state": int(g["ed_id"].iloc[0]),
                "current_state": int(g["ed_id"].iloc[-1]),
            }
            counts = g["ed_id"].value_counts()
            for action in self.top_actions_:
                count = int(counts.get(action, 0))
                row[f"action_count_{action}"] = count
                row[f"action_prop_{action}"] = count / max(total_actions, 1)
                row[f"time_in_state_{action}"] = float(
                    g.loc[g["ed_id"] == action, "gap_seconds"].sum()
                )
            for action in self.key_actions:
                hit = g[g["ed_id"] == action]
                row[f"time_to_action_{action}"] = (
                    float(g.loc[: hit.index[0], "gap_seconds"].sum()) if len(hit) else np.nan
                )
                row[f"seen_action_{action}"] = int(len(hit) > 0)
            rows.append(row)
        return pd.DataFrame(rows).fillna(-1)


class ClusteredCTMC:
    """Cluster journeys, then fit one GlobalCTMC generator per cluster."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.feature_builder = JourneyFeatureBuilder()
        self.models_: dict[int, GlobalCTMC] = {}
        self.assignments_: pd.DataFrame | None = None
        self.pipeline_ = None
        self.feature_columns_: list[str] = []

    def fit(self, transitions: pd.DataFrame) -> "ClusteredCTMC":
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        features = self.feature_builder.fit_transform(transitions)
        x = features.drop(columns=["id"])
        self.feature_columns_ = x.columns.tolist()

        self.pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("cluster", KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)),
            ]
        )
        clusters = self.pipeline_.fit_predict(x)
        assignments = features[["id"]].copy()
        assignments["cluster"] = clusters
        self.assignments_ = assignments

        clustered_transitions = transitions.merge(assignments, on="id", how="inner")
        for cluster_id, g in clustered_transitions.groupby("cluster"):
            self.models_[int(cluster_id)] = GlobalCTMC().fit(g)
        return self

    def cluster_summary(self) -> pd.DataFrame:
        if self.assignments_ is None:
            raise RuntimeError("Call fit() before reading cluster summaries.")
        return (
            self.assignments_["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="n_journeys")
        )

    def predict_clusters(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before predicting clusters.")
        return self.pipeline_.predict(features[self.feature_columns_])

    def predict_success_probability(
        self,
        features: pd.DataFrame,
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
        fallback_model: GlobalCTMC | None = None,
    ) -> np.ndarray:
        clusters = self.predict_clusters(features)
        probs = np.zeros(len(features), dtype=float)
        for cluster_id in sorted(set(clusters)):
            mask = clusters == cluster_id
            model = self.models_.get(int(cluster_id), fallback_model)
            if model is None:
                continue
            probs[mask] = model.absorption_probability(
                features.loc[mask, "current_state"],
                success_state=success_state,
                horizon_seconds=horizon_seconds,
            )
        return np.clip(probs, 0.0, 1.0)

    def predict_expected_hitting_time(
        self,
        features: pd.DataFrame,
        success_state: int = 28,
        fallback_model: GlobalCTMC | None = None,
    ) -> np.ndarray:
        clusters = self.predict_clusters(features)
        times = np.full(len(features), np.inf, dtype=float)
        for cluster_id in sorted(set(clusters)):
            mask = clusters == cluster_id
            model = self.models_.get(int(cluster_id), fallback_model)
            if model is None:
                continue
            times[mask] = model.expected_hitting_time(
                features.loc[mask, "current_state"],
                success_state=success_state,
            )
        return times


class NeuralRateCTMC:
    """
    Neural exponential-rate CTMC for direct success timing.

    This model does not predict the next state. It estimates a personalized
    success intensity lambda(x) and uses the exponential waiting-time model

        P(T_success <= t | x) = 1 - exp(-lambda(x) t).

    The target rate is derived from truncated journey snapshots. Successful
    rows use the observed remaining time to order_shipped when available;
    incomplete rows are treated as censored at the 60-day horizon with a small
    baseline hazard.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        horizon_seconds: float = 60 * 24 * 60 * 60,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.horizon_seconds = horizon_seconds
        self.random_state = random_state
        self.pipeline_ = None
        self.feature_columns_: list[str] = []

    def fit(self, training_df: pd.DataFrame) -> "NeuralRateCTMC":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        data = training_df.copy()
        if "prefix_actions" in data.columns:
            data = data.drop(columns=["prefix_actions"])

        self.feature_columns_ = [
            c for c in data.columns if c not in {"id", "label", "final_outcome", "remaining_time_to_success_seconds"}
        ]

        if "remaining_time_to_success_seconds" in data.columns:
            remaining = data["remaining_time_to_success_seconds"].to_numpy(dtype=float)
            positive_time = np.maximum(remaining, 1.0)
        else:
            positive_time = np.maximum(data.get("prefix_duration_seconds", self.horizon_seconds), 1.0)

        labels = data["label"].to_numpy(dtype=int)
        target_rate = np.where(
            labels == 1,
            1.0 / positive_time,
            1.0 / (10.0 * self.horizon_seconds),
        )

        regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        random_state=self.random_state,
                        max_iter=300,
                        early_stopping=True,
                    ),
                ),
            ]
        )

        self.pipeline_ = TransformedTargetRegressor(
            regressor=regressor,
            func=np.log,
            inverse_func=np.exp,
        )
        self.pipeline_.fit(data[self.feature_columns_], np.clip(target_rate, 1e-12, None))
        return self

    def predict_lambda(self, rows: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Call fit() before predicting neural rates.")
        x = rows.copy()
        if "prefix_actions" in x.columns:
            x = x.drop(columns=["prefix_actions"])
        for col in self.feature_columns_:
            if col not in x.columns:
                x[col] = 0
        rates = self.pipeline_.predict(x[self.feature_columns_])
        return np.clip(np.asarray(rates, dtype=float), 1e-12, None)

    def predict_success_probability(
        self,
        rows: pd.DataFrame,
        horizon_seconds: float | None = None,
    ) -> np.ndarray:
        horizon = self.horizon_seconds if horizon_seconds is None else horizon_seconds
        lambdas = self.predict_lambda(rows)
        return np.clip(1.0 - np.exp(-lambdas * horizon), 0.0, 1.0)

    def predict_expected_success_time(self, rows: pd.DataFrame) -> np.ndarray:
        return 1.0 / self.predict_lambda(rows)


class ModelComparison:
    """Benchmark CTMC-derived/tabular features against standard classifiers."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.results_: pd.DataFrame | None = None

    def run(self, training_df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = training_df.copy()
        y = df["label"].astype(int)
        x = df.drop(columns=[c for c in ["label", "id", "final_outcome"] if c in df.columns])

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        models = {
            "logistic_regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=self.random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=150, random_state=self.random_state, n_jobs=-1, class_weight="balanced_subsample"
            ),
        }
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        except ImportError:
            pass

        rows = []
        for name, model in models.items():
            model.fit(x_train, y_train)
            probs = model.predict_proba(x_test)[:, 1]
            rows.append(
                {
                    "model": name,
                    "roc_auc": roc_auc_score(y_test, probs),
                    "average_precision": average_precision_score(y_test, probs),
                    "log_loss": log_loss(y_test, probs, labels=[0, 1]),
                    "brier_score": brier_score_loss(y_test, probs),
                }
            )

        self.results_ = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
        return self.results_


def sample_pipeline(max_journeys: int = 50_000, n_clusters: int = 4) -> dict[str, object]:
    """Convenience function used by the exploration notebook."""
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys)

    global_ctmc = GlobalCTMC().fit(transitions)
    clustered_ctmc = ClusteredCTMC(n_clusters=n_clusters).fit(transitions)
    features = clustered_ctmc.feature_builder.transform(transitions)

    return {
        "transitions": transitions,
        "features": features,
        "global_ctmc": global_ctmc,
        "clustered_ctmc": clustered_ctmc,
    }
