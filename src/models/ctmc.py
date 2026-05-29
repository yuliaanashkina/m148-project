"""
Continuous-time Markov chain utilities for the M148 capstone project.

The module is intentionally organized around the planned modeling progression:

1. Global CTMC: estimate one generator matrix Q from all journeys.
2. Clustered CTMC: cluster journeys without using the outcome label, then fit
   one Q per segment.
3. Personalized neural-style CTMC: learn customer-specific transition
   probabilities and holding rates, then combine them as q_ij(x).
4. Benchmarks: compare against standard tabular classifiers.

The code uses the checkpointed parquet files produced by the data engineering workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Windows + MKL/OpenMP can hang on tiny KMeans jobs when many worker threads are
# spawned. Keep sklearn clustering deterministic and boring unless the caller
# explicitly overrides these environment variables before import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FLATTENED_PATH = DATA_DIR / "journeys_flattened.parquet"
DEFAULT_TRAINING_PATH = DATA_DIR / "journey_training_optionA.parquet"
DEFAULT_REALISTIC_TRAINING_PATH = DATA_DIR / "journey_training_realistic_truncation.parquet"
DEFAULT_OPEN_REALISTIC_FEATURES_PATH = DATA_DIR / "open_journeys_realistic_features.parquet"
DEFAULT_LABELED_PATH = DATA_DIR / "journeys_labeled.parquet"
DEFAULT_EVENT_DEFINITIONS_PATH = DATA_DIR / "Event Definitions.csv"
SUCCESS_STATE = 28
FAILURE_STATE = -1
TIMEOUT_SECONDS = 60 * 24 * 60 * 60

# Customer-initiated events inferred from data/Event Definitions.csv. Company
# responses and fulfillment/system events should not reset the inactivity clock.
DEFAULT_CUSTOMER_ACTION_STATES = {
    2,   # campaign_click
    3,   # application_web_submit
    4,   # browse_products
    5,   # view_cart
    6,   # begin_checkout
    7,   # place_order_web
    8,   # place_downpayment
    9,   # customer_requested_catalog_digital
    10,  # fingerhut_university
    11,  # add_to_cart
    18,  # place_order_phone
    19,  # application_web_view
    22,  # pre_application_3rd_party_affiliates
    23,  # site_registration
    25,  # place_downpayment_phone
}


def _as_seconds(values: pd.Series) -> pd.Series:
    """Convert pandas timedeltas to nonnegative seconds."""
    seconds = values.dt.total_seconds()
    return seconds.clip(lower=0).fillna(0.0)


def load_customer_action_states(
    definitions_path: Path = DEFAULT_EVENT_DEFINITIONS_PATH,
    include_success: bool = False,
) -> set[int]:
    """Return ed_ids treated as customer actions for inactivity modeling.

    The CSV does not include an actor column, so this uses a conservative
    curated set checked against Event Definitions.csv. Success can be included
    as a terminal outcome state, but it is not considered a customer action.
    """
    if definitions_path.exists():
        definitions = pd.read_csv(definitions_path)
        known_ids = set(definitions["event_definition_id"].astype(int))
        unknown = DEFAULT_CUSTOMER_ACTION_STATES - known_ids
        if unknown:
            raise ValueError(f"Customer action state ids missing from definitions: {sorted(unknown)}")

    states = set(DEFAULT_CUSTOMER_ACTION_STATES)
    if include_success:
        states.add(SUCCESS_STATE)
    return states


@dataclass
class CTMCData:
    """Loads flattened journeys and returns event-level or journey-level views."""

    flattened_path: Path = DEFAULT_FLATTENED_PATH
    training_path: Path = DEFAULT_TRAINING_PATH
    realistic_training_path: Path = DEFAULT_REALISTIC_TRAINING_PATH
    open_realistic_features_path: Path = DEFAULT_OPEN_REALISTIC_FEATURES_PATH
    labeled_path: Path = DEFAULT_LABELED_PATH
    event_definitions_path: Path = DEFAULT_EVENT_DEFINITIONS_PATH

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

    def load_realistic_snapshot_features(self, max_rows: int | None = None) -> pd.DataFrame:
        q = pl.scan_parquet(self.realistic_training_path)
        if max_rows is not None:
            q = q.head(max_rows)
        return q.collect().to_pandas()

    def load_open_realistic_features(self) -> pd.DataFrame:
        return pl.read_parquet(self.open_realistic_features_path).to_pandas()

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

        # Pre-filter labeled journeys to only the IDs in base before exploding.
        # Without this, exploding all 1.27M journeys (~63M events) then joining
        # is ~20x slower than joining first (to ~75k IDs) then exploding.
        base_ids = base.select(["id", "snapshot_num_actions"])
        snapshot_times = (
            pl.scan_parquet(self.labeled_path)
            .select(["id", "journey", "end_time"])
            .join(base_ids, on="id", how="inner")
            .explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
            .with_columns(pl.int_range(1, pl.len() + 1).over("id").alias("action_num"))
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

    def customer_action_states(self, include_success: bool = False) -> set[int]:
        return load_customer_action_states(
            self.event_definitions_path,
            include_success=include_success,
        )

    def events(
        self,
        max_journeys: int | None = None,
        customer_actions_only: bool = False,
        include_success: bool = True,
    ) -> pd.DataFrame:
        journeys = self.load_journeys(max_journeys=max_journeys)
        events = (
            journeys.select(["id", "journey"])
            .explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
        )
        pdf = events.to_pandas()
        pdf["event_timestamp"] = pd.to_datetime(pdf["event_timestamp"], utc=True)
        if customer_actions_only:
            state_set = self.customer_action_states(include_success=include_success)
            pdf = pdf[pdf["ed_id"].astype(int).isin(state_set)].copy()
        return pdf

    def transition_table(
        self,
        max_journeys: int | None = None,
        customer_actions_only: bool = True,
        include_success: bool = True,
    ) -> pd.DataFrame:
        events = self.events(
            max_journeys=max_journeys,
            customer_actions_only=customer_actions_only,
            include_success=include_success,
        )
        events["next_state"] = events.groupby("id")["ed_id"].shift(-1)
        events["next_timestamp"] = events.groupby("id")["event_timestamp"].shift(-1)
        transitions = events.dropna(subset=["next_state", "next_timestamp"]).copy()
        transitions["state"] = transitions["ed_id"].astype(int)
        transitions["next_state"] = transitions["next_state"].astype(int)
        transitions["dt_seconds"] = _as_seconds(
            transitions["next_timestamp"] - transitions["event_timestamp"]
        )
        return transitions[["id", "state", "next_state", "dt_seconds"]]

    def timeout_transition_table(
        self,
        max_journeys: int | None = None,
        customer_actions_only: bool = True,
        success_state: int = SUCCESS_STATE,
        failure_state: int = FAILURE_STATE,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ) -> pd.DataFrame:
        """Transitions with an explicit absorbing inactivity-failure state.

        Successful journeys contribute ordinary transitions into success_state.
        Incomplete journeys add one terminal transition from their last observed
        modeling state to failure_state after timeout_seconds. When
        customer_actions_only=True, company/system events are ignored as states
        and do not reset the inactivity clock.
        """
        q = (
            pl.scan_parquet(self.labeled_path)
            .filter(pl.col("journey_status").is_in(["successful", "incomplete"]))
            .select(["id", "journey", "journey_status"])
        )
        if max_journeys is not None:
            q = q.head(max_journeys)

        events = (
            q.explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
            .collect()
            .to_pandas()
        )
        events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
        events["ed_id"] = events["ed_id"].astype(int)

        if customer_actions_only:
            state_set = self.customer_action_states(include_success=True)
            events = events[events["ed_id"].isin(state_set)].copy()

        events["next_state"] = events.groupby("id")["ed_id"].shift(-1)
        events["next_timestamp"] = events.groupby("id")["event_timestamp"].shift(-1)
        transitions = events.dropna(subset=["next_state", "next_timestamp"]).copy()
        transitions["state"] = transitions["ed_id"].astype(int)
        transitions["next_state"] = transitions["next_state"].astype(int)
        transitions["dt_seconds"] = _as_seconds(
            transitions["next_timestamp"] - transitions["event_timestamp"]
        )
        transitions = transitions[["id", "state", "next_state", "dt_seconds"]]

        last_events = (
            events.sort_values(["id", "event_timestamp", "ed_id"])
            .groupby("id", as_index=False)
            .tail(1)
        )
        incomplete_last = last_events[last_events["journey_status"] == "incomplete"]
        fail_rows = pd.DataFrame(
            {
                "id": incomplete_last["id"].to_numpy(),
                "state": incomplete_last["ed_id"].astype(int).to_numpy(),
                "next_state": np.full(len(incomplete_last), failure_state, dtype=int),
                "dt_seconds": np.full(len(incomplete_last), float(timeout_seconds)),
            }
        )
        out = pd.concat([transitions, fail_rows], ignore_index=True)
        out = out[out["state"] != out["next_state"]].copy()
        return out[["id", "state", "next_state", "dt_seconds"]]

    def suffix_timeout_transition_table(
        self,
        snapshots: pd.DataFrame | None = None,
        max_rows: int | None = None,
        customer_actions_only: bool = True,
        success_state: int = SUCCESS_STATE,
        failure_state: int = FAILURE_STATE,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ) -> pd.DataFrame:
        """Build target-aligned suffix transitions from truncated snapshots.

        Each snapshot contributes transitions starting at the last observed
        customer-action state at snapshot_time. If the next modeling event is
        more than timeout_seconds away, the suffix absorbs in failure before
        that event can occur.
        """
        if snapshots is None:
            snapshots = self.load_realistic_snapshot_features(max_rows=max_rows)
        elif max_rows is not None:
            snapshots = snapshots.head(max_rows).copy()

        snapshots = snapshots.copy()
        snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], utc=True)
        if "snapshot_id" not in snapshots.columns:
            snapshots["snapshot_id"] = (
                snapshots["id"].astype(str) + "::" + snapshots["sample_num"].astype(str)
            )

        ids = pl.DataFrame({"id": snapshots["id"].drop_duplicates().to_list()}).lazy()
        events = (
            pl.scan_parquet(self.labeled_path)
            .select(["id", "journey"])
            .join(ids, on="id", how="inner")
            .explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
            .collect()
            .to_pandas()
        )
        events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
        events["ed_id"] = events["ed_id"].astype(int)
        if customer_actions_only:
            state_set = self.customer_action_states(include_success=True)
            events = events[events["ed_id"].isin(state_set)].copy()

        events_by_id = {
            journey_id: g.sort_values(["event_timestamp", "ed_id"])
            for journey_id, g in events.groupby("id", sort=False)
        }

        rows = []
        for snap in snapshots.itertuples(index=False):
            journey_id = getattr(snap, "id")
            snapshot_id = getattr(snap, "snapshot_id")
            snapshot_time = getattr(snap, "snapshot_time")
            journey_events = events_by_id.get(journey_id)
            if journey_events is None or journey_events.empty:
                continue

            observed = journey_events[journey_events["event_timestamp"] <= snapshot_time]
            if observed.empty:
                continue

            current_state = int(observed.iloc[-1]["ed_id"])
            current_time = snapshot_time
            future = journey_events[journey_events["event_timestamp"] > snapshot_time]
            absorbed = False

            for event in future.itertuples(index=False):
                next_state = int(event.ed_id)
                next_time = event.event_timestamp
                gap = max((next_time - current_time).total_seconds(), 0.0)
                if gap > timeout_seconds:
                    rows.append({
                        "snapshot_id": snapshot_id,
                        "id": journey_id,
                        "state": current_state,
                        "next_state": failure_state,
                        "dt_seconds": float(timeout_seconds),
                    })
                    absorbed = True
                    break

                rows.append({
                    "snapshot_id": snapshot_id,
                    "id": journey_id,
                    "state": current_state,
                    "next_state": next_state,
                    "dt_seconds": gap,
                })
                if next_state == success_state:
                    absorbed = True
                    break
                current_state = next_state
                current_time = next_time

            if not absorbed:
                rows.append({
                    "snapshot_id": snapshot_id,
                    "id": journey_id,
                    "state": current_state,
                    "next_state": failure_state,
                    "dt_seconds": float(timeout_seconds),
                })

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["snapshot_id", "id", "state", "next_state", "dt_seconds"])
        return out[out["state"] != out["next_state"]][
            ["snapshot_id", "id", "state", "next_state", "dt_seconds"]
        ].reset_index(drop=True)

    def prefix_transition_table(
        self,
        max_rows: int | None = None,
        customer_actions_only: bool = True,
        include_success: bool = True,
    ) -> pd.DataFrame:
        """Transition table from truncated journey prefixes — no label leakage.

        Reconstructs events up to the 70%-snapshot cutpoint from journeys_labeled.parquet
        so that current_state never reveals whether the journey reached the success state.
        Use this instead of transition_table() when building CTMC evaluation features.
        """
        base = pl.scan_parquet(self.training_path).select(["id", "snapshot_num_actions"])
        if max_rows is not None:
            base = base.head(max_rows)
        base_ids = base.collect().lazy()

        events = (
            pl.scan_parquet(self.labeled_path)
            .select(["id", "journey"])
            .join(base_ids, on="id", how="inner")
            .explode("journey")
            .unnest("journey")
            .sort(["id", "event_timestamp", "ed_id"])
            .with_columns(pl.int_range(1, pl.len() + 1).over("id").alias("action_num"))
            .filter(pl.col("action_num") <= pl.col("snapshot_num_actions"))
            .collect()
        )
        pdf = events.to_pandas()
        pdf["event_timestamp"] = pd.to_datetime(pdf["event_timestamp"], utc=True)
        if customer_actions_only:
            state_set = self.customer_action_states(include_success=include_success)
            pdf = pdf[pdf["ed_id"].astype(int).isin(state_set)].copy()
        pdf["next_state"] = pdf.groupby("id")["ed_id"].shift(-1)
        pdf["next_timestamp"] = pdf.groupby("id")["event_timestamp"].shift(-1)
        transitions = pdf.dropna(subset=["next_state", "next_timestamp"]).copy()
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


class HigherOrderCTMC:
    """Augmented-state CTMC using the last k customer actions as state.

    A second-order model uses states like (previous_action, current_action).
    A third-order model uses (two_back, previous_action, current_action).
    This keeps the CTMC rate-estimation machinery but lets the current state
    carry short action-history memory.
    """

    def __init__(
        self,
        order: int = 2,
        success_state: int = SUCCESS_STATE,
        start_state: int = -999,
        min_time: float = 1e-9,
    ) -> None:
        if order < 2:
            raise ValueError("HigherOrderCTMC requires order >= 2.")
        self.order = int(order)
        self.success_state = int(success_state)
        self.start_state = int(start_state)
        self.min_time = float(min_time)
        self.success_aug_state_: tuple[int, ...] = tuple([self.success_state] * self.order)
        self.states_: list[tuple[int, ...]] = []
        self.state_to_idx_: dict[tuple[int, ...], int] = {}
        self.Q_: np.ndarray | None = None
        self.transition_counts_: pd.DataFrame | None = None
        self.time_in_state_: pd.Series | None = None

    def fit_from_events(self, events: pd.DataFrame) -> "HigherOrderCTMC":
        """Fit from sorted event rows with columns id, event_timestamp, ed_id."""
        required = {"id", "event_timestamp", "ed_id"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"events missing required columns: {sorted(missing)}")

        df = events[["id", "event_timestamp", "ed_id"]].copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
        df["ed_id"] = df["ed_id"].astype(int)
        df = df.sort_values(["id", "event_timestamp", "ed_id"])

        counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        time_in_state: dict[tuple[int, ...], float] = {}
        states: set[tuple[int, ...]] = {self.success_aug_state_}

        for _, group in df.groupby("id", sort=False):
            action_ids = group["ed_id"].astype(int).to_list()
            times = group["event_timestamp"].to_list()
            if len(action_ids) < 2:
                continue

            for m in range(len(action_ids) - 1):
                current_history = self._history_from_prefix(action_ids[: m + 1])
                next_action = int(action_ids[m + 1])
                next_history = (
                    self.success_aug_state_
                    if next_action == self.success_state
                    else self._history_from_prefix(action_ids[: m + 2])
                )
                if current_history == next_history:
                    continue

                dt = max((times[m + 1] - times[m]).total_seconds(), 0.0)
                states.add(current_history)
                states.add(next_history)
                counts[(current_history, next_history)] = (
                    counts.get((current_history, next_history), 0) + 1
                )
                time_in_state[current_history] = time_in_state.get(current_history, 0.0) + dt

        self.states_ = sorted(states, key=lambda x: tuple(x))
        self.state_to_idx_ = {state: idx for idx, state in enumerate(self.states_)}
        n_states = len(self.states_)
        q = np.zeros((n_states, n_states), dtype=float)

        for (from_state, to_state), count in counts.items():
            i = self.state_to_idx_[from_state]
            j = self.state_to_idx_[to_state]
            exposure = max(time_in_state.get(from_state, 0.0), self.min_time)
            q[i, j] = count / exposure

        np.fill_diagonal(q, 0.0)
        q[np.diag_indices_from(q)] = -q.sum(axis=1)
        success_idx = self.state_to_idx_.get(self.success_aug_state_)
        if success_idx is not None:
            q[success_idx, :] = 0.0

        self.Q_ = q
        self.transition_counts_ = self._counts_frame(counts)
        self.time_in_state_ = pd.Series(time_in_state, dtype=float).reindex(
            self.states_, fill_value=0.0
        )
        return self

    def absorption_probability(
        self,
        histories: Iterable[Iterable[int] | tuple[int, ...]],
        success_state: int = SUCCESS_STATE,
        horizon_seconds: float = TIMEOUT_SECONDS,
        fallback_states: Iterable[int] | None = None,
        fallback_model: GlobalCTMC | None = None,
    ) -> np.ndarray:
        """Estimate P(hit success within horizon | current action history)."""
        self._check_fit()
        histories_list = list(histories)

        fallback_probs: np.ndarray
        if fallback_model is not None and fallback_states is not None:
            fallback_probs = fallback_model.absorption_probability(
                fallback_states,
                success_state=success_state,
                horizon_seconds=horizon_seconds,
            )
        else:
            fallback_probs = np.full(len(histories_list), self._empirical_success_prior())

        if self.success_aug_state_ not in self.state_to_idx_:
            return np.clip(fallback_probs, 0.0, 1.0)

        try:
            from scipy.linalg import expm
        except ImportError as exc:
            raise ImportError("scipy is required for higher-order CTMC predictions") from exc

        p = expm(self.Q_ * horizon_seconds)
        success_idx = self.state_to_idx_[self.success_aug_state_]

        probs = []
        for idx, history in enumerate(histories_list):
            state = self._history_from_prefix(list(history))
            state_idx = self.state_to_idx_.get(state)
            probs.append(
                float(fallback_probs[idx])
                if state_idx is None
                else float(p[state_idx, success_idx])
            )
        return np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)

    def histories_from_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return one current augmented history per open journey."""
        required = {"id", "event_timestamp", "ed_id"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"events missing required columns: {sorted(missing)}")

        df = events[["id", "event_timestamp", "ed_id"]].copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
        df["ed_id"] = df["ed_id"].astype(int)
        df = df.sort_values(["id", "event_timestamp", "ed_id"])

        rows = []
        for journey_id, group in df.groupby("id", sort=False):
            actions = group["ed_id"].astype(int).to_list()
            if not actions:
                continue
            history = self._history_from_prefix(actions)
            rows.append(
                {
                    "id": journey_id,
                    "current_state": int(actions[-1]),
                    "current_history": history,
                }
            )
        return pd.DataFrame(rows)

    def _history_from_prefix(self, actions: list[int]) -> tuple[int, ...]:
        clean = [int(a) for a in actions]
        if clean and clean[-1] == self.success_state:
            return self.success_aug_state_
        padded = [self.start_state] * max(0, self.order - len(clean)) + clean
        return tuple(padded[-self.order :])

    def _empirical_success_prior(self) -> float:
        if self.transition_counts_ is None or self.transition_counts_.empty:
            return 0.0
        success_col = repr(self.success_aug_state_)
        if success_col not in self.transition_counts_.columns:
            return 0.0
        total = float(self.transition_counts_.to_numpy().sum())
        if total <= 0:
            return 0.0
        return float(self.transition_counts_[success_col].sum() / total)

    def _counts_frame(
        self,
        counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int],
    ) -> pd.DataFrame:
        labels = [repr(state) for state in self.states_]
        frame = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
        for (from_state, to_state), count in counts.items():
            frame.loc[repr(from_state), repr(to_state)] = int(count)
        return frame

    def _check_fit(self) -> None:
        if self.Q_ is None:
            raise RuntimeError("Call fit_from_events() before reading CTMC outputs.")


class TimeoutAbsorbingCTMC(GlobalCTMC):
    """CTMC for P(success before 60-day inactivity failure).

    Fit this on CTMCData.timeout_transition_table(), which adds an explicit
    absorbing failure state for incomplete journeys. Prediction solves the
    absorbing CTMC linear system instead of asking only whether success is hit
    within a fixed calendar horizon.
    """

    def __init__(
        self,
        success_state: int = SUCCESS_STATE,
        failure_state: int = FAILURE_STATE,
        min_time: float = 1e-9,
    ) -> None:
        super().__init__(min_time=min_time)
        self.success_state = success_state
        self.failure_state = failure_state
        self.absorption_by_state_: dict[int, float] = {}
        self.fallback_: float = 0.0

    def fit(self, transitions: pd.DataFrame) -> "TimeoutAbsorbingCTMC":
        transitions = transitions[
            ~transitions["state"].astype(int).isin({self.success_state, self.failure_state})
        ].copy()
        super().fit(transitions)
        self._fit_absorption_probabilities()
        return self

    def predict_success_probability(
        self,
        current_states: Iterable[int],
        success_state: int | None = None,
        horizon_seconds: float | None = None,
    ) -> np.ndarray:
        return self.absorption_probability(
            current_states,
            success_state=success_state or self.success_state,
            horizon_seconds=TIMEOUT_SECONDS if horizon_seconds is None else horizon_seconds,
        )

    def absorption_probability(
        self,
        current_states: Iterable[int],
        success_state: int = SUCCESS_STATE,
        horizon_seconds: float = TIMEOUT_SECONDS,
    ) -> np.ndarray:
        self._check_fit()
        probs = [
            self.absorption_by_state_.get(int(state), self.fallback_)
            for state in current_states
        ]
        return np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)

    def _fit_absorption_probabilities(self) -> None:
        if self.Q_ is None:
            raise RuntimeError("Call fit() before computing absorption probabilities.")

        success = self.success_state
        failure = self.failure_state
        self.absorption_by_state_ = {success: 1.0, failure: 0.0}

        transient = [s for s in self.states_ if s not in {success, failure}]
        if transient and success in self.Q_.columns:
            q_tt = self.Q_.loc[transient, transient].to_numpy(dtype=float)
            q_ts = self.Q_.loc[transient, success].to_numpy(dtype=float)
            try:
                probs = np.linalg.solve(q_tt, -q_ts)
            except np.linalg.LinAlgError:
                probs = np.linalg.lstsq(q_tt, -q_ts, rcond=None)[0]
            for state, prob in zip(transient, probs):
                self.absorption_by_state_[state] = float(np.clip(prob, 0.0, 1.0))

        if self.transition_counts_ is not None:
            success_hits = (
                float(self.transition_counts_[success].sum())
                if success in self.transition_counts_.columns else 0.0
            )
            fail_hits = (
                float(self.transition_counts_[failure].sum())
                if failure in self.transition_counts_.columns else 0.0
            )
            denom = success_hits + fail_hits
            self.fallback_ = success_hits / denom if denom > 0 else 0.0


class SemiMarkovTimeoutModel:
    """Empirical semi-Markov model for success before inactivity timeout.

    This model does not assume exponential waiting times. It treats each
    observed outgoing transition as an empirical sample: if the next action
    occurs within timeout_seconds, the process moves to that next state; if no
    action occurs for timeout_seconds, it absorbs in failure.
    """

    def __init__(
        self,
        success_state: int = SUCCESS_STATE,
        failure_state: int = FAILURE_STATE,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_iter: int = 500,
        tol: float = 1e-10,
    ) -> None:
        self.success_state = success_state
        self.failure_state = failure_state
        self.timeout_seconds = timeout_seconds
        self.max_iter = max_iter
        self.tol = tol
        self.states_: list[int] = []
        self.samples_: pd.DataFrame | None = None
        self.success_prob_by_state_: dict[int, float] = {}
        self.fallback_: float = 0.0

    def fit(self, transitions: pd.DataFrame) -> "SemiMarkovTimeoutModel":
        df = transitions.copy()
        df = df[df["state"] != df["next_state"]].copy()
        df["state"] = df["state"].astype(int)
        df["next_state"] = df["next_state"].astype(int)
        df["dt_seconds"] = pd.to_numeric(df["dt_seconds"], errors="coerce").fillna(0.0)
        self.samples_ = df
        self.states_ = sorted(set(df["state"]).union(set(df["next_state"])))
        self._fit_value_function()
        return self

    def predict_success_probability(
        self,
        current_states: Iterable[int],
        success_state: int | None = None,
        horizon_seconds: float | None = None,
    ) -> np.ndarray:
        probs = [
            self.success_prob_by_state_.get(int(state), self.fallback_)
            for state in current_states
        ]
        return np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)

    def _fit_value_function(self) -> None:
        if self.samples_ is None:
            raise RuntimeError("Call fit() before computing success probabilities.")

        grouped = {
            int(state): g[["next_state", "dt_seconds"]].to_numpy()
            for state, g in self.samples_.groupby("state", sort=False)
        }
        values = {
            state: 0.5
            for state in self.states_
            if state not in {self.success_state, self.failure_state}
        }
        values[self.success_state] = 1.0
        values[self.failure_state] = 0.0

        for _ in range(self.max_iter):
            delta = 0.0
            updated = values.copy()
            for state, rows in grouped.items():
                if state in {self.success_state, self.failure_state}:
                    continue
                sample_values = []
                for next_state, dt_seconds in rows:
                    next_state = int(next_state)
                    if float(dt_seconds) > self.timeout_seconds:
                        sample_values.append(0.0)
                    elif next_state == self.success_state:
                        sample_values.append(1.0)
                    elif next_state == self.failure_state:
                        sample_values.append(0.0)
                    else:
                        sample_values.append(values.get(next_state, self.fallback_))
                new_value = float(np.mean(sample_values)) if sample_values else self.fallback_
                updated[state] = new_value
                delta = max(delta, abs(new_value - values.get(state, 0.0)))
            values = updated
            if delta < self.tol:
                break

        terminal = self.samples_[self.samples_["next_state"].isin([self.success_state, self.failure_state])]
        denom = len(terminal)
        self.fallback_ = (
            float((terminal["next_state"] == self.success_state).mean())
            if denom > 0 else 0.0
        )
        self.success_prob_by_state_ = {
            int(state): float(np.clip(prob, 0.0, 1.0))
            for state, prob in values.items()
        }


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
        df = transitions.sort_values("id").copy()

        # Cumulative time within each journey — used for time_to_action_*.
        df["_cumtime"] = df.groupby("id", sort=False)["dt_seconds"].cumsum()

        # Base per-journey aggregates (one groupby pass).
        base = (
            df.groupby("id", sort=False)
            .agg(
                num_transitions=("state", "count"),
                total_observed_time=("dt_seconds", "sum"),
                first_state=("state", "first"),
                current_state=("next_state", "last"),
            )
            .reset_index()
        )
        base["avg_gap_seconds"] = (
            base["total_observed_time"] / base["num_transitions"].clip(lower=1)
        )

        # Action counts and dwell times — all top actions in two pivot ops.
        act_sub = df[df["state"].isin(self.top_actions_)]
        count_pivot = (
            act_sub.groupby(["id", "state"])["state"].count()
            .unstack(fill_value=0)
            .rename(columns=lambda c: f"action_count_{c}")
        )
        time_pivot = (
            act_sub.groupby(["id", "state"])["dt_seconds"].sum()
            .unstack(fill_value=0.0)
            .rename(columns=lambda c: f"time_in_state_{c}")
        )
        base = base.merge(count_pivot.reset_index(), on="id", how="left")
        base = base.merge(time_pivot.reset_index(), on="id", how="left")

        total_actions = base["num_transitions"] + 1  # states visited = transitions + 1
        for action in self.top_actions_:
            cc = f"action_count_{action}"
            tc = f"time_in_state_{action}"
            if cc not in base.columns:
                base[cc] = 0
            if tc not in base.columns:
                base[tc] = 0.0
            base[cc] = base[cc].fillna(0).astype(int)
            base[tc] = base[tc].fillna(0.0)
            base[f"action_prop_{action}"] = base[cc] / total_actions.clip(lower=1)

        # Key action features — cumtime at first hit.
        for action in self.key_actions:
            hit = (
                df[df["next_state"] == action]
                .groupby("id", sort=False)["_cumtime"]
                .first()
                .rename(f"time_to_action_{action}")
                .reset_index()
            )
            base = base.merge(hit, on="id", how="left")
            base[f"seen_action_{action}"] = (
                base[f"time_to_action_{action}"].notna().astype(int)
            )

        ordered = (
            ["id", "num_transitions", "total_observed_time", "avg_gap_seconds",
             "first_state", "current_state"]
            + [c for a in self.top_actions_
               for c in [f"action_count_{a}", f"action_prop_{a}", f"time_in_state_{a}"]]
            + [c for a in self.key_actions
               for c in [f"time_to_action_{a}", f"seen_action_{a}"]]
        )
        return base[[c for c in ordered if c in base.columns]].fillna(-1)

    def features_from_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """Build the same feature surface from open/test event rows."""
        df = events.sort_values(["id", "event_timestamp"]).copy()

        df["gap_seconds"] = (
            df.groupby("id", sort=False)["event_timestamp"]
            .diff()
            .dt.total_seconds()
            .clip(lower=0)
            .fillna(0.0)
        )
        df["_cumtime"] = df.groupby("id", sort=False)["gap_seconds"].cumsum()

        base = (
            df.groupby("id", sort=False)
            .agg(
                total_actions=("ed_id", "count"),
                total_observed_time=("gap_seconds", "sum"),
                first_state=("ed_id", "first"),
                current_state=("ed_id", "last"),
            )
            .reset_index()
        )
        base["num_transitions"] = (base["total_actions"] - 1).clip(lower=0)
        base["avg_gap_seconds"] = (
            base["total_observed_time"] / base["num_transitions"].clip(lower=1)
        )

        act_sub = df[df["ed_id"].isin(self.top_actions_)]
        count_pivot = (
            act_sub.groupby(["id", "ed_id"])["ed_id"].count()
            .unstack(fill_value=0)
            .rename(columns=lambda c: f"action_count_{c}")
        )
        time_pivot = (
            act_sub.groupby(["id", "ed_id"])["gap_seconds"].sum()
            .unstack(fill_value=0.0)
            .rename(columns=lambda c: f"time_in_state_{c}")
        )
        base = base.merge(count_pivot.reset_index(), on="id", how="left")
        base = base.merge(time_pivot.reset_index(), on="id", how="left")

        for action in self.top_actions_:
            cc = f"action_count_{action}"
            tc = f"time_in_state_{action}"
            if cc not in base.columns:
                base[cc] = 0
            if tc not in base.columns:
                base[tc] = 0.0
            base[cc] = base[cc].fillna(0).astype(int)
            base[tc] = base[tc].fillna(0.0)
            base[f"action_prop_{action}"] = base[cc] / base["total_actions"].clip(lower=1)

        for action in self.key_actions:
            hit = (
                df[df["ed_id"] == action]
                .groupby("id", sort=False)["_cumtime"]
                .first()
                .rename(f"time_to_action_{action}")
                .reset_index()
            )
            base = base.merge(hit, on="id", how="left")
            base[f"seen_action_{action}"] = (
                base[f"time_to_action_{action}"].notna().astype(int)
            )

        ordered = (
            ["id", "num_transitions", "total_observed_time", "avg_gap_seconds",
             "first_state", "current_state"]
            + [c for a in self.top_actions_
               for c in [f"action_count_{a}", f"action_prop_{a}", f"time_in_state_{a}"]]
            + [c for a in self.key_actions
               for c in [f"time_to_action_{a}", f"seen_action_{a}"]]
        )
        return base[[c for c in ordered if c in base.columns]].fillna(-1)


class ClusteredCTMC:
    """Cluster journeys, then fit one CTMC generator per cluster.

    Set use_spectral_clustering=True to replace the original KMeans segmenter
    with nearest-neighbor spectral clustering. SpectralClustering has no
    predict() method, so fitting also trains a KNN assignment model in the same
    preprocessed feature space for future/open journeys.

    When timeout_absorbing=True, clusters are still learned from ordinary
    customer-action transitions, but the per-cluster generators are fit on
    timeout-augmented transitions with absorbing success and inactivity states.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        use_spectral_clustering: bool = False,
        cluster_feature_mode: str = "auto",
        timeout_absorbing: bool = False,
        spectral_n_neighbors: int = 20,
        spectral_assign_neighbors: int = 25,
        spectral_max_fit_rows: int = 25_000,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.use_spectral_clustering = use_spectral_clustering
        self.cluster_feature_mode = cluster_feature_mode
        self.timeout_absorbing = timeout_absorbing
        self.spectral_n_neighbors = spectral_n_neighbors
        self.spectral_assign_neighbors = spectral_assign_neighbors
        self.spectral_max_fit_rows = spectral_max_fit_rows
        self.feature_builder = JourneyFeatureBuilder()
        self.models_: dict[int, GlobalCTMC] = {}
        self.assignments_: pd.DataFrame | None = None
        self.pipeline_ = None
        self.clusterer_ = None
        self.assignment_model_ = None
        self.feature_columns_: list[str] = []
        self.cluster_columns_: list[str] = []

    def fit(
        self,
        transitions: pd.DataFrame,
        model_transitions: pd.DataFrame | None = None,
    ) -> "ClusteredCTMC":
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.impute import SimpleImputer
        from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse.csgraph import connected_components

        self.models_ = {}
        self.assignment_model_ = None

        features = self.feature_builder.fit_transform(transitions)
        x = features.drop(columns=["id"])
        self.feature_columns_ = x.columns.tolist()
        self.cluster_columns_ = self._select_cluster_columns(x)

        x_cluster = self._aligned_cluster_frame(x)
        self.pipeline_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        x_prepared = self.pipeline_.fit_transform(x_cluster)

        if self.use_spectral_clustering:
            fit_idx = np.arange(len(x_cluster))
            if len(fit_idx) > self.spectral_max_fit_rows:
                rng = np.random.default_rng(self.random_state)
                fit_idx = np.sort(
                    rng.choice(fit_idx, size=self.spectral_max_fit_rows, replace=False)
                )
            x_spectral = x_prepared[fit_idx]
            if self.n_clusters > len(x_spectral):
                raise ValueError("n_clusters cannot exceed the number of spectral fit rows.")

            n_neighbors = SnapshotFeatureClusteredCTMC._connected_neighbor_count(
                x_spectral,
                initial=self.spectral_n_neighbors,
                max_neighbors=min(150, max(1, len(x_spectral) - 1)),
                kneighbors_graph=kneighbors_graph,
                connected_components=connected_components,
            )
            self.clusterer_ = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                assign_labels="kmeans",
                random_state=self.random_state,
                n_init=10,
            )
            spectral_labels = self.clusterer_.fit_predict(x_spectral)

            assign_neighbors = min(
                max(1, self.spectral_assign_neighbors),
                max(1, len(x_spectral)),
            )
            self.assignment_model_ = KNeighborsClassifier(
                n_neighbors=assign_neighbors,
                weights="distance",
            )
            self.assignment_model_.fit(x_spectral, spectral_labels)
            clusters = self.assignment_model_.predict(x_prepared)
        else:
            self.clusterer_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            clusters = self.clusterer_.fit_predict(x_prepared)

        assignments = features[["id"]].copy()
        assignments["cluster"] = clusters
        self.assignments_ = assignments

        transitions_for_models = model_transitions if model_transitions is not None else transitions
        clustered_transitions = transitions_for_models.merge(assignments, on="id", how="inner")
        model_cls = TimeoutAbsorbingCTMC if self.timeout_absorbing else GlobalCTMC
        for cluster_id, g in clustered_transitions.groupby("cluster"):
            self.models_[int(cluster_id)] = model_cls().fit(g)
        return self

    def _select_cluster_columns(self, x: pd.DataFrame) -> list[str]:
        if self.cluster_feature_mode not in {"auto", "all_numeric"}:
            raise ValueError("cluster_feature_mode must be 'auto' or 'all_numeric'")

        numeric_cols = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric journey features available for clustering.")

        if self.cluster_feature_mode == "all_numeric":
            return numeric_cols

        prop_cols = [c for c in numeric_cols if c.startswith("action_prop_")]
        state_cols = [c for c in ["first_state", "current_state"] if c in numeric_cols]
        if not self.use_spectral_clustering:
            # Preserve the original KMeans feature surface by default.
            return prop_cols + state_cols if prop_cols or state_cols else numeric_cols

        direct_success_cols = {"time_to_action_28", "seen_action_28"}
        stable_prefix_cols = [
            c for c in numeric_cols
            if c not in direct_success_cols
            and not c.startswith("label")
            and c != "final_outcome"
        ]
        return stable_prefix_cols if stable_prefix_cols else numeric_cols

    def _aligned_cluster_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=features.index)
        for col in self.cluster_columns_:
            x[col] = features[col] if col in features.columns else 0.0
        x = x.apply(pd.to_numeric, errors="coerce")
        return x.replace([np.inf, -np.inf], np.nan)

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
        if self.pipeline_ is None or self.clusterer_ is None:
            raise RuntimeError("Call fit() before predicting clusters.")
        x = self._aligned_cluster_frame(features)
        x_prepared = self.pipeline_.transform(x)
        if self.use_spectral_clustering:
            if self.assignment_model_ is None:
                raise RuntimeError("Spectral assignment model is missing; refit the model.")
            return self.assignment_model_.predict(x_prepared)
        return self.clusterer_.predict(x_prepared)

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
            current_states = features.loc[mask, "current_state"]
            if isinstance(model, TimeoutAbsorbingCTMC):
                probs[mask] = model.predict_success_probability(current_states)
            else:
                probs[mask] = model.absorption_probability(
                    current_states,
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


class SnapshotFeatureClusteredCTMC:
    """Cluster truncated snapshots using fully preprocessed features.

    Clustering uses the realistic truncation feature table, while each cluster
    gets a TimeoutAbsorbingCTMC fit on suffix transitions after the snapshot.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        use_spectral_clustering: bool = False,
        spectral_n_neighbors: int = 50,
        spectral_assign_neighbors: int = 25,
        spectral_max_fit_rows: int = 25_000,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.use_spectral_clustering = use_spectral_clustering
        self.spectral_n_neighbors = spectral_n_neighbors
        self.spectral_assign_neighbors = spectral_assign_neighbors
        self.spectral_max_fit_rows = spectral_max_fit_rows
        self.feature_columns_: list[str] = []
        self.encoded_columns_: list[str] = []
        self.pipeline_ = None
        self.clusterer_ = None
        self.assignment_model_ = None
        self.assignments_: pd.DataFrame | None = None
        self.models_: dict[int, TimeoutAbsorbingCTMC] = {}
        self.global_fallback_: TimeoutAbsorbingCTMC | None = None

    def fit(
        self,
        snapshots: pd.DataFrame,
        suffix_transitions: pd.DataFrame,
    ) -> "SnapshotFeatureClusteredCTMC":
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.impute import SimpleImputer
        from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse.csgraph import connected_components

        self.models_ = {}
        self.assignment_model_ = None

        features = snapshots.copy()
        if "snapshot_id" not in features.columns:
            features["snapshot_id"] = (
                features["id"].astype(str) + "::" + features["sample_num"].astype(str)
            )

        x = self._feature_frame(features, fit=True)
        self.pipeline_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        x_prepared = self.pipeline_.fit_transform(x)

        if self.use_spectral_clustering:
            fit_idx = np.arange(len(x_prepared))
            if len(fit_idx) > self.spectral_max_fit_rows:
                rng = np.random.default_rng(self.random_state)
                fit_idx = np.sort(
                    rng.choice(fit_idx, size=self.spectral_max_fit_rows, replace=False)
                )
            x_spectral = x_prepared[fit_idx]
            n_neighbors = self._connected_neighbor_count(
                x_spectral,
                initial=self.spectral_n_neighbors,
                max_neighbors=min(150, max(1, len(x_spectral) - 1)),
                kneighbors_graph=kneighbors_graph,
                connected_components=connected_components,
            )
            self.clusterer_ = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                assign_labels="kmeans",
                random_state=self.random_state,
                n_init=10,
            )
            labels_fit = self.clusterer_.fit_predict(x_spectral)
            assign_neighbors = min(
                max(1, self.spectral_assign_neighbors),
                max(1, len(x_spectral)),
            )
            self.assignment_model_ = KNeighborsClassifier(
                n_neighbors=assign_neighbors,
                weights="distance",
            )
            self.assignment_model_.fit(x_spectral, labels_fit)
            clusters = self.assignment_model_.predict(x_prepared)
        else:
            self.clusterer_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            clusters = self.clusterer_.fit_predict(x_prepared)

        assignments = features[["snapshot_id"]].copy()
        assignments["cluster"] = clusters
        self.assignments_ = assignments

        self.global_fallback_ = TimeoutAbsorbingCTMC().fit(suffix_transitions)
        clustered_transitions = suffix_transitions.merge(assignments, on="snapshot_id", how="inner")
        for cluster_id, group in clustered_transitions.groupby("cluster"):
            self.models_[int(cluster_id)] = TimeoutAbsorbingCTMC().fit(group)
        return self

    def predict_success_probability(self, snapshots: pd.DataFrame) -> np.ndarray:
        if self.global_fallback_ is None:
            raise RuntimeError("Call fit() before predicting.")
        clusters = self.predict_clusters(snapshots)
        current_states = snapshots["current_state"] if "current_state" in snapshots.columns else snapshots["last_ed_id"]
        probs = np.zeros(len(snapshots), dtype=float)
        for cluster_id in sorted(set(clusters)):
            mask = clusters == cluster_id
            model = self.models_.get(int(cluster_id), self.global_fallback_)
            probs[mask] = model.predict_success_probability(current_states.loc[mask])
        return np.clip(probs, 0.0, 1.0)

    def predict_clusters(self, snapshots: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None or self.clusterer_ is None:
            raise RuntimeError("Call fit() before predicting clusters.")
        x = self._feature_frame(snapshots, fit=False)
        x_prepared = self.pipeline_.transform(x)
        if self.use_spectral_clustering:
            if self.assignment_model_ is None:
                raise RuntimeError("Spectral assignment model is missing.")
            return self.assignment_model_.predict(x_prepared)
        return self.clusterer_.predict(x_prepared)

    def cluster_summary(self) -> pd.DataFrame:
        if self.assignments_ is None:
            raise RuntimeError("Call fit() before reading cluster summaries.")
        return (
            self.assignments_["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="n_snapshots")
        )

    def _feature_frame(self, snapshots: pd.DataFrame, fit: bool) -> pd.DataFrame:
        drop_cols = {
            "id", "sample_num", "snapshot_id", "journey_status", "source_table",
            "label", "snapshot_time", "first_event_time", "last_event_time",
            "prefix_actions",
        }
        base = snapshots.drop(columns=[c for c in drop_cols if c in snapshots.columns]).copy()
        encoded = pd.get_dummies(base, dummy_na=True)
        encoded = encoded.apply(pd.to_numeric, errors="coerce")
        if fit:
            self.encoded_columns_ = encoded.columns.tolist()
            return encoded.replace([np.inf, -np.inf], np.nan)
        for col in self.encoded_columns_:
            if col not in encoded.columns:
                encoded[col] = 0.0
        return encoded[self.encoded_columns_].replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _connected_neighbor_count(
        x,
        initial: int,
        max_neighbors: int,
        kneighbors_graph,
        connected_components,
    ) -> int:
        if len(x) <= 2:
            return 1
        n_neighbors = min(max(1, initial), max_neighbors)
        while n_neighbors < max_neighbors:
            graph = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=False)
            n_components, _ = connected_components(graph, directed=False)
            if n_components == 1:
                return n_neighbors
            n_neighbors = min(max_neighbors, max(n_neighbors + 1, int(n_neighbors * 1.5)))
        return n_neighbors


class XGBoostRateCTMC:
    """
    CTMC with a XGBoost-predicted per-journey rate scaling factor.

    The global Q matrix (direction and relative rates between states) is kept
    fixed. XGBoost predicts one scalar α per journey: how much faster or slower
    that user transitions relative to the global average.

        Q(x) = α(x) · Q_global
        P(absorbed by t | x, s) = expm(α(x) · Q_global · t)[s, success_state]

    α is estimated via proper MLE for the exponential holding-time model.
    For journey j with transitions from state i having holding time t:

        α_j = n_j / Σ_ij (q_i · t_ij)

    where q_i = -Q_global[i,i] is the global exit rate from state i.

    Under this model α=1 recovers GlobalCTMC exactly. α>1 means the user
    moves faster than average (higher absorption probability); α<1 means slower.

    Training target: one log(α_j) per journey — low variance, well-identified.
    Previous design trained on log(1/dt) per individual transition, which has
    Var = π²/6 ≈ 1.65 regardless of the true rate — an irreducibly noisy target.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.device = device
        self.feature_builder = JourneyFeatureBuilder()
        self.global_ctmc_: GlobalCTMC | None = None
        self.scale_model_ = None
        self.states_: list[int] = []
        self.feature_cols_: list[str] = []

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import xgboost as xgb
            dm = xgb.DMatrix([[0.0]], label=[0])
            xgb.train({"device": "cuda", "verbosity": 0}, dm, num_boost_round=1)
            print("  XGBoostRateCTMC: CUDA GPU detected, using device=cuda")
            return "cuda"
        except Exception:
            print("  XGBoostRateCTMC: no CUDA GPU found, using device=cpu")
            return "cpu"

    def fit(self, transitions: pd.DataFrame) -> "XGBoostRateCTMC":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBoostRateCTMC") from exc

        device = self._resolve_device(self.device)

        # Fit global CTMC — this defines Q_global and q_i for every state.
        self.global_ctmc_ = GlobalCTMC().fit(transitions)
        self.states_ = self.global_ctmc_.states_

        # Global exit rate q_i = -Q_ii for each state.
        q_by_state = {
            s: max(float(-self.global_ctmc_.Q_.loc[s, s]), 1e-12)
            for s in self.states_
        }

        # Per-journey MLE of rate scale α:
        #   α_j = n_j / Σ_{transitions in j} (q_{from_state} · dt_seconds)
        # Under Exp(α · q_i), this is the exact MLE for α.
        df = transitions[transitions["state"] != transitions["next_state"]].copy()
        df["q_from"] = df["state"].map(q_by_state).fillna(0.0)
        df["weighted_dt"] = df["q_from"] * df["dt_seconds"]

        per_journey = (
            df.groupby("id")
            .agg(n_transitions=("dt_seconds", "count"),
                 sum_weighted_dt=("weighted_dt", "sum"))
        )
        per_journey["alpha"] = (
            per_journey["n_transitions"] / per_journey["sum_weighted_dt"].clip(lower=1e-9)
        )
        per_journey["log_alpha"] = np.log(per_journey["alpha"].clip(lower=1e-6, upper=1e6))

        # Journey-level features (one row per journey).
        journey_feats = self.feature_builder.fit_transform(transitions)
        self.feature_cols_ = [c for c in journey_feats.columns if c != "id"]

        train = journey_feats.merge(per_journey[["log_alpha"]], on="id", how="inner")
        X = train[self.feature_cols_]
        y = train["log_alpha"].to_numpy()

        self.scale_model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            device=device,
            n_jobs=-1,
            verbosity=0,
        )
        self.scale_model_.fit(X, y)
        return self

    def predict_success_probability(
        self,
        features: pd.DataFrame,
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> np.ndarray:
        from scipy.linalg import expm

        if self.scale_model_ is None or self.global_ctmc_ is None:
            raise RuntimeError("Call fit() before predicting.")
        if success_state not in self.states_:
            return np.zeros(len(features))

        state_to_idx = {s: i for i, s in enumerate(self.states_)}
        success_idx = state_to_idx[success_state]

        # Align feature columns — fill missing with 0.
        X = pd.DataFrame(index=features.index)
        for col in self.feature_cols_:
            X[col] = features[col] if col in features.columns else 0.0

        log_alphas = self.scale_model_.predict(X)
        alphas = np.exp(np.clip(log_alphas, -6, 6))  # cap at ≈400× speed-up/slow-down

        # Base Q with success state made absorbing.
        Q_base = self.global_ctmc_.Q_.to_numpy().copy()
        Q_base[success_idx] = 0.0

        current_states = features["current_state"].to_numpy(dtype=int)
        probs = np.zeros(len(features))

        for i, (alpha, current_state) in enumerate(zip(alphas, current_states)):
            current_idx = state_to_idx.get(int(current_state))
            if current_idx is None:
                continue
            P_t = expm(alpha * Q_base * horizon_seconds)
            probs[i] = float(P_t[current_idx, success_idx])

        return np.clip(probs, 0.0, 1.0)


class TimeStratifiedCTMC:
    """
    Non-homogeneous CTMC: one Q matrix per journey-age window.

    Each transition is placed into a time window based on cumulative elapsed
    time within its journey. Prediction selects the Q matrix matching the test
    journey's current age, then computes absorption probability over the full
    remaining horizon.

    Default windows (edges at 1 day and 7 days):
        early : 0 – 1 day
        mid   : 1 – 7 days
        late  : 7+ days

    If a bin has fewer than min_bin_transitions transitions it falls back to the
    global (all-data) Q so sparse late-journey data never causes a degenerate matrix.
    """

    def __init__(
        self,
        bin_edges_seconds: tuple[float, ...] = (86_400.0, 604_800.0),
        min_bin_transitions: int = 200,
    ) -> None:
        self.bin_edges_seconds = tuple(sorted(bin_edges_seconds))
        self.min_bin_transitions = min_bin_transitions
        self.feature_builder = JourneyFeatureBuilder()
        self.models_: dict[int, GlobalCTMC] = {}
        self.global_fallback_: GlobalCTMC | None = None
        self.n_bins_: int = len(self.bin_edges_seconds) + 1

    def _to_bin(self, seconds: np.ndarray | pd.Series) -> np.ndarray:
        return np.digitize(np.asarray(seconds, dtype=float), self.bin_edges_seconds, right=False)

    def fit(self, transitions: pd.DataFrame) -> "TimeStratifiedCTMC":
        df = transitions[transitions["state"] != transitions["next_state"]].copy()
        df = df.sort_values("id")
        df["_cumtime"] = df.groupby("id", sort=False)["dt_seconds"].cumsum()
        df["_bin"] = self._to_bin(df["_cumtime"].to_numpy())

        self.global_fallback_ = GlobalCTMC().fit(
            df.drop(columns=["_cumtime", "_bin"])
        )
        self.n_bins_ = len(self.bin_edges_seconds) + 1

        for b in range(self.n_bins_):
            bin_data = df[df["_bin"] == b].drop(columns=["_cumtime", "_bin"])
            if len(bin_data) >= self.min_bin_transitions:
                self.models_[b] = GlobalCTMC().fit(bin_data)
            else:
                self.models_[b] = self.global_fallback_

        return self

    def predict_success_probability(
        self,
        features: pd.DataFrame,
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> np.ndarray:
        """
        features must include 'current_state' and 'total_observed_time'.
        total_observed_time selects which Q matrix to use per journey.
        """
        if not self.models_:
            raise RuntimeError("Call fit() before predicting.")

        total_times = features["total_observed_time"].to_numpy(dtype=float)
        current_states = features["current_state"].to_numpy(dtype=int)
        bin_idx = self._to_bin(total_times)

        probs = np.zeros(len(features))
        for b in sorted(set(bin_idx)):
            mask = bin_idx == b
            model = self.models_.get(int(b), self.global_fallback_)
            probs[mask] = model.absorption_probability(
                current_states[mask],
                success_state=success_state,
                horizon_seconds=horizon_seconds,
            )

        return np.clip(probs, 0.0, 1.0)

    def bin_summary(self) -> pd.DataFrame:
        edges = list(self.bin_edges_seconds)
        rows = []
        for b in range(self.n_bins_):
            if b == 0:
                label = f"0 – {edges[0] / 3600:.1f}h"
            elif b < len(edges):
                label = f"{edges[b-1] / 3600:.1f}h – {edges[b] / 86400:.1f}d"
            else:
                label = f"{edges[-1] / 86400:.1f}d +"
            model = self.models_.get(b, self.global_fallback_)
            n_tr = (
                int(model.transition_counts_.to_numpy().sum())
                if model is not None and model.transition_counts_ is not None
                else 0
            )
            rows.append({
                "bin": b,
                "window": label,
                "n_transitions": n_tr,
                "n_states": len(model.states_) if model else 0,
            })
        return pd.DataFrame(rows)


class PiecewiseTimeVaryingTimeoutCTMC:
    """Piecewise-constant non-homogeneous timeout-aware CTMC.

    The model estimates one generator matrix Q_b per journey-age bin. Exposure
    time is split across bins, while an observed jump i -> j is counted in the
    bin where the jump occurs. Prediction propagates a probability row vector
    through every crossed bin using matrix exponentials.
    """

    def __init__(
        self,
        bin_edges_seconds: tuple[float, ...] = (
            86_400.0,        # 1 day
            604_800.0,       # 7 days
            2_592_000.0,     # 30 days
            5_184_000.0,     # 60 days
        ),
        success_state: int = SUCCESS_STATE,
        failure_state: int = FAILURE_STATE,
        min_state_transitions: int = 20,
        min_exposure_seconds: float = 3_600.0,
        min_time: float = 1e-9,
    ) -> None:
        self.bin_edges_seconds = tuple(sorted(float(x) for x in bin_edges_seconds))
        self.success_state = int(success_state)
        self.failure_state = int(failure_state)
        self.min_state_transitions = int(min_state_transitions)
        self.min_exposure_seconds = float(min_exposure_seconds)
        self.min_time = float(min_time)

        self.feature_builder = JourneyFeatureBuilder()
        self.n_bins_: int = len(self.bin_edges_seconds) + 1
        self.states_: list[int] = []
        self.state_to_idx_: dict[int, int] = {}
        self.Q_by_bin_: dict[int, pd.DataFrame] = {}
        self.counts_by_bin_: dict[int, pd.DataFrame] = {}
        self.exposure_by_bin_: dict[int, pd.Series] = {}
        self.row_fallback_by_bin_: dict[int, pd.Series] = {}
        self.global_Q_: pd.DataFrame | None = None
        self._eigendecomp_by_bin_: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
        self.fallback_: float = 0.0

    def _to_bin(self, seconds: np.ndarray | pd.Series | list[float]) -> np.ndarray:
        return np.digitize(np.asarray(seconds, dtype=float), self.bin_edges_seconds, right=False)

    def fit(
        self,
        transitions: pd.DataFrame,
        feature_transitions: pd.DataFrame | None = None,
    ) -> "PiecewiseTimeVaryingTimeoutCTMC":
        """Fit bin-specific Q matrices from timeout-aware transitions."""
        df = transitions.copy()
        df = df[df["state"] != df["next_state"]].copy()
        df["state"] = df["state"].astype(int)
        df["next_state"] = df["next_state"].astype(int)
        df["dt_seconds"] = pd.to_numeric(df["dt_seconds"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df = df[~df["state"].isin({self.success_state, self.failure_state})].copy()
        df["_row_order"] = np.arange(len(df))
        df = df.sort_values(["id", "_row_order"])

        if feature_transitions is not None:
            self.feature_builder.fit_transform(feature_transitions)
        else:
            self.feature_builder.fit_transform(df.drop(columns=["_row_order"]))

        states = sorted(
            set(df["state"].astype(int))
            .union(set(df["next_state"].astype(int)))
            .union({self.success_state, self.failure_state})
        )
        self.states_ = states
        self.state_to_idx_ = {state: idx for idx, state in enumerate(states)}

        n_states = len(states)
        counts = np.zeros((self.n_bins_, n_states, n_states), dtype=float)
        exposure = np.zeros((self.n_bins_, n_states), dtype=float)

        for _, group in df.groupby("id", sort=False):
            elapsed = 0.0
            for row in group.itertuples(index=False):
                state = int(row.state)
                next_state = int(row.next_state)
                dt = float(row.dt_seconds)
                start = elapsed
                end = elapsed + dt

                state_idx = self.state_to_idx_[state]
                for bin_idx, overlap in self._interval_bin_overlaps(start, end):
                    exposure[bin_idx, state_idx] += overlap

                jump_bin = int(self._to_bin([end])[0])
                jump_bin = min(max(jump_bin, 0), self.n_bins_ - 1)
                if next_state in self.state_to_idx_:
                    next_idx = self.state_to_idx_[next_state]
                    counts[jump_bin, state_idx, next_idx] += 1.0

                elapsed = end

        global_counts = counts.sum(axis=0)
        global_exposure = exposure.sum(axis=0)
        global_q = self._estimate_q_from_counts(global_counts, global_exposure)
        self.global_Q_ = pd.DataFrame(global_q, index=states, columns=states)

        success_idx = self.state_to_idx_[self.success_state]
        failure_idx = self.state_to_idx_[self.failure_state]

        for bin_idx in range(self.n_bins_):
            q = np.zeros((n_states, n_states), dtype=float)
            fallback_mask = np.zeros(n_states, dtype=bool)
            outgoing_counts = counts[bin_idx].sum(axis=1)

            for state, state_idx in self.state_to_idx_.items():
                if state in {self.success_state, self.failure_state}:
                    continue

                enough_data = (
                    exposure[bin_idx, state_idx] >= self.min_exposure_seconds
                    and outgoing_counts[state_idx] >= self.min_state_transitions
                )
                if enough_data:
                    row_q = counts[bin_idx, state_idx] / max(exposure[bin_idx, state_idx], self.min_time)
                    row_q[state_idx] = 0.0
                    row_q[state_idx] = -row_q.sum()
                    q[state_idx] = row_q
                else:
                    q[state_idx] = global_q[state_idx]
                    fallback_mask[state_idx] = True

            q[success_idx, :] = 0.0
            q[failure_idx, :] = 0.0

            self.Q_by_bin_[bin_idx] = pd.DataFrame(q, index=states, columns=states)
            self.counts_by_bin_[bin_idx] = pd.DataFrame(counts[bin_idx], index=states, columns=states)
            self.exposure_by_bin_[bin_idx] = pd.Series(exposure[bin_idx], index=states)
            self.row_fallback_by_bin_[bin_idx] = pd.Series(fallback_mask, index=states)

        self._fit_eigendecompositions()

        terminal_counts = global_counts[:, success_idx].sum() + global_counts[:, failure_idx].sum()
        self.fallback_ = (
            float(global_counts[:, success_idx].sum() / terminal_counts)
            if terminal_counts > 0 else 0.0
        )
        return self

    def predict_success_probability(
        self,
        features: pd.DataFrame,
        success_state: int = SUCCESS_STATE,
        horizon_seconds: float = TIMEOUT_SECONDS,
    ) -> np.ndarray:
        """Predict P(order_shipped by horizon) under piecewise Q(t)."""
        self._check_fit()
        if success_state != self.success_state:
            raise ValueError("PiecewiseTimeVaryingTimeoutCTMC only supports its fitted success_state.")

        if "current_state" not in features.columns or "total_observed_time" not in features.columns:
            raise ValueError("features must contain current_state and total_observed_time")

        if not self._eigendecomp_by_bin_:
            self._fit_eigendecompositions()

        success_idx = self.state_to_idx_.get(self.success_state)
        if success_idx is None:
            return np.zeros(len(features), dtype=float)

        probs = np.zeros(len(features), dtype=float)
        current_states = features["current_state"].to_numpy(dtype=int)
        current_ages = features["total_observed_time"].to_numpy(dtype=float)

        for row_idx, (state, age_seconds) in enumerate(zip(current_states, current_ages)):
            state_idx = self.state_to_idx_.get(int(state))
            if state_idx is None:
                probs[row_idx] = self.fallback_
                continue

            p = np.zeros(len(self.states_), dtype=float)
            p[state_idx] = 1.0
            t = max(float(age_seconds), 0.0)
            remaining = float(horizon_seconds)

            while remaining > 1e-9:
                bin_idx = min(int(self._to_bin([t])[0]), self.n_bins_ - 1)
                next_boundary = (
                    self.bin_edges_seconds[bin_idx]
                    if bin_idx < len(self.bin_edges_seconds)
                    else np.inf
                )
                if np.isfinite(next_boundary):
                    segment = min(remaining, max(next_boundary - t, 0.0))
                    if segment <= 1e-9:
                        # Move exactly-on-boundary predictions into the next bin.
                        t = next_boundary + 1e-6
                        continue
                else:
                    segment = remaining

                p = self._propagate_row(p, bin_idx, segment)
                t += segment
                remaining -= segment

            probs[row_idx] = p[success_idx]

        return np.clip(probs, 0.0, 1.0)

    def bin_summary(self) -> pd.DataFrame:
        self._check_fit()
        rows = []
        for bin_idx in range(self.n_bins_):
            lower, upper = self._bin_bounds(bin_idx)
            if np.isinf(upper):
                window = f"{lower / 86400:.0f}d+"
            elif lower == 0:
                window = f"0-{upper / 86400:.0f}d"
            else:
                window = f"{lower / 86400:.0f}-{upper / 86400:.0f}d"

            counts = self.counts_by_bin_[bin_idx]
            exposure = self.exposure_by_bin_[bin_idx]
            fallback = self.row_fallback_by_bin_[bin_idx]
            rows.append({
                "bin": bin_idx,
                "window": window,
                "n_transitions": int(counts.to_numpy().sum()),
                "exposure_days": float(exposure.sum() / 86400),
                "n_fallback_rows": int(fallback.sum()),
                "n_states": len(self.states_),
            })
        return pd.DataFrame(rows)

    def _interval_bin_overlaps(self, start: float, end: float):
        if end <= start:
            return
        for bin_idx in range(self.n_bins_):
            lower, upper = self._bin_bounds(bin_idx)
            overlap_start = max(start, lower)
            overlap_end = min(end, upper)
            if overlap_end > overlap_start:
                yield bin_idx, overlap_end - overlap_start
            if end <= upper:
                break

    def _bin_bounds(self, bin_idx: int) -> tuple[float, float]:
        lower = 0.0 if bin_idx == 0 else self.bin_edges_seconds[bin_idx - 1]
        upper = (
            self.bin_edges_seconds[bin_idx]
            if bin_idx < len(self.bin_edges_seconds)
            else np.inf
        )
        return lower, upper

    def _estimate_q_from_counts(self, counts: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        n_states = len(self.states_)
        q = np.zeros((n_states, n_states), dtype=float)
        for state, state_idx in self.state_to_idx_.items():
            if state in {self.success_state, self.failure_state}:
                continue
            if exposure[state_idx] <= 0:
                continue
            row_q = counts[state_idx] / max(float(exposure[state_idx]), self.min_time)
            row_q[state_idx] = 0.0
            row_q[state_idx] = -row_q.sum()
            q[state_idx] = row_q
        return q

    def _fit_eigendecompositions(self) -> None:
        self._eigendecomp_by_bin_ = {}
        for bin_idx, q_df in self.Q_by_bin_.items():
            q = q_df.to_numpy(dtype=float)
            try:
                eigvals, eigvecs = np.linalg.eig(q)
                inv_eigvecs = np.linalg.inv(eigvecs)
                self._eigendecomp_by_bin_[bin_idx] = (eigvals, eigvecs, inv_eigvecs)
            except np.linalg.LinAlgError:
                self._eigendecomp_by_bin_[bin_idx] = None

    def _propagate_row(self, p: np.ndarray, bin_idx: int, segment_seconds: float) -> np.ndarray:
        eig = self._eigendecomp_by_bin_.get(bin_idx)
        if eig is None:
            try:
                from scipy.linalg import expm
            except ImportError as exc:
                raise ImportError("scipy is required for temporal CTMC predictions") from exc
            out = p @ expm(self.Q_by_bin_[bin_idx].to_numpy(dtype=float) * segment_seconds)
        else:
            eigvals, eigvecs, inv_eigvecs = eig
            out = (p @ eigvecs * np.exp(eigvals * segment_seconds)) @ inv_eigvecs
        return np.clip(np.real_if_close(out, tol=1000).real, 0.0, 1.0)

    def _check_fit(self) -> None:
        if not self.Q_by_bin_ or self.global_Q_ is None:
            raise RuntimeError("Call fit() before using PiecewiseTimeVaryingTimeoutCTMC.")


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
