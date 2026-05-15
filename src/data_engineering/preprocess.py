"""Reusable preprocessing primitives for the data engineering workflow.

Full pipeline (raw CSV → training snapshots):
    1. deduplicate_events   — drop duplicate (id, event_name, timestamp) rows
    2. parse_timestamps     — parse event_timestamp strings to UTC datetime
    3. flatten_journeys     — one row per journey with nested event structs
    4. label_journeys       — classify as successful / incomplete / other
    5. truncate_journeys    — multi-sample time-based snapshots per journey
    6. add_action_features  — count columns for top-N ed_ids
    7. add_derived_features — entropy, max repeat, milestone flags, proportions
    8. build_pipeline       — end-to-end convenience wrapper

Importable without side effects. Run the pipeline via:
    python -m src.data_engineering.build_training_parquets
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

SUCCESS_ED_ID: int = 28
HORIZON_DAYS: int = 60
TOP_N_ACTIONS: int = 20
TOP_N_EVENTS: int = 50
MAX_SAMPLES_PER_JOURNEY: int = 60
SUCCESS_EVENT: str = "order_shipped"

MILESTONE_ACTIONS: dict[str, int] = {
    "has_add_to_cart":        11,
    "has_begin_checkout":      6,
    "has_application_submit":  3,
    "has_place_downpayment":   7,
    "has_place_order":         8,
    "has_account_activation": 29,
}


# ---------------------------------------------------------------------------
# Step 1 — Deduplication
# ---------------------------------------------------------------------------

def deduplicate_events(q: pl.LazyFrame) -> pl.LazyFrame:
    """Drop duplicate (id, event_name, event_timestamp) rows."""
    return q.unique(subset=["id", "event_name", "event_timestamp"])


# ---------------------------------------------------------------------------
# Step 2 — Timestamp parsing
# ---------------------------------------------------------------------------

def parse_timestamps(q: pl.LazyFrame) -> pl.LazyFrame:
    """Parse event_timestamp strings to UTC-aware datetime."""
    return q.with_columns(
        pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
    )


# ---------------------------------------------------------------------------
# Step 3 — Journey flattening
# ---------------------------------------------------------------------------

def flatten_journeys(q: pl.LazyFrame) -> pl.DataFrame:
    """Aggregate events into one row per journey.

    Each journey struct contains {event_timestamp, event_name, ed_id} so that
    CTMCData.events() can unnest it directly while newer feature builders can
    use event names.

    Returns columns: id, journey, num_actions, num_unique_actions,
    start_time, end_time, duration_seconds, first_action, last_action,
    has_order_shipped.
    """
    return (
        q.sort(["id", "event_timestamp", "ed_id"])
        .group_by("id")
        .agg([
            pl.struct(["event_timestamp", "event_name", "ed_id"]).alias("journey"),
            pl.len().alias("num_actions"),
            pl.col("ed_id").n_unique().alias("num_unique_actions"),
            pl.col("event_timestamp").min().alias("start_time"),
            pl.col("event_timestamp").max().alias("end_time"),
            (pl.col("event_timestamp").max() - pl.col("event_timestamp").min())
                .dt.total_seconds().alias("duration_seconds"),
            pl.col("ed_id").first().alias("first_action"),
            pl.col("ed_id").last().alias("last_action"),
            (pl.col("ed_id") == SUCCESS_ED_ID).any().alias("has_order_shipped"),
        ])
        .collect()
    )


# ---------------------------------------------------------------------------
# Step 4 — Journey labeling
# ---------------------------------------------------------------------------

def label_journeys(
    journeys: pl.DataFrame,
    dataset_end_time,
    success_ed_id: int = SUCCESS_ED_ID,
    horizon_days: int = HORIZON_DAYS,
) -> pl.DataFrame:
    """Assign journey_status: 'successful' | 'incomplete' | 'other'.

    - successful:  last action is success_ed_id (order_shipped, ed_id=28)
    - incomplete:  not successful AND last action >= horizon_days before dataset end
    - other:       recently-active journeys with no clear final outcome yet
    """
    return (
        journeys
        .with_columns([
            (pl.col("last_action") == success_ed_id).alias("is_successful"),
            (
                (pl.lit(dataset_end_time) - pl.col("end_time")).dt.total_days()
                >= horizon_days
            ).alias("inactive_60_days"),
        ])
        .with_columns(
            pl.when(pl.col("is_successful"))
              .then(pl.lit("successful"))
              .when((~pl.col("is_successful")) & pl.col("inactive_60_days"))
              .then(pl.lit("incomplete"))
              .otherwise(pl.lit("other"))
              .alias("journey_status")
        )
    )


# ---------------------------------------------------------------------------
# Step 5 — Time-based truncation (from truncation.ipynb)
# ---------------------------------------------------------------------------

def _truncate_one_journey(
    row: dict,
    rng: random.Random,
    max_samples: int = MAX_SAMPLES_PER_JOURNEY,
) -> list[dict]:
    """Generate random time-cutoff prefix snapshots for one journey.

    Number of samples scales with journey duration (1 per day of duration,
    capped at max_samples). Each snapshot is a prefix of the journey up to a
    uniformly random cutoff time within [start_time, end_time].
    """
    journey = row["journey"]
    if not journey:
        return []

    start_time = row["start_time"]
    end_time = row["end_time"]
    if start_time is None or end_time is None:
        return []

    start_ts = start_time.timestamp()
    end_ts = end_time.timestamp()
    duration_days = (end_ts - start_ts) / 86400

    n_samples = min(max(1, math.ceil(duration_days)), max_samples)

    event_times = [e["event_timestamp"].timestamp() for e in journey]
    event_ids = [int(e["ed_id"]) for e in journey]

    results: list[dict] = []
    for sample_num in range(1, n_samples + 1):
        cutoff_ts = rng.uniform(start_ts, end_ts)

        keep_idx = sum(1 for t in event_times if t <= cutoff_ts)
        if keep_idx == 0:
            continue

        prefix_ids = event_ids[:keep_idx]
        first_time = journey[0]["event_timestamp"]
        last_time = journey[keep_idx - 1]["event_timestamp"]

        prefix_duration = int((last_time - first_time).total_seconds()) if keep_idx > 1 else 0
        avg_gap = prefix_duration / (keep_idx - 1) if keep_idx > 1 else 0.0
        time_since_prev = (
            int(
                (journey[keep_idx - 1]["event_timestamp"]
                 - journey[keep_idx - 2]["event_timestamp"]).total_seconds()
            )
            if keep_idx > 1 else 0
        )

        results.append({
            "id": row["id"],
            "sample_num": sample_num,
            "label": 1 if row["journey_status"] == "successful" else 0,
            "journey_status": row["journey_status"],
            "full_num_actions": len(journey),
            "snapshot_num_actions": keep_idx,
            "snapshot_frac_of_journey": keep_idx / len(journey),
            "first_action_so_far": prefix_ids[0],
            "current_last_action": prefix_ids[-1],
            "num_unique_actions_so_far": len(set(prefix_ids)),
            "prefix_duration_seconds": prefix_duration,
            "avg_gap_seconds": avg_gap,
            "time_since_prev_action_seconds": time_since_prev,
            "prefix_actions": prefix_ids,
        })

    return results


def truncate_journeys(
    labeled: pl.DataFrame,
    seed: int = 42,
    max_samples: int = MAX_SAMPLES_PER_JOURNEY,
) -> pl.DataFrame:
    """Build training snapshots via random time-based truncation.

    Only 'successful' and 'incomplete' journeys are included.
    Each journey produces between 1 and max_samples rows depending on its
    duration in days.
    """
    model_base = labeled.filter(
        pl.col("journey_status").is_in(["successful", "incomplete"])
    )
    rng = random.Random(seed)
    rows: list[dict] = []
    for row in model_base.iter_rows(named=True):
        rows.extend(_truncate_one_journey(row, rng, max_samples))
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 — Action count features
# ---------------------------------------------------------------------------

def add_action_features(
    training_df: pl.DataFrame,
    top_ed_ids: Sequence[int] | None = None,
    top_n: int = TOP_N_ACTIONS,
) -> pl.DataFrame:
    """Add one count column per top-N most frequent ed_id in training prefixes.

    If top_ed_ids is provided, use those ids; otherwise derive from training_df.
    Returns the DataFrame with new columns named action_count_<ed_id>.
    """
    if top_ed_ids is None:
        top_ed_ids = (
            training_df.explode("prefix_actions")
            .group_by("prefix_actions")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(top_n)
            .get_column("prefix_actions")
            .to_list()
        )

    for ed_id in top_ed_ids:
        training_df = training_df.with_columns(
            pl.col("prefix_actions")
            .list.eval(pl.element() == int(ed_id))
            .list.sum()
            .alias(f"action_count_{ed_id}")
        )

    return training_df


# ---------------------------------------------------------------------------
# Step 7 — Derived features
# ---------------------------------------------------------------------------

def add_derived_features(
    training_df: pl.DataFrame,
    top_ed_ids: list[int] | None = None,
) -> pl.DataFrame:
    """Compute derived features from prefix_actions and existing count columns.

    Features added:
    - action_entropy         : Shannon entropy of action distribution in the prefix
    - max_action_repeat      : max count of any single action in the prefix
    - has_<milestone>        : 0/1 flag for each key funnel step (see MILESTONE_ACTIONS)
    - action_prop_<ed_id>    : action_count_<ed_id> / snapshot_num_actions (normalised)

    If top_ed_ids is None, action_count_* columns already present in
    training_df are used to derive proportion features.

    Note: _entropy and _max_repeat use map_elements (row-wise Python) rather
    than a pure polars explode approach because each row is an independent
    snapshot — (id, sample_num) pairs are not unique journeys — making the
    explode+group_by approach more complex with no correctness benefit at
    capstone scale.
    """
    def _entropy(actions) -> float:
        actions = list(actions) if actions is not None else []
        if not actions:
            return 0.0
        counts = Counter(actions)
        total = len(actions)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def _max_repeat(actions) -> int:
        actions = list(actions) if actions is not None else []
        if not actions:
            return 0
        return max(Counter(actions).values())

    training_df = training_df.with_columns(
        pl.col("prefix_actions")
        .map_elements(_entropy, return_dtype=pl.Float64)
        .alias("action_entropy")
    )
    training_df = training_df.with_columns(
        pl.col("prefix_actions")
        .map_elements(_max_repeat, return_dtype=pl.Int64)
        .alias("max_action_repeat")
    )

    # milestone flags — native list.contains, polars executes all 6 in one plan node
    training_df = training_df.with_columns([
        pl.col("prefix_actions")
        .list.contains(ed_id)
        .cast(pl.Int8)
        .alias(col_name)
        for col_name, ed_id in MILESTONE_ACTIONS.items()
    ])

    # proportion features: action_count_X / snapshot_num_actions
    if top_ed_ids is None:
        top_ed_ids = [
            int(col.removeprefix("action_count_"))
            for col in training_df.columns
            if col.startswith("action_count_")
        ]

    prop_exprs = [
        (pl.col(f"action_count_{eid}").cast(pl.Float64) / pl.col("snapshot_num_actions"))
        .alias(f"action_prop_{eid}")
        for eid in top_ed_ids
    ]
    training_df = training_df.with_columns(prop_exprs)

    # fill NaN from 0/0 when snapshot_num_actions == 0 (edge case, but safe)
    prop_col_names = [f"action_prop_{eid}" for eid in top_ed_ids]
    training_df = training_df.with_columns(
        [pl.col(c).fill_nan(0.0) for c in prop_col_names]
    )

    return training_df


# ---------------------------------------------------------------------------
# Step 7b — Realistic time truncation + event-name features
# ---------------------------------------------------------------------------

def clean_col_name(x) -> str:
    return (
        str(x)
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def realistic_top_events(
    labeled: pl.DataFrame,
    *,
    top_n_events: int = TOP_N_EVENTS,
    success_event: str = SUCCESS_EVENT,
) -> list[str]:
    """Return the top event names used for realistic truncation count columns."""
    model_base = labeled.filter(
        pl.col("journey_status").is_in(["successful", "incomplete"])
    )
    if model_base.is_empty():
        return []

    events = (
        model_base
        .explode("journey")
        .unnest("journey")
    )
    event_col = "event_name" if "event_name" in events.columns else "ed_id"
    success_values = [success_event]
    if event_col == "ed_id":
        success_values.append(str(SUCCESS_ED_ID))

    top = (
        events
        .with_columns(pl.col(event_col).cast(pl.Utf8).alias("_event_name_for_counts"))
        .filter(pl.col("_event_name_for_counts").is_not_null())
        .filter(~pl.col("_event_name_for_counts").is_in(success_values))
        .group_by("_event_name_for_counts")
        .len()
        .sort("len", descending=True)
        .head(top_n_events)
        .get_column("_event_name_for_counts")
        .to_list()
    )
    return [str(x) for x in top]


def add_realistic_truncation_features(
    labeled: pl.DataFrame,
    *,
    seed: int = 42,
    max_samples_per_journey: int = MAX_SAMPLES_PER_JOURNEY,
    top_n_events: int = TOP_N_EVENTS,
    top_events: Sequence[str] | None = None,
    success_event: str = SUCCESS_EVENT,
) -> pl.DataFrame:
    """Build multi-snapshot prefix features from the pulled notebook.

    Successful journeys are truncated no later than 70% of the time to
    completion to reduce success leakage. Incomplete journeys are sampled over
    their observed active window. The output is one row per snapshot and keeps
    `snapshot_time`, `last_ed_id`, and rich prefix features for CTMC clustering.
    """
    rng = random.Random(seed)
    model_base = labeled.filter(
        pl.col("journey_status").is_in(["successful", "incomplete"])
    )

    if top_events is None:
        top_events = realistic_top_events(
            model_base,
            top_n_events=top_n_events,
            success_event=success_event,
        )
    else:
        top_events = [str(x) for x in top_events]

    rows: list[dict] = []
    for row in model_base.iter_rows(named=True):
        journey = row["journey"]
        if not journey:
            continue

        start_time = row["start_time"]
        end_time = row["end_time"]
        if start_time is None or end_time is None:
            continue

        status = row["journey_status"]
        label = 1 if status == "successful" else 0

        event_times = [e["event_timestamp"] for e in journey]
        event_ts = [t.timestamp() for t in event_times]
        event_ids = [int(e["ed_id"]) for e in journey]
        event_names = [str(e.get("event_name", e["ed_id"])) for e in journey]

        completion_times = [
            t for t, name in zip(event_times, event_names)
            if name == success_event
        ]
        effective_end = completion_times[0] if completion_times else end_time

        start_ts = start_time.timestamp()
        end_ts = effective_end.timestamp()
        if end_ts <= start_ts:
            continue

        duration_days = (end_ts - start_ts) / 86400
        n_samples = min(max(1, math.ceil(duration_days)), max_samples_per_journey)

        for sample_num in range(1, n_samples + 1):
            if status == "successful":
                max_allowed_ts = start_ts + 0.70 * (end_ts - start_ts)
                cutoff_ts = rng.uniform(start_ts, max_allowed_ts)
            else:
                cutoff_ts = rng.uniform(start_ts, end_ts)

            keep_idx = sum(1 for t in event_ts if t <= cutoff_ts)
            if keep_idx == 0:
                continue

            prefix_times = event_times[:keep_idx]
            prefix_ids = event_ids[:keep_idx]
            prefix_names = event_names[:keep_idx]

            non_leak = [
                i for i, name in enumerate(prefix_names)
                if name != success_event
            ]
            if not non_leak:
                continue

            prefix_times = [prefix_times[i] for i in non_leak]
            prefix_ids = [prefix_ids[i] for i in non_leak]
            prefix_names = [prefix_names[i] for i in non_leak]

            first_time = prefix_times[0]
            last_time = prefix_times[-1]
            snapshot_time = last_time
            snapshot_age_days = (snapshot_time - start_time).total_seconds() / 86400
            active_span_days = (last_time - first_time).total_seconds() / 86400

            gaps = [
                (prefix_times[i] - prefix_times[i - 1]).total_seconds() / 86400
                for i in range(1, len(prefix_times))
            ]
            counts = Counter(prefix_names)

            out = {
                "id": row["id"],
                "sample_num": sample_num,
                "snapshot_id": f"{row['id']}::{sample_num}",
                "journey_status": status,
                "source_table": "complete" if status == "successful" else "incomplete",
                "label": label,
                "snapshot_time": snapshot_time,
                "snapshot_age_days": snapshot_age_days,
                "num_events": len(prefix_names),
                "num_unique_event_names": len(set(prefix_names)),
                "num_unique_ed_ids": len(set(prefix_ids)),
                "first_event_time": first_time,
                "last_event_time": last_time,
                "first_event_name": prefix_names[0],
                "last_event_name": prefix_names[-1],
                "first_ed_id": prefix_ids[0],
                "last_ed_id": prefix_ids[-1],
                "current_state": prefix_ids[-1],
                "active_span_days": active_span_days,
                "days_since_last_event": 0.0,
                "events_per_day_since_start": len(prefix_names) / (snapshot_age_days + 1),
                "events_per_active_day": len(prefix_names) / (active_span_days + 1),
                "avg_gap_days": float(np.mean(gaps)) if gaps else 0.0,
                "max_gap_days": float(np.max(gaps)) if gaps else 0.0,
                "std_gap_days": float(np.std(gaps)) if gaps else 0.0,
                "last1_event": prefix_names[-1],
                "last2_event": prefix_names[-2] if len(prefix_names) >= 2 else "missing",
                "last3_event": prefix_names[-3] if len(prefix_names) >= 3 else "missing",
                "last4_event": prefix_names[-4] if len(prefix_names) >= 4 else "missing",
                "last5_event": prefix_names[-5] if len(prefix_names) >= 5 else "missing",
                "prefix_actions": prefix_ids,
            }

            for window in [1, 3, 7, 14, 30]:
                out[f"events_last_{window}d"] = sum(
                    (snapshot_time - t).total_seconds() / 86400 <= window
                    for t in prefix_times
                )

            out["recent_1d_to_30d"] = out["events_last_1d"] / (out["events_last_30d"] + 1)
            out["recent_3d_to_30d"] = out["events_last_3d"] / (out["events_last_30d"] + 1)
            out["recent_7d_to_30d"] = out["events_last_7d"] / (out["events_last_30d"] + 1)

            for event in top_events:
                out[f"cnt_{clean_col_name(event)}"] = counts.get(event, 0)

            rows.append(out)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).fill_null(0)


def build_open_journey_realistic_features(
    events: pl.DataFrame,
    *,
    success_event: str = SUCCESS_EVENT,
) -> pl.DataFrame:
    """Build one realistic feature row per open/test journey.

    Open journeys are already observed prefixes, so this does not sample or
    truncate. It mirrors add_realistic_truncation_features on the full observed
    prefix and omits training-only label/suffix fields.
    """
    if events.is_empty():
        return pl.DataFrame()

    required = {"id", "event_timestamp", "event_name", "ed_id"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Open journey events missing required columns: {sorted(missing)}")

    events = (
        events
        .with_columns([
            pl.col("event_timestamp").str.to_datetime(time_zone="UTC")
            if events.schema["event_timestamp"] == pl.Utf8
            else pl.col("event_timestamp"),
            pl.col("event_name").cast(pl.Utf8),
            pl.col("ed_id").cast(pl.Int64),
        ])
        .sort(["id", "event_timestamp", "ed_id"])
    )

    rows: list[dict] = []
    for row in (
        events.group_by("id")
        .agg(pl.struct(["event_timestamp", "event_name", "ed_id"]).alias("journey"))
        .iter_rows(named=True)
    ):
        journey = row["journey"]
        if not journey:
            continue

        prefix_times = [e["event_timestamp"] for e in journey if e["event_name"] != success_event]
        prefix_ids = [int(e["ed_id"]) for e in journey if e["event_name"] != success_event]
        prefix_names = [str(e["event_name"]) for e in journey if e["event_name"] != success_event]
        if not prefix_names:
            continue

        first_time = prefix_times[0]
        last_time = prefix_times[-1]
        snapshot_time = last_time
        snapshot_age_days = (snapshot_time - first_time).total_seconds() / 86400
        active_span_days = snapshot_age_days
        gaps = [
            (prefix_times[i] - prefix_times[i - 1]).total_seconds() / 86400
            for i in range(1, len(prefix_times))
        ]
        counts = Counter(prefix_names)

        out = {
            "id": row["id"],
            "sample_num": 1,
            "snapshot_id": f"{row['id']}::open",
            "journey_status": "open",
            "source_table": "open",
            "snapshot_time": snapshot_time,
            "snapshot_age_days": snapshot_age_days,
            "num_events": len(prefix_names),
            "num_unique_event_names": len(set(prefix_names)),
            "num_unique_ed_ids": len(set(prefix_ids)),
            "first_event_time": first_time,
            "last_event_time": last_time,
            "first_event_name": prefix_names[0],
            "last_event_name": prefix_names[-1],
            "first_ed_id": prefix_ids[0],
            "last_ed_id": prefix_ids[-1],
            "current_state": prefix_ids[-1],
            "active_span_days": active_span_days,
            "days_since_last_event": 0.0,
            "events_per_day_since_start": len(prefix_names) / (snapshot_age_days + 1),
            "events_per_active_day": len(prefix_names) / (active_span_days + 1),
            "avg_gap_days": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap_days": float(np.max(gaps)) if gaps else 0.0,
            "std_gap_days": float(np.std(gaps)) if gaps else 0.0,
            "last1_event": prefix_names[-1],
            "last2_event": prefix_names[-2] if len(prefix_names) >= 2 else "missing",
            "last3_event": prefix_names[-3] if len(prefix_names) >= 3 else "missing",
            "last4_event": prefix_names[-4] if len(prefix_names) >= 4 else "missing",
            "last5_event": prefix_names[-5] if len(prefix_names) >= 5 else "missing",
            "prefix_actions": prefix_ids,
        }

        for window in [1, 3, 7, 14, 30]:
            out[f"events_last_{window}d"] = sum(
                (snapshot_time - t).total_seconds() / 86400 <= window
                for t in prefix_times
            )

        out["recent_1d_to_30d"] = out["events_last_1d"] / (out["events_last_30d"] + 1)
        out["recent_3d_to_30d"] = out["events_last_3d"] / (out["events_last_30d"] + 1)
        out["recent_7d_to_30d"] = out["events_last_7d"] / (out["events_last_30d"] + 1)

        for event, count in counts.items():
            out[f"cnt_{clean_col_name(event)}"] = count

        rows.append(out)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).fill_null(0)


# ---------------------------------------------------------------------------
# Step 8 — End-to-end pipeline
# ---------------------------------------------------------------------------

def build_pipeline(
    raw_csv: Path,
    *,
    seed: int = 42,
    max_samples: int = MAX_SAMPLES_PER_JOURNEY,
    success_ed_id: int = SUCCESS_ED_ID,
    horizon_days: int = HORIZON_DAYS,
    top_n: int = TOP_N_ACTIONS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run the full preprocessing pipeline from raw CSV to training snapshots.

    Steps: dedup → parse timestamps → flatten → label → truncate → action features.

    Returns:
        (journeys_flattened, journeys_labeled, training_snapshots)
    """
    q = pl.scan_csv(raw_csv)
    q = deduplicate_events(q)
    q = parse_timestamps(q)

    journeys_flat = flatten_journeys(q)

    dataset_end_time = (
        pl.scan_csv(raw_csv)
        .with_columns(pl.col("event_timestamp").str.to_datetime(time_zone="UTC"))
        .select(pl.col("event_timestamp").max())
        .collect()
        .item()
    )

    journeys_lab = label_journeys(
        journeys_flat, dataset_end_time, success_ed_id, horizon_days
    )

    snapshots = truncate_journeys(journeys_lab, seed=seed, max_samples=max_samples)
    snapshots = add_action_features(snapshots, top_n=top_n)
    snapshots = add_derived_features(snapshots)

    return journeys_flat, journeys_lab, snapshots
