"""
Build the three parquet files required by CTMCData and tabular models.

Input:  data/dat_train1_clean.csv   (deduplicated event log, already exists)
Output: data/journeys_flattened.parquet
        data/journeys_labeled.parquet
        data/journey_training_optionA.parquet

Each output uses ed_id integers (not event_name strings) so that:
  - CTMCData.events() / transition_table() work unmodified
  - tabular_submission.build_open_journey_features() column names align
  - SUCCESS_STATE=28 (order_shipped) resolves correctly

Run:
    python -m src.data_engineering.build_training_parquets
    python -m src.data_engineering.build_training_parquets --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ed_id=28 is order_shipped — verified from dat_train1_clean.csv
SUCCESS_ED_ID = 28
TOP_N_ACTIONS = 20
HORIZON_DAYS = 60


def build_flattened(clean_csv: Path, output: Path) -> pl.DataFrame:
    """Flatten event log to one row per journey.

    Journey structs contain {event_timestamp, ed_id} so that
    CTMCData.events() can unnest them directly.
    """
    print(f"[1/3] Building {output.name} ...")
    q = (
        pl.scan_csv(clean_csv)
        .with_columns(pl.col("event_timestamp").str.to_datetime(time_zone="UTC"))
        .sort(["id", "event_timestamp", "ed_id"])
    )
    df_flat = q.group_by("id").agg([
        pl.struct(["event_timestamp", "ed_id"]).alias("journey"),
        pl.len().alias("num_actions"),
        pl.col("ed_id").n_unique().alias("num_unique_actions"),
        pl.col("event_timestamp").min().alias("start_time"),
        pl.col("event_timestamp").max().alias("end_time"),
        (pl.col("event_timestamp").max() - pl.col("event_timestamp").min())
            .dt.total_seconds().alias("duration_seconds"),
        pl.col("ed_id").first().alias("first_action_id"),
        pl.col("ed_id").last().alias("last_action_id"),
        pl.col("event_name").first().alias("first_action"),
        pl.col("event_name").last().alias("last_action"),
    ])
    df_flat.sink_parquet(output)
    result = pl.read_parquet(output)
    n = result.shape[0]
    print(f"      {n:,} journeys written")
    return result


def build_labeled(flattened: pl.DataFrame, clean_csv: Path, output: Path) -> pl.DataFrame:
    """Assign journey_status: successful / incomplete / other."""
    print(f"[2/3] Building {output.name} ...")
    dataset_end_time = (
        pl.scan_csv(clean_csv)
        .with_columns(pl.col("event_timestamp").str.to_datetime(time_zone="UTC"))
        .select(pl.col("event_timestamp").max())
        .collect()
        .item()
    )
    labeled = flattened.with_columns([
        (pl.col("last_action_id") == SUCCESS_ED_ID).alias("is_successful"),
        (
            (pl.lit(dataset_end_time) - pl.col("end_time")).dt.total_days() >= HORIZON_DAYS
        ).alias("inactive_60_days"),
    ]).with_columns(
        pl.when(pl.col("is_successful"))
          .then(pl.lit("successful"))
          .when((~pl.col("is_successful")) & pl.col("inactive_60_days"))
          .then(pl.lit("incomplete"))
          .otherwise(pl.lit("other"))
          .alias("journey_status")
    )
    labeled.write_parquet(output)
    for row in labeled["journey_status"].value_counts().sort("journey_status").iter_rows(named=True):
        print(f"      {row['journey_status']}: {row['count']:,}")
    return labeled


def _top_ed_ids(labeled: pl.DataFrame) -> list[int]:
    model_base = labeled.filter(pl.col("journey_status").is_in(["successful", "incomplete"]))
    top = (
        model_base
        .explode("journey")
        .unnest("journey")
        .group_by("ed_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(TOP_N_ACTIONS)
        .get_column("ed_id")
        .to_list()
    )
    return [int(x) for x in top]


def _make_snapshot(row: dict, top_ed_ids: list[int]) -> dict:
    journey = row["journey"]
    n = len(journey)
    k = 1 if n <= 1 else random.randint(1, n - 1)

    prefix = journey[:k]
    timestamps = [s["event_timestamp"] for s in prefix]
    actions = [int(s["ed_id"]) for s in prefix]

    prefix_duration = int((timestamps[-1] - timestamps[0]).total_seconds()) if k > 1 else 0
    avg_gap = prefix_duration / (k - 1) if k > 1 else 0.0
    time_since_prev = int((timestamps[-1] - timestamps[-2]).total_seconds()) if k > 1 else 0

    action_counts = Counter(actions)
    out: dict = {
        "id": row["id"],
        "label": 1 if row["journey_status"] == "successful" else 0,
        "final_outcome": row["journey_status"],
        "full_num_actions": n,
        "snapshot_num_actions": k,
        "snapshot_frac_of_journey": k / n,
        "first_action_so_far": actions[0],
        "current_last_action": actions[-1],
        "num_unique_actions_so_far": len(set(actions)),
        "prefix_duration_seconds": prefix_duration,
        "avg_gap_seconds": avg_gap,
        "time_since_prev_action_seconds": time_since_prev,
        "prefix_actions": actions,
    }
    for ed_id in top_ed_ids:
        out[f"action_count_{ed_id}"] = action_counts.get(ed_id, 0)
    return out


def build_training(labeled: pl.DataFrame, output: Path, random_state: int = 42) -> pl.DataFrame:
    """One random prefix snapshot per labeled journey, using ed_id integers."""
    print(f"[3/3] Building {output.name} ...")
    random.seed(random_state)

    model_base = labeled.filter(pl.col("journey_status").is_in(["successful", "incomplete"]))
    print(f"      {model_base.shape[0]:,} labeled journeys")

    top_ed_ids = _top_ed_ids(labeled)
    print(f"      top ed_ids: {top_ed_ids}")

    rows = [_make_snapshot(row, top_ed_ids) for row in model_base.iter_rows(named=True)]
    training_df = pl.DataFrame(rows)
    training_df.write_parquet(output)
    for row in training_df.group_by("label").agg(pl.len().alias("n")).sort("label").iter_rows(named=True):
        print(f"      label={row['label']}: {row['n']:,}")
    return training_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build training parquets from cleaned event log.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing dat_train1_clean.csv and where parquets will be written.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    opts = parser.parse_args(argv)

    data_dir: Path = opts.data_dir
    clean_csv = data_dir / "dat_train1_clean.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(
            f"Missing input file: {clean_csv}\n"
            "Place dat_train1_clean.csv in the data/ directory and re-run."
        )

    flattened = build_flattened(clean_csv, data_dir / "journeys_flattened.parquet")
    labeled = build_labeled(flattened, clean_csv, data_dir / "journeys_labeled.parquet")
    build_training(labeled, data_dir / "journey_training_optionA.parquet", opts.random_state)
    print("\nDone. Run ctmc_submission.py to generate Kaggle submissions.")


if __name__ == "__main__":
    main()
