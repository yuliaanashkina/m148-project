"""
Build the three parquet files required by CTMCData and tabular models.

Input:  data/dat_train1_clean.csv   (deduplicated event log, already exists)
Output: data/journeys_flattened.parquet
        data/journeys_labeled.parquet
        data/journey_training_optionA.parquet
        data/journey_training_realistic_truncation.parquet

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
import time
from collections import Counter
from pathlib import Path

import polars as pl

try:
    from .preprocess import (
        add_realistic_truncation_features,
        build_open_journey_realistic_features,
        realistic_top_events,
    )
except ImportError:
    from preprocess import (
        add_realistic_truncation_features,
        build_open_journey_realistic_features,
        realistic_top_events,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ed_id=28 is order_shipped — verified from dat_train1_clean.csv
SUCCESS_ED_ID = 28
TOP_N_ACTIONS = 20
HORIZON_DAYS = 60


def build_flattened(clean_csv: Path, output: Path) -> pl.DataFrame:
    """Flatten event log to one row per journey.

    Journey structs contain {event_timestamp, event_name, ed_id} so that
    CTMCData.events() can unnest them directly and feature engineering can use
    event names.
    """
    print(f"[1/3] Building {output.name} ...")
    q = (
        pl.scan_csv(clean_csv)
        .with_columns(pl.col("event_timestamp").str.to_datetime(time_zone="UTC"))
        .sort(["id", "event_timestamp", "ed_id"])
    )
    df_flat = q.group_by("id").agg([
        pl.struct(["event_timestamp", "event_name", "ed_id"]).alias("journey"),
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


def _journey_struct_has_event_name(df: pl.DataFrame) -> bool:
    """Whether nested journey structs include the newer event_name field."""
    journey_dtype = df.schema.get("journey")
    if journey_dtype is None or getattr(journey_dtype, "inner", None) is None:
        return False
    fields = getattr(journey_dtype.inner, "fields", [])
    return any(field.name == "event_name" for field in fields)


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


def build_realistic_training(
    labeled: pl.DataFrame,
    output: Path,
    random_state: int = 42,
    max_samples_per_journey: int = 60,
    top_n_events: int = 50,
    max_journeys: int | None = None,
    batch_size: int = 5_000,
    resume: bool = False,
) -> pl.DataFrame:
    """Multi-snapshot feature table from truncation&features.ipynb logic.

    The full table can be large, so it is generated in journey batches and
    written as parquet shards before the final concatenated parquet is built.
    """
    print(f"[4/4] Building {output.name} ...", flush=True)
    started = time.perf_counter()
    parts_dir = output.parent / "feature_engineered" / "realistic_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    if max_journeys is not None:
        labeled = labeled.filter(pl.col("journey_status").is_in(["successful", "incomplete"]))
        labeled = labeled.sample(n=min(max_journeys, labeled.height), shuffle=True, seed=random_state)
    else:
        labeled = labeled.filter(pl.col("journey_status").is_in(["successful", "incomplete"]))

    if labeled.is_empty():
        empty = pl.DataFrame()
        empty.write_parquet(output)
        print("      0 labeled journeys; wrote empty parquet", flush=True)
        return empty

    if batch_size <= 0:
        raise ValueError("--realistic-batch-size must be positive")

    if not resume:
        for old_part in parts_dir.glob("part_*.parquet"):
            old_part.unlink()

    top_events = realistic_top_events(
        labeled,
        top_n_events=top_n_events,
    )
    print(f"      {labeled.height:,} labeled journeys", flush=True)
    print(f"      top event count columns: {len(top_events):,}", flush=True)

    n_batches = (labeled.height + batch_size - 1) // batch_size
    total_snapshots = 0
    written_parts: list[Path] = []
    for batch_idx, start in enumerate(range(0, labeled.height, batch_size), start=1):
        part_path = parts_dir / f"part_{batch_idx:05d}.parquet"
        if resume and part_path.exists():
            part_rows = pl.scan_parquet(part_path).select(pl.len()).collect().item()
            total_snapshots += int(part_rows)
            written_parts.append(part_path)
            elapsed = time.perf_counter() - started
            print(
                f"      batch {batch_idx:,}/{n_batches:,} already exists | "
                f"journeys {min(start + batch_size, labeled.height):,}/{labeled.height:,} | "
                f"snapshots {total_snapshots:,} | elapsed {elapsed/60:.1f} min",
                flush=True,
            )
            continue

        batch = labeled.slice(start, batch_size)
        batch_df = add_realistic_truncation_features(
            batch,
            seed=random_state + batch_idx - 1,
            max_samples_per_journey=max_samples_per_journey,
            top_n_events=top_n_events,
            top_events=top_events,
        )
        batch_df.write_parquet(part_path)
        total_snapshots += batch_df.height
        written_parts.append(part_path)
        elapsed = time.perf_counter() - started
        print(
            f"      batch {batch_idx:,}/{n_batches:,} written | "
            f"journeys {min(start + batch_size, labeled.height):,}/{labeled.height:,} | "
            f"batch snapshots {batch_df.height:,} | total snapshots {total_snapshots:,} | "
            f"elapsed {elapsed/60:.1f} min",
            flush=True,
        )

    if not written_parts:
        empty = pl.DataFrame()
        empty.write_parquet(output)
        print("      no shards produced; wrote empty parquet", flush=True)
        return empty

    print(f"      concatenating {len(written_parts):,} shards -> {output.name}", flush=True)
    shard_scans = [pl.scan_parquet(str(part)) for part in sorted(written_parts)]
    pl.concat(shard_scans, how="diagonal_relaxed").sink_parquet(output)

    summary = pl.scan_parquet(output)
    final_rows = summary.select(pl.len()).collect().item()
    print(f"      {int(final_rows):,} snapshots written", flush=True)
    if "label" in pl.read_parquet(output, n_rows=0).columns:
        for row in (
            summary
            .group_by("label")
            .agg(pl.len().alias("n"))
            .sort("label")
            .collect()
            .iter_rows(named=True)
        ):
            print(f"      label={row['label']}: {row['n']:,}", flush=True)
    return pl.scan_parquet(output).head(0).collect()


def build_open_realistic_features(
    open_csv: Path,
    output: Path,
) -> pl.DataFrame:
    """Build one cached realistic feature row per open/test journey."""
    print(f"[open] Building {output.name} ...")
    if not open_csv.exists():
        raise FileNotFoundError(f"Missing open journey file: {open_csv}")
    events = pl.read_csv(open_csv)
    features = build_open_journey_realistic_features(events)
    features.write_parquet(output)
    print(f"      {features.shape[0]:,} open journey rows written")
    return features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build training parquets from cleaned event log.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing dat_train1_clean.csv and where parquets will be written.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-samples-per-journey", type=int, default=60)
    parser.add_argument("--top-n-events", type=int, default=50)
    parser.add_argument(
        "--realistic-max-journeys",
        type=int,
        default=None,
        help="Optional cap for building the realistic multi-snapshot feature table.",
    )
    parser.add_argument(
        "--realistic-batch-size",
        type=int,
        default=5_000,
        help="Number of labeled journeys per realistic feature shard.",
    )
    parser.add_argument(
        "--resume-realistic",
        action="store_true",
        help="Reuse existing data/feature_engineered/realistic_parts/part_*.parquet shards.",
    )
    parser.add_argument(
        "--only-realistic",
        action="store_true",
        help="Use existing journeys_labeled.parquet and only rebuild the realistic feature table.",
    )
    parser.add_argument(
        "--only-open-realistic",
        action="store_true",
        help="Only rebuild cached realistic features for open_journeys1.csv.",
    )
    opts = parser.parse_args(argv)

    data_dir: Path = opts.data_dir
    if opts.only_open_realistic:
        build_open_realistic_features(
            data_dir / "open_journeys1.csv",
            data_dir / "open_journeys_realistic_features.parquet",
        )
        print("\nDone. Run ctmc_pipeline truncated to score cached open features.")
        return

    clean_csv = data_dir / "dat_train1_clean.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(
            f"Missing input file: {clean_csv}\n"
            "Place dat_train1_clean.csv in the data/ directory and re-run."
        )

    labeled_path = data_dir / "journeys_labeled.parquet"
    effective_realistic_max_journeys = opts.realistic_max_journeys
    if opts.only_realistic:
        if not labeled_path.exists():
            raise FileNotFoundError(f"Missing {labeled_path}; run without --only-realistic first.")
        if opts.realistic_max_journeys is not None:
            # Avoid materializing the full nested journey table just to run a
            # bounded smoke/test build.  The parquet is hundreds of MB and can
            # put memory pressure on smaller machines before batching begins.
            labeled = (
                pl.scan_parquet(labeled_path)
                .filter(pl.col("journey_status").is_in(["successful", "incomplete"]))
                .head(opts.realistic_max_journeys)
                .collect()
            )
            effective_realistic_max_journeys = None
        else:
            labeled = pl.read_parquet(labeled_path)
        if not _journey_struct_has_event_name(labeled):
            raise ValueError(
                f"{labeled_path} was built before journey structs included event_name. "
                "Run once without --only-realistic to refresh journeys_flattened.parquet "
                "and journeys_labeled.parquet, then rerun --only-realistic if needed."
            )
    else:
        flattened = build_flattened(clean_csv, data_dir / "journeys_flattened.parquet")
        labeled = build_labeled(flattened, clean_csv, labeled_path)
        build_training(labeled, data_dir / "journey_training_optionA.parquet", opts.random_state)
    build_realistic_training(
        labeled,
        data_dir / "journey_training_realistic_truncation.parquet",
        random_state=opts.random_state,
        max_samples_per_journey=opts.max_samples_per_journey,
        top_n_events=opts.top_n_events,
        max_journeys=effective_realistic_max_journeys,
        batch_size=opts.realistic_batch_size,
        resume=opts.resume_realistic,
    )
    if opts.only_realistic:
        print("\nDone. Run --only-open-realistic separately to refresh cached open/test features.")
        return

    build_open_realistic_features(
        data_dir / "open_journeys1.csv",
        data_dir / "open_journeys_realistic_features.parquet",
    )
    print("\nDone. Run ctmc_submission.py to generate Kaggle submissions.")


if __name__ == "__main__":
    main()
