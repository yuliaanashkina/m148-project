"""
Standalone CTMC pipeline — fit models, tune clustering k, generate submissions.

Run this instead of the notebook when you just need fresh CSVs.

Usage
-----
# First time: fit all models and write submissions in one shot
python -m src.models.ctmc_pipeline all

# Or separate phases:
python -m src.models.ctmc_pipeline train    # fit + serialize models
python -m src.models.ctmc_pipeline predict  # load cached models + write CSVs
python -m src.models.ctmc_pipeline tune-k   # Optuna over k, then train + predict

Common options
--------------
--max-journeys N      rows of training transition data   (default: 100_000)
--n-clusters N        k for ClusteredCTMC               (default: 3)
--n-tune-trials N     Optuna trials when using tune-k   (default: 20)
--models-dir PATH     where to save/load .joblib files  (default: results/models/)
--output-dir PATH     where to write submission CSVs    (default: results/submissions/)
--test-events PATH    open_journeys1.csv from Kaggle    (default: data/open_journeys1.csv)
--sample PATH         Kaggle all-zeros template CSV     (default: data/open_journeys1_flattened_all0.csv)
--no-cache            always retrain even if models exist

Model files
-----------
  results/models/global_ctmc.joblib
  results/models/clustered_ctmc.joblib
  results/models/xgboost_rate_ctmc.joblib
  results/models/time_stratified_ctmc.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "results" / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "submissions"
DEFAULT_TEST_EVENTS = DATA_DIR / "open_journeys1.csv"
DEFAULT_SAMPLE = DATA_DIR / "open_journeys1_flattened_all0.csv"

SUCCESS_STATE = 28
HORIZON_SECONDS = 60 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# Imports (kept lazy so the module can be imported without all deps loaded)
# ---------------------------------------------------------------------------

def _import_ctmc():
    try:
        from .ctmc import CTMCData, GlobalCTMC, ClusteredCTMC, XGBoostRateCTMC, TimeStratifiedCTMC
    except ImportError:
        from ctmc import CTMCData, GlobalCTMC, ClusteredCTMC, XGBoostRateCTMC, TimeStratifiedCTMC
    return CTMCData, GlobalCTMC, ClusteredCTMC, XGBoostRateCTMC, TimeStratifiedCTMC


def _import_submission_helpers():
    try:
        from .ctmc_submission import (
            load_test_events,
            test_features_from_events,
            write_flattened_all0_template,
            write_submission,
        )
        from .tabular_submission import brier_report, calibrate_prevalence, TEST_PREVALENCE
    except ImportError:
        from ctmc_submission import (
            load_test_events,
            test_features_from_events,
            write_flattened_all0_template,
            write_submission,
        )
        from tabular_submission import brier_report, calibrate_prevalence, TEST_PREVALENCE
    return load_test_events, test_features_from_events, write_flattened_all0_template, write_submission, brier_report, calibrate_prevalence, TEST_PREVALENCE


# ---------------------------------------------------------------------------
# Train phase
# ---------------------------------------------------------------------------

def best_optuna_params() -> dict:
    """Read best hyperparameters from Optuna trial CSVs if they exist.

    Returns a dict with keys 'n_clusters' and/or 'bin_edges_seconds' when the
    corresponding trial files are present in results/.
    """
    params = {}
    results_dir = PROJECT_ROOT / "results"

    k_path = results_dir / "optuna_ctmc_clustering_k_trials.csv"
    if k_path.exists():
        df = pd.read_csv(k_path)
        df = df[df["value"].notna()]
        if not df.empty:
            best_row = df.loc[df["value"].idxmin()]
            params["n_clusters"] = int(best_row["n_clusters"])
            print(f"  [Optuna] best n_clusters = {params['n_clusters']}  (brier={best_row['value']:.5f})")

    ts_path = results_dir / "optuna_time_stratified_ctmc_trials.csv"
    if ts_path.exists():
        df = pd.read_csv(ts_path)
        df = df[df["value"].notna()]
        if not df.empty:
            best_row = df.loc[df["value"].idxmin()]
            e1h = float(best_row["edge1_hours"])
            dh  = float(best_row["delta_hours"])
            e2h = e1h + dh
            params["bin_edges_seconds"] = (e1h * 3600.0, e2h * 3600.0)
            print(
                f"  [Optuna] best bin edges = [{e1h:.1f}h, {e2h:.1f}h]  "
                f"(brier={best_row['value']:.5f})"
            )

    return params


def train(
    n_clusters: int = 3,
    bin_edges_seconds: tuple[float, ...] = (86_400.0, 604_800.0),
    max_journeys: int | None = 100_000,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> tuple:
    """Fit GlobalCTMC, ClusteredCTMC, XGBoostRateCTMC, TimeStratifiedCTMC; serialize."""
    import joblib

    CTMCData, GlobalCTMC, ClusteredCTMC, XGBoostRateCTMC, TimeStratifiedCTMC = _import_ctmc()

    print(f"Loading up to {max_journeys:,} training journeys...")
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys)
    print(f"  {len(transitions):,} transitions from {transitions['id'].nunique():,} journeys")

    print("Fitting GlobalCTMC...")
    global_ctmc = GlobalCTMC().fit(transitions)

    print(f"Fitting ClusteredCTMC (k={n_clusters})...")
    clustered = ClusteredCTMC(n_clusters=n_clusters).fit(transitions)
    print(clustered.cluster_summary().to_string(index=False))

    print("Fitting XGBoostRateCTMC...")
    xgb_ctmc = XGBoostRateCTMC().fit(transitions)

    edge_labels = [f"{e/3600:.1f}h" for e in bin_edges_seconds]
    print(f"Fitting TimeStratifiedCTMC (edges={edge_labels})...")
    time_stratified = TimeStratifiedCTMC(bin_edges_seconds=bin_edges_seconds).fit(transitions)
    print(time_stratified.bin_summary().to_string(index=False))

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_ctmc,      models_dir / "global_ctmc.joblib")
    joblib.dump(clustered,        models_dir / "clustered_ctmc.joblib")
    joblib.dump(xgb_ctmc,         models_dir / "xgboost_rate_ctmc.joblib")
    joblib.dump(time_stratified,  models_dir / "time_stratified_ctmc.joblib")
    print(f"Saved models → {models_dir}")

    return global_ctmc, clustered, xgb_ctmc, time_stratified


# ---------------------------------------------------------------------------
# Predict phase
# ---------------------------------------------------------------------------

def predict(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    global_ctmc=None,
    clustered=None,
    xgb_ctmc=None,
    time_stratified=None,
) -> dict[str, pd.DataFrame]:
    """Load (or reuse) fitted models and write Kaggle submission CSVs."""
    import joblib

    (
        load_test_events, test_features_from_events,
        write_flattened_all0_template, write_submission,
        brier_report, calibrate_prevalence, TEST_PREVALENCE,
    ) = _import_submission_helpers()

    # Load models if not passed in directly.
    if global_ctmc is None:
        print(f"Loading models from {models_dir}...")
        global_ctmc      = joblib.load(models_dir / "global_ctmc.joblib")
        clustered        = joblib.load(models_dir / "clustered_ctmc.joblib")
        xgb_ctmc         = joblib.load(models_dir / "xgboost_rate_ctmc.joblib")
        time_stratified  = joblib.load(models_dir / "time_stratified_ctmc.joblib")

    print(f"Loading test events from {test_events_path}...")
    events = load_test_events(test_events_path)

    if not sample_path.exists():
        write_flattened_all0_template(events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    # Build features — clustered_features has current_state + total_observed_time
    # needed by both ClusteredCTMC and TimeStratifiedCTMC.
    clustered_features = test_features_from_events(events, clustered.feature_builder)
    xgb_features       = xgb_ctmc.feature_builder.features_from_events(events)

    print(f"Test prevalence target: {TEST_PREVALENCE:.4f}")

    # Raw probabilities.
    global_probs_raw = global_ctmc.absorption_probability(
        clustered_features["current_state"],
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
    )
    clustered_probs_raw = clustered.predict_success_probability(
        clustered_features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_model=global_ctmc,
    )
    print("  Running XGBoostRateCTMC (computing per-journey Q matrices)...")
    xgb_probs_raw = xgb_ctmc.predict_success_probability(
        xgb_features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
    )
    time_stratified_probs_raw = time_stratified.predict_success_probability(
        clustered_features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
    )

    blended_raw = 0.25 * (
        global_probs_raw
        + clustered_probs_raw
        + xgb_probs_raw
        + time_stratified_probs_raw
    )

    global_probs          = calibrate_prevalence(global_probs_raw)
    clustered_probs       = calibrate_prevalence(clustered_probs_raw)
    xgb_probs             = calibrate_prevalence(xgb_probs_raw)
    time_stratified_probs = calibrate_prevalence(time_stratified_probs_raw)
    blended_probs         = calibrate_prevalence(blended_raw)

    brier_report("global_ctmc",          global_probs)
    brier_report("clustered_ctmc",       clustered_probs)
    brier_report("xgboost_rate_ctmc",    xgb_probs)
    brier_report("time_stratified_ctmc", time_stratified_probs)
    brier_report("blend_ctmc",           blended_probs)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "global": write_submission(
            clustered_features["id"], global_probs,
            output_dir / "ctmc_global_submission.csv", sample_path=sample_path,
        ),
        "clustered": write_submission(
            clustered_features["id"], clustered_probs,
            output_dir / "ctmc_clustered_submission.csv", sample_path=sample_path,
        ),
        "xgboost_rate": write_submission(
            xgb_features["id"], xgb_probs,
            output_dir / "ctmc_xgboost_rate_submission.csv", sample_path=sample_path,
        ),
        "time_stratified": write_submission(
            clustered_features["id"], time_stratified_probs,
            output_dir / "ctmc_time_stratified_submission.csv", sample_path=sample_path,
        ),
        "blend": write_submission(
            clustered_features["id"], blended_probs,
            output_dir / "ctmc_blend_submission.csv", sample_path=sample_path,
        ),
    }

    print("\nWrote submissions:")
    for name, df in outputs.items():
        print(f"  {name}: {len(df):,} rows  mean_prob={df['order_shipped'].mean():.4f}")
    return outputs


# ---------------------------------------------------------------------------
# Tune-k phase
# ---------------------------------------------------------------------------

def tune_k(
    max_journeys: int = 100_000,
    n_trials: int = 20,
    k_min: int = 2,
    k_max: int = 12,
) -> int:
    """Run Optuna over ClusteredCTMC n_clusters; return the best k."""
    try:
        from .optuna_tuning import tune_ctmc_clustering, save_trials
    except ImportError:
        from optuna_tuning import tune_ctmc_clustering, save_trials

    print(f"Tuning n_clusters over [{k_min}, {k_max}] with {n_trials} Optuna trials...")
    study = tune_ctmc_clustering(
        max_journeys=max_journeys,
        n_trials=n_trials,
        k_min=k_min,
        k_max=k_max,
    )
    save_trials(study)

    best_k = int(study.best_params["n_clusters"])
    print(f"\nOptuna best n_clusters = {best_k}  (brier = {study.best_value:.5f})")

    # Print the full trial table sorted by score.
    rows = [
        {"n_clusters": t.params["n_clusters"], "brier_score": t.value}
        for t in study.trials
        if t.value is not None
    ]
    df = pd.DataFrame(rows).sort_values("brier_score").drop_duplicates("n_clusters")
    print(df.to_string(index=False))
    return best_k


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit CTMC models and generate Kaggle submission CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["train", "predict", "all", "tune-k"],
        help=(
            "train: fit and save models; "
            "predict: load models and write CSVs; "
            "all: train then predict; "
            "tune-k: Optuna search for best k, then all"
        ),
    )
    parser.add_argument("--max-journeys",    type=int,   default=100_000)
    parser.add_argument("--n-clusters",      type=int,   default=3)
    parser.add_argument("--n-tune-trials",   type=int,   default=20)
    parser.add_argument("--k-min",           type=int,   default=2)
    parser.add_argument("--k-max",           type=int,   default=12)
    parser.add_argument("--bin-edge1-hours", type=float, default=24.0,
                        help="TimeStratifiedCTMC early/mid boundary in hours")
    parser.add_argument("--bin-edge2-hours", type=float, default=168.0,
                        help="TimeStratifiedCTMC mid/late boundary in hours")
    parser.add_argument("--models-dir",      type=Path,  default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir",      type=Path,  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-events",     type=Path,  default=DEFAULT_TEST_EVENTS)
    parser.add_argument("--sample",          type=Path,  default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Refit models even if .joblib files already exist in --models-dir",
    )
    parser.add_argument(
        "--use-optuna-results",
        action="store_true",
        help="Read best n_clusters and bin edges from Optuna trial CSVs in results/",
    )
    args = parser.parse_args()

    n_clusters = args.n_clusters
    bin_edges = (args.bin_edge1_hours * 3600.0, args.bin_edge2_hours * 3600.0)

    if args.use_optuna_results:
        print("Reading best hyperparameters from Optuna results...")
        optuna_params = best_optuna_params()
        if "n_clusters" in optuna_params:
            n_clusters = optuna_params["n_clusters"]
        if "bin_edges_seconds" in optuna_params:
            bin_edges = optuna_params["bin_edges_seconds"]

    if args.mode == "tune-k":
        n_clusters = tune_k(
            max_journeys=args.max_journeys,
            n_trials=args.n_tune_trials,
            k_min=args.k_min,
            k_max=args.k_max,
        )
        # Fall through to train + predict with the tuned k.
        args.mode = "all"

    models_cached = (
        not args.no_cache
        and (args.models_dir / "global_ctmc.joblib").exists()
        and (args.models_dir / "clustered_ctmc.joblib").exists()
        and (args.models_dir / "xgboost_rate_ctmc.joblib").exists()
        and (args.models_dir / "time_stratified_ctmc.joblib").exists()
    )

    if args.mode in ("train", "all"):
        if models_cached and args.mode == "all":
            print(
                f"Cached models found in {args.models_dir}. "
                "Skipping training (use --no-cache to force refit)."
            )
            global_ctmc = clustered = xgb_ctmc = time_stratified = None
        else:
            global_ctmc, clustered, xgb_ctmc, time_stratified = train(
                n_clusters=n_clusters,
                bin_edges_seconds=bin_edges,
                max_journeys=args.max_journeys,
                models_dir=args.models_dir,
            )
    else:
        global_ctmc = clustered = xgb_ctmc = time_stratified = None

    if args.mode in ("predict", "all"):
        if not args.test_events.exists():
            print(
                f"Test events file not found: {args.test_events}\n"
                "Add data/open_journeys1.csv from Kaggle to generate submissions."
            )
            return
        predict(
            test_events_path=args.test_events,
            sample_path=args.sample,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            global_ctmc=global_ctmc,
            clustered=clustered,
            xgb_ctmc=xgb_ctmc,
            time_stratified=time_stratified,
        )


if __name__ == "__main__":
    main()
