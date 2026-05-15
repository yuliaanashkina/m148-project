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
python -m src.models.ctmc_pipeline clustered --use-spectral-clustering
                                          # fit/predict only clustered CTMC

Common options
--------------
--max-journeys N      rows of training transition data   (default: 100_000)
--n-clusters N        k for ClusteredCTMC               (default: 3)
--use-spectral-clustering
                       use spectral clustering instead of KMeans for CTMC segments
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
  results/models/timeout_absorbing_ctmc.joblib
  results/models/semi_markov_timeout.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

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
        from .ctmc import (
            CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC,
            SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC,
        )
    except ImportError:
        from ctmc import (
            CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC,
            SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC,
        )
    return CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC, SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC


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

    Returns a dict with key 'n_clusters' when the corresponding trial file is
    present in results/.
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

    return params


def train(
    n_clusters: int = 3,
    use_spectral_clustering: bool = False,
    max_journeys: int | None = 100_000,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> tuple:
    """Fit GlobalCTMC, ClusteredCTMC, and timeout-aware models; serialize."""
    import joblib

    CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC, SemiMarkovTimeoutModel, _ = _import_ctmc()

    print(f"Loading up to {max_journeys:,} training journeys...")
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys, customer_actions_only=True)
    timeout_transitions = data.timeout_transition_table(max_journeys=max_journeys)
    print(f"  {len(transitions):,} transitions from {transitions['id'].nunique():,} journeys")
    print(f"  {len(timeout_transitions):,} timeout-aware transitions")

    print("Fitting GlobalCTMC...")
    global_ctmc = GlobalCTMC().fit(transitions)

    cluster_method = "spectral" if use_spectral_clustering else "kmeans"
    print(f"Fitting ClusteredCTMC (method={cluster_method}, k={n_clusters})...")
    clustered = ClusteredCTMC(
        n_clusters=n_clusters,
        use_spectral_clustering=use_spectral_clustering,
    ).fit(transitions)
    print(clustered.cluster_summary().to_string(index=False))

    print("Fitting TimeoutAbsorbingCTMC...")
    timeout_ctmc = TimeoutAbsorbingCTMC().fit(timeout_transitions)

    print("Fitting SemiMarkovTimeoutModel...")
    semi_markov = SemiMarkovTimeoutModel().fit(timeout_transitions)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(global_ctmc,  models_dir / "global_ctmc.joblib")
    joblib.dump(clustered,    models_dir / "clustered_ctmc.joblib")
    joblib.dump(timeout_ctmc, models_dir / "timeout_absorbing_ctmc.joblib")
    joblib.dump(semi_markov,  models_dir / "semi_markov_timeout.joblib")
    print(f"Saved models -> {models_dir}")

    return global_ctmc, clustered, timeout_ctmc, semi_markov


def run_clustered_only(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    n_clusters: int = 3,
    use_spectral_clustering: bool = False,
    max_journeys: int | None = 100_000,
) -> pd.DataFrame:
    """Fit GlobalCTMC + ClusteredCTMC and write only the clustered submission."""
    import joblib

    CTMCData, GlobalCTMC, ClusteredCTMC, *_ = _import_ctmc()
    (
        load_test_events, test_features_from_events,
        write_flattened_all0_template, write_submission,
        brier_report, calibrate_prevalence, TEST_PREVALENCE,
    ) = _import_submission_helpers()

    method = "spectral" if use_spectral_clustering else "kmeans"
    print(f"Loading up to {max_journeys:,} training journeys...")
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys, customer_actions_only=True)
    print(f"  {len(transitions):,} transitions from {transitions['id'].nunique():,} journeys")

    print("Fitting GlobalCTMC fallback...")
    global_ctmc = GlobalCTMC().fit(transitions)

    print(f"Fitting ClusteredCTMC only (method={method}, k={n_clusters})...")
    clustered = ClusteredCTMC(
        n_clusters=n_clusters,
        use_spectral_clustering=use_spectral_clustering,
    ).fit(transitions)
    print(clustered.cluster_summary().to_string(index=False))

    models_dir.mkdir(parents=True, exist_ok=True)
    suffix = "spectral" if use_spectral_clustering else "kmeans"
    joblib.dump(global_ctmc, models_dir / f"{suffix}_global_ctmc.joblib")
    joblib.dump(clustered, models_dir / f"{suffix}_clustered_ctmc.joblib")

    if not test_events_path.exists():
        raise FileNotFoundError(
            f"Test events file not found: {test_events_path}. "
            "Add data/open_journeys1.csv to generate submissions."
        )

    print(f"Loading test events from {test_events_path}...")
    raw_events = load_test_events(test_events_path)
    events = raw_events[raw_events["ed_id"].isin(data.customer_action_states(include_success=True))].copy()
    if not sample_path.exists():
        write_flattened_all0_template(raw_events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    features = test_features_from_events(events, clustered.feature_builder)
    print(f"Test prevalence target: {TEST_PREVALENCE:.4f}")
    probs_raw = clustered.predict_success_probability(
        features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_model=global_ctmc,
    )
    probs = calibrate_prevalence(probs_raw)
    brier_report(f"{suffix}_clustered_ctmc", probs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ctmc_{suffix}_clustered_submission.csv"
    output = write_submission(
        features["id"],
        probs,
        output_path,
        sample_path=sample_path,
    )
    print(f"Wrote clustered-only submission: {output_path}")
    return output


def run_truncated_snapshot_ctmc(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    n_clusters: int = 3,
    use_spectral_clustering: bool = False,
    max_rows: int | None = 100_000,
) -> pd.DataFrame:
    """Fit CTMC using realistic truncated snapshots and suffix transitions."""
    import joblib

    CTMCData, *_, SnapshotFeatureClusteredCTMC = _import_ctmc()
    (
        _load_test_events, _test_features_from_events,
        write_flattened_all0_template, write_submission,
        brier_report, calibrate_prevalence, _TEST_PREVALENCE,
    ) = _import_submission_helpers()
    try:
        from ..data_engineering.preprocess import build_open_journey_realistic_features
    except ImportError:
        from src.data_engineering.preprocess import build_open_journey_realistic_features
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    data = CTMCData()
    print(f"Loading up to {max_rows:,} realistic truncated snapshots...")
    snapshots = data.load_realistic_snapshot_features(max_rows=max_rows)
    if snapshots.empty:
        raise FileNotFoundError(
            f"No realistic snapshot features found at {data.realistic_training_path}. "
            "Run python -m src.data_engineering.build_training_parquets first."
        )
    print(f"  {len(snapshots):,} snapshots from {snapshots['id'].nunique():,} journeys")

    print("Building suffix timeout transitions from each snapshot...")
    suffix_transitions = data.suffix_timeout_transition_table(snapshots=snapshots)
    print(f"  {len(suffix_transitions):,} suffix transitions")

    method = "spectral" if use_spectral_clustering else "kmeans"
    print(f"Fitting SnapshotFeatureClusteredCTMC (method={method}, k={n_clusters})...")
    model = SnapshotFeatureClusteredCTMC(
        n_clusters=n_clusters,
        use_spectral_clustering=use_spectral_clustering,
    ).fit(snapshots, suffix_transitions)
    print(model.cluster_summary().to_string(index=False))

    probs_raw = model.predict_success_probability(snapshots)
    probs = calibrate_prevalence(probs_raw)
    y = snapshots["label"].astype(int).to_numpy()
    probs_clip = np.clip(probs, 1e-6, 1 - 1e-6)
    metrics = pd.DataFrame([{
        "model": f"truncated_snapshot_{method}_ctmc",
        "n_snapshots": len(snapshots),
        "n_journeys": snapshots["id"].nunique(),
        "brier_score": brier_score_loss(y, probs_clip),
        "log_loss": log_loss(y, probs_clip, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, probs_clip),
        "average_precision": average_precision_score(y, probs_clip),
        "mean_prob": float(np.mean(probs)),
    }])
    brier_report(metrics.loc[0, "model"], probs)

    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "spectral" if use_spectral_clustering else "kmeans"
    joblib.dump(model, models_dir / f"truncated_snapshot_{suffix}_ctmc.joblib")
    metrics.to_csv(output_dir / f"ctmc_truncated_snapshot_{suffix}_metrics.csv", index=False)
    pd.DataFrame({
        "snapshot_id": snapshots["snapshot_id"],
        "id": snapshots["id"],
        "label": snapshots["label"],
        "order_shipped": probs,
    }).to_csv(output_dir / f"ctmc_truncated_snapshot_{suffix}_predictions.csv", index=False)

    if data.open_realistic_features_path.exists():
        print(f"Loading cached open journey features from {data.open_realistic_features_path}...")
        open_features = data.load_open_realistic_features()
    else:
        if not test_events_path.exists():
            raise FileNotFoundError(
                f"Test events file not found: {test_events_path}. "
                "Add data/open_journeys1.csv to generate submissions."
            )
        print(f"Building open journey features from {test_events_path}...")
        open_events = pl.read_csv(test_events_path)
        open_features_pl = build_open_journey_realistic_features(open_events)
        data.open_realistic_features_path.parent.mkdir(parents=True, exist_ok=True)
        open_features_pl.write_parquet(data.open_realistic_features_path)
        open_features = open_features_pl.to_pandas()
    print(f"  {len(open_features):,} open journey feature rows")

    if sample_path is not None and not sample_path.exists():
        if not test_events_path.exists():
            raise FileNotFoundError(f"Cannot create sample template; missing {test_events_path}")
        template_events = pl.read_csv(test_events_path).select("id").unique().to_pandas()
        write_flattened_all0_template(template_events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    open_probs_raw = model.predict_success_probability(open_features)
    open_probs = calibrate_prevalence(open_probs_raw)
    brier_report(f"truncated_snapshot_{suffix}_submission", open_probs)
    submission = write_submission(
        open_features["id"],
        open_probs,
        output_dir / f"ctmc_truncated_snapshot_{suffix}_submission.csv",
        sample_path=sample_path,
    )
    print(
        f"Wrote truncated snapshot metrics/predictions/submission to {output_dir} "
        f"({len(submission):,} submission rows)"
    )
    return metrics


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
    timeout_ctmc=None,
    semi_markov=None,
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
        global_ctmc  = joblib.load(models_dir / "global_ctmc.joblib")
        clustered    = joblib.load(models_dir / "clustered_ctmc.joblib")
        timeout_ctmc = joblib.load(models_dir / "timeout_absorbing_ctmc.joblib")
        semi_markov  = joblib.load(models_dir / "semi_markov_timeout.joblib")

    print(f"Loading test events from {test_events_path}...")
    raw_events = load_test_events(test_events_path)
    CTMCData, *_ = _import_ctmc()
    state_set = CTMCData().customer_action_states(include_success=True)
    events = raw_events[raw_events["ed_id"].isin(state_set)].copy()

    if not sample_path.exists():
        write_flattened_all0_template(raw_events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    # Build features on customer-action states only. Company/system events do
    # not reset the 60-day inactivity clock.
    clustered_features = test_features_from_events(events, clustered.feature_builder)

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
    timeout_probs_raw = timeout_ctmc.predict_success_probability(clustered_features["current_state"])
    semi_markov_probs_raw = semi_markov.predict_success_probability(clustered_features["current_state"])

    blended_raw = 0.25 * (
        global_probs_raw
        + clustered_probs_raw
        + timeout_probs_raw
        + semi_markov_probs_raw
    )

    global_probs          = calibrate_prevalence(global_probs_raw)
    clustered_probs       = calibrate_prevalence(clustered_probs_raw)
    timeout_probs         = calibrate_prevalence(timeout_probs_raw)
    semi_markov_probs     = calibrate_prevalence(semi_markov_probs_raw)
    blended_probs         = calibrate_prevalence(blended_raw)

    brier_report("global_ctmc",             global_probs)
    brier_report("clustered_ctmc",          clustered_probs)
    brier_report("timeout_absorbing_ctmc",  timeout_probs)
    brier_report("semi_markov_timeout",     semi_markov_probs)
    brier_report("blend_ctmc",              blended_probs)

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
        "timeout_absorbing": write_submission(
            clustered_features["id"], timeout_probs,
            output_dir / "ctmc_timeout_absorbing_submission.csv", sample_path=sample_path,
        ),
        "semi_markov_timeout": write_submission(
            clustered_features["id"], semi_markov_probs,
            output_dir / "ctmc_semi_markov_timeout_submission.csv", sample_path=sample_path,
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
        choices=["train", "predict", "all", "tune-k", "clustered", "truncated"],
        help=(
            "train: fit and save models; "
            "predict: load models and write CSVs; "
            "all: train then predict; "
            "tune-k: Optuna search for best k, then all; "
            "clustered: fit/predict only GlobalCTMC + ClusteredCTMC; "
            "truncated: fit CTMC from realistic truncated snapshots"
        ),
    )
    parser.add_argument("--max-journeys",    type=int,   default=100_000)
    parser.add_argument("--n-clusters",      type=int,   default=3)
    parser.add_argument(
        "--use-spectral-clustering",
        action="store_true",
        help="Use spectral clustering instead of KMeans inside ClusteredCTMC.",
    )
    parser.add_argument("--n-tune-trials",   type=int,   default=20)
    parser.add_argument("--k-min",           type=int,   default=2)
    parser.add_argument("--k-max",           type=int,   default=12)
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
        help="Read best n_clusters from Optuna trial CSVs in results/",
    )
    args = parser.parse_args()

    n_clusters = args.n_clusters

    if args.use_optuna_results:
        print("Reading best hyperparameters from Optuna results...")
        optuna_params = best_optuna_params()
        if "n_clusters" in optuna_params:
            n_clusters = optuna_params["n_clusters"]

    if args.mode == "clustered":
        if not args.test_events.exists():
            print(
                f"Test events file not found: {args.test_events}\n"
                "Add data/open_journeys1.csv from Kaggle to generate submissions."
            )
            return
        run_clustered_only(
            test_events_path=args.test_events,
            sample_path=args.sample,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            n_clusters=n_clusters,
            use_spectral_clustering=args.use_spectral_clustering,
            max_journeys=args.max_journeys,
        )
        return

    if args.mode == "truncated":
        run_truncated_snapshot_ctmc(
            test_events_path=args.test_events,
            sample_path=args.sample,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            n_clusters=n_clusters,
            use_spectral_clustering=args.use_spectral_clustering,
            max_rows=args.max_journeys,
        )
        return

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
        and not args.use_spectral_clustering
        and (args.models_dir / "global_ctmc.joblib").exists()
        and (args.models_dir / "clustered_ctmc.joblib").exists()
        and (args.models_dir / "timeout_absorbing_ctmc.joblib").exists()
        and (args.models_dir / "semi_markov_timeout.joblib").exists()
    )

    if args.mode in ("train", "all"):
        if models_cached and args.mode == "all":
            print(
                f"Cached models found in {args.models_dir}. "
                "Skipping training (use --no-cache to force refit)."
            )
            global_ctmc = clustered = timeout_ctmc = semi_markov = None
        else:
            global_ctmc, clustered, timeout_ctmc, semi_markov = train(
                n_clusters=n_clusters,
                use_spectral_clustering=args.use_spectral_clustering,
                max_journeys=args.max_journeys,
                models_dir=args.models_dir,
            )
    else:
        global_ctmc = clustered = timeout_ctmc = semi_markov = None

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
            timeout_ctmc=timeout_ctmc,
            semi_markov=semi_markov,
        )


if __name__ == "__main__":
    main()
