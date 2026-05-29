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
python -m src.models.ctmc_pipeline suite    # run all retained CTMC submission modes
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
--test-prevalence P   override prevalence calibration target
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
import os
from pathlib import Path

# Avoid Windows MKL/OpenMP hangs in sklearn KMeans/SpectralClustering. Users can
# still override these before launch if they intentionally want more threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
            SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC, HigherOrderCTMC,
            PiecewiseTimeVaryingTimeoutCTMC,
        )
    except ImportError:
        from ctmc import (
            CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC,
            SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC, HigherOrderCTMC,
            PiecewiseTimeVaryingTimeoutCTMC,
        )
    return (
        CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC,
        SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC, HigherOrderCTMC,
        PiecewiseTimeVaryingTimeoutCTMC,
    )


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


def _path_arg(value: str) -> Path:
    """Parse CLI path arguments and reject empty PowerShell variables."""
    if value is None or not str(value).strip():
        raise argparse.ArgumentTypeError(
            "path value is empty; set the PowerShell variable or pass a literal path"
        )
    return Path(value)


def _target_prevalence(default_prevalence: float, test_prevalence: float | None) -> float:
    """Resolve the prevalence target used for prior-probability calibration."""
    return float(default_prevalence if test_prevalence is None else test_prevalence)


def _calibrate_with_target(probs: np.ndarray, calibrate_prevalence, test_prevalence: float | None) -> np.ndarray:
    """Apply existing calibration, optionally overriding the Kaggle test prevalence."""
    if test_prevalence is None:
        return calibrate_prevalence(probs)
    return calibrate_prevalence(probs, test_prev=float(test_prevalence))


def _brier_report_with_target(name: str, probs: np.ndarray, brier_report, test_prevalence: float | None) -> None:
    """Print probability diagnostics against the active prevalence target."""
    if test_prevalence is None:
        brier_report(name, probs)
    else:
        brier_report(name, probs, test_prev=float(test_prevalence))


def _cached_models_are_timeout_aware(models_dir: Path) -> bool:
    """Return True only for the new two-absorbing-state cache format."""
    try:
        import joblib

        _CTMCData, _GlobalCTMC, _ClusteredCTMC, TimeoutAbsorbingCTMC, *_ = _import_ctmc()
        global_model = joblib.load(models_dir / "global_ctmc.joblib")
        clustered_model = joblib.load(models_dir / "clustered_ctmc.joblib")
    except Exception:
        return False
    return (
        isinstance(global_model, TimeoutAbsorbingCTMC)
        and bool(getattr(clustered_model, "timeout_absorbing", False))
    )


def train(
    n_clusters: int = 3,
    use_spectral_clustering: bool = False,
    max_journeys: int | None = 100_000,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> tuple:
    """Fit GlobalCTMC, ClusteredCTMC, and timeout-aware models; serialize."""
    import joblib

    CTMCData, GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC, SemiMarkovTimeoutModel, *_ = _import_ctmc()

    print(f"Loading up to {max_journeys:,} training journeys...")
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys, customer_actions_only=True)
    timeout_transitions = data.timeout_transition_table(max_journeys=max_journeys)
    print(f"  {len(transitions):,} transitions from {transitions['id'].nunique():,} journeys")
    print(f"  {len(timeout_transitions):,} timeout-aware transitions")

    print("Fitting global TimeoutAbsorbingCTMC...")
    global_ctmc = TimeoutAbsorbingCTMC().fit(timeout_transitions)

    cluster_method = "spectral" if use_spectral_clustering else "kmeans"
    print(f"Fitting timeout-aware ClusteredCTMC (method={cluster_method}, k={n_clusters})...")
    clustered = ClusteredCTMC(
        n_clusters=n_clusters,
        use_spectral_clustering=use_spectral_clustering,
        timeout_absorbing=True,
    ).fit(transitions, model_transitions=timeout_transitions)
    print(clustered.cluster_summary().to_string(index=False))

    timeout_ctmc = global_ctmc

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
    test_prevalence: float | None = None,
) -> pd.DataFrame:
    """Fit global + clustered timeout-aware CTMCs and write one submission."""
    import joblib

    CTMCData, _GlobalCTMC, ClusteredCTMC, TimeoutAbsorbingCTMC, *_ = _import_ctmc()
    (
        load_test_events, test_features_from_events,
        write_flattened_all0_template, write_submission,
        brier_report, calibrate_prevalence, TEST_PREVALENCE,
    ) = _import_submission_helpers()

    method = "spectral" if use_spectral_clustering else "kmeans"
    print(f"Loading up to {max_journeys:,} training journeys...")
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys, customer_actions_only=True)
    timeout_transitions = data.timeout_transition_table(max_journeys=max_journeys)
    print(f"  {len(transitions):,} transitions from {transitions['id'].nunique():,} journeys")
    print(f"  {len(timeout_transitions):,} timeout-aware transitions")

    print("Fitting global TimeoutAbsorbingCTMC fallback...")
    global_ctmc = TimeoutAbsorbingCTMC().fit(timeout_transitions)

    print(f"Fitting timeout-aware ClusteredCTMC only (method={method}, k={n_clusters})...")
    clustered = ClusteredCTMC(
        n_clusters=n_clusters,
        use_spectral_clustering=use_spectral_clustering,
        timeout_absorbing=True,
    ).fit(transitions, model_transitions=timeout_transitions)
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
    target_prev = _target_prevalence(TEST_PREVALENCE, test_prevalence)
    print(f"Test prevalence target: {target_prev:.4f}")
    probs_raw = clustered.predict_success_probability(
        features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_model=global_ctmc,
    )
    probs = _calibrate_with_target(probs_raw, calibrate_prevalence, test_prevalence)
    _brier_report_with_target(f"{suffix}_clustered_ctmc", probs, brier_report, test_prevalence)

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


def run_temporal_ctmc(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    max_journeys: int | None = 100_000,
    test_prevalence: float | None = None,
) -> pd.DataFrame:
    """Fit piecewise time-varying timeout CTMC and write one submission."""
    import joblib

    CTMCData, _GlobalCTMC, _ClusteredCTMC, _TimeoutAbsorbingCTMC, _SemiMarkovTimeoutModel, _SnapshotFeatureClusteredCTMC, _HigherOrderCTMC, PiecewiseTimeVaryingTimeoutCTMC = _import_ctmc()
    (
        load_test_events, _test_features_from_events,
        write_flattened_all0_template, write_submission,
        brier_report, calibrate_prevalence, TEST_PREVALENCE,
    ) = _import_submission_helpers()

    print(f"Loading up to {max_journeys:,} training journeys...")
    data = CTMCData()
    transitions = data.transition_table(max_journeys=max_journeys, customer_actions_only=True)
    timeout_transitions = data.timeout_transition_table(max_journeys=max_journeys)
    print(f"  {len(transitions):,} transitions from {transitions['id'].nunique():,} journeys")
    print(f"  {len(timeout_transitions):,} timeout-aware transitions")

    print("Fitting PiecewiseTimeVaryingTimeoutCTMC...")
    temporal = PiecewiseTimeVaryingTimeoutCTMC().fit(
        timeout_transitions,
        feature_transitions=transitions,
    )
    print(temporal.bin_summary().to_string(index=False))

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(temporal, models_dir / "piecewise_time_varying_ctmc.joblib")

    if not test_events_path.exists():
        raise FileNotFoundError(
            f"Test events file not found: {test_events_path}. "
            "Add the open journey event CSV to generate submissions."
        )

    print(f"Loading test events from {test_events_path}...")
    raw_events = load_test_events(test_events_path)
    events = raw_events[
        raw_events["ed_id"].isin(data.customer_action_states(include_success=True))
    ].copy()
    if not sample_path.exists():
        write_flattened_all0_template(raw_events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    features = temporal.feature_builder.features_from_events(events)
    target_prev = _target_prevalence(TEST_PREVALENCE, test_prevalence)
    print(f"Test prevalence target: {target_prev:.4f}")
    probs_raw = temporal.predict_success_probability(
        features,
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
    )
    probs = _calibrate_with_target(probs_raw, calibrate_prevalence, test_prevalence)
    _brier_report_with_target("temporal_ctmc", probs, brier_report, test_prevalence)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ctmc_temporal_submission.csv"
    output = write_submission(
        features["id"],
        probs,
        output_path,
        sample_path=sample_path,
    )
    print(f"Wrote temporal CTMC submission: {output_path}")
    return output


def run_higher_order_ctmc(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    order: int = 2,
    max_journeys: int | None = 100_000,
    test_prevalence: float | None = None,
) -> pd.DataFrame:
    """Fit an order-k augmented-state CTMC and write one submission."""
    import joblib

    CTMCData, GlobalCTMC, _ClusteredCTMC, _TimeoutAbsorbingCTMC, _SemiMarkovTimeoutModel, _SnapshotFeatureClusteredCTMC, HigherOrderCTMC, _PiecewiseTimeVaryingTimeoutCTMC = _import_ctmc()
    (
        load_test_events, _test_features_from_events,
        write_flattened_all0_template, write_submission,
        brier_report, calibrate_prevalence, TEST_PREVALENCE,
    ) = _import_submission_helpers()

    data = CTMCData()
    print(f"Loading up to {max_journeys:,} training journeys...")
    train_events = data.events(
        max_journeys=max_journeys,
        customer_actions_only=True,
        include_success=True,
    )
    print(f"  {len(train_events):,} customer-action events from {train_events['id'].nunique():,} journeys")

    print("Fitting first-order GlobalCTMC fallback...")
    transitions = data.transition_table(
        max_journeys=max_journeys,
        customer_actions_only=True,
        include_success=True,
    )
    global_ctmc = GlobalCTMC().fit(transitions)

    print(f"Fitting order-{order} augmented-state CTMC...")
    higher_order = HigherOrderCTMC(order=order).fit_from_events(train_events)
    print(f"  {len(higher_order.states_):,} augmented states")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(higher_order, models_dir / f"higher_order_{order}_ctmc.joblib")

    if not test_events_path.exists():
        raise FileNotFoundError(
            f"Test events file not found: {test_events_path}. "
            "Add data/open_journeys1.csv to generate submissions."
        )

    print(f"Loading test events from {test_events_path}...")
    raw_events = load_test_events(test_events_path)
    state_set = data.customer_action_states(include_success=True)
    events = raw_events[raw_events["ed_id"].isin(state_set)].copy()
    if not sample_path.exists():
        write_flattened_all0_template(raw_events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    histories = higher_order.histories_from_events(events)
    target_prev = _target_prevalence(TEST_PREVALENCE, test_prevalence)
    print(f"Test prevalence target: {target_prev:.4f}")
    probs_raw = higher_order.absorption_probability(
        histories["current_history"],
        success_state=SUCCESS_STATE,
        horizon_seconds=HORIZON_SECONDS,
        fallback_states=histories["current_state"],
        fallback_model=global_ctmc,
    )
    probs = _calibrate_with_target(probs_raw, calibrate_prevalence, test_prevalence)
    _brier_report_with_target(f"higher_order_{order}_ctmc", probs, brier_report, test_prevalence)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ctmc_higher_order_{order}_submission.csv"
    output = write_submission(
        histories["id"],
        probs,
        output_path,
        sample_path=sample_path,
    )
    print(f"Wrote order-{order} CTMC submission: {output_path}")
    return output


def run_truncated_snapshot_ctmc(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    n_clusters: int = 3,
    use_spectral_clustering: bool = False,
    max_rows: int | None = 100_000,
    test_prevalence: float | None = None,
) -> pd.DataFrame:
    """Fit CTMC using realistic truncated snapshots and suffix transitions."""
    import joblib

    CTMCData, _GlobalCTMC, _ClusteredCTMC, _TimeoutAbsorbingCTMC, _SemiMarkovTimeoutModel, SnapshotFeatureClusteredCTMC, _HigherOrderCTMC, _PiecewiseTimeVaryingTimeoutCTMC = _import_ctmc()
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
    probs = _calibrate_with_target(probs_raw, calibrate_prevalence, test_prevalence)
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
    _brier_report_with_target(metrics.loc[0, "model"], probs, brier_report, test_prevalence)

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

    if test_events_path.resolve() == DEFAULT_TEST_EVENTS.resolve():
        open_realistic_features_path = data.open_realistic_features_path
    else:
        open_realistic_features_path = (
            data.open_realistic_features_path.parent
            / f"{test_events_path.stem}_realistic_features.parquet"
        )

    if open_realistic_features_path.exists():
        print(f"Loading cached open journey features from {open_realistic_features_path}...")
        open_features = pl.read_parquet(open_realistic_features_path).to_pandas()
    else:
        if not test_events_path.exists():
            raise FileNotFoundError(
                f"Test events file not found: {test_events_path}. "
                "Add the open journey event CSV to generate submissions."
            )
        print(f"Building open journey features from {test_events_path}...")
        open_events = pl.read_csv(test_events_path)
        open_features_pl = build_open_journey_realistic_features(open_events)
        open_realistic_features_path.parent.mkdir(parents=True, exist_ok=True)
        open_features_pl.write_parquet(open_realistic_features_path)
        open_features = open_features_pl.to_pandas()
    print(f"  {len(open_features):,} open journey feature rows")

    if sample_path is not None and not sample_path.exists():
        if not test_events_path.exists():
            raise FileNotFoundError(f"Cannot create sample template; missing {test_events_path}")
        template_events = pl.read_csv(test_events_path).select("id").unique().to_pandas()
        write_flattened_all0_template(template_events, sample_path)
        print(f"Created Kaggle ID template: {sample_path}")

    open_probs_raw = model.predict_success_probability(open_features)
    open_probs = _calibrate_with_target(open_probs_raw, calibrate_prevalence, test_prevalence)
    _brier_report_with_target(f"truncated_snapshot_{suffix}_submission", open_probs, brier_report, test_prevalence)
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
    test_prevalence: float | None = None,
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

    target_prev = _target_prevalence(TEST_PREVALENCE, test_prevalence)
    print(f"Test prevalence target: {target_prev:.4f}")

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

    global_probs          = _calibrate_with_target(global_probs_raw, calibrate_prevalence, test_prevalence)
    clustered_probs       = _calibrate_with_target(clustered_probs_raw, calibrate_prevalence, test_prevalence)
    timeout_probs         = _calibrate_with_target(timeout_probs_raw, calibrate_prevalence, test_prevalence)
    semi_markov_probs     = _calibrate_with_target(semi_markov_probs_raw, calibrate_prevalence, test_prevalence)
    blended_probs         = _calibrate_with_target(blended_raw, calibrate_prevalence, test_prevalence)

    _brier_report_with_target("global_ctmc",             global_probs,      brier_report, test_prevalence)
    _brier_report_with_target("clustered_ctmc",          clustered_probs,   brier_report, test_prevalence)
    _brier_report_with_target("timeout_absorbing_ctmc",  timeout_probs,     brier_report, test_prevalence)
    _brier_report_with_target("semi_markov_timeout",     semi_markov_probs, brier_report, test_prevalence)
    _brier_report_with_target("blend_ctmc",              blended_probs,     brier_report, test_prevalence)

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


def run_suite(
    test_events_path: Path = DEFAULT_TEST_EVENTS,
    sample_path: Path = DEFAULT_SAMPLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    n_clusters: int = 3,
    max_journeys: int | None = 100_000,
    test_prevalence: float | None = None,
) -> dict[str, object]:
    """Run the retained CTMC submission suite for one open-journey file."""
    print("\n=== CTMC suite: global, KMeans clustered, timeout, semi-Markov, blend ===")
    global_ctmc, clustered, timeout_ctmc, semi_markov = train(
        n_clusters=n_clusters,
        use_spectral_clustering=False,
        max_journeys=max_journeys,
        models_dir=models_dir,
    )
    outputs: dict[str, object] = {
        "base": predict(
            test_events_path=test_events_path,
            sample_path=sample_path,
            output_dir=output_dir,
            models_dir=models_dir,
            global_ctmc=global_ctmc,
            clustered=clustered,
            timeout_ctmc=timeout_ctmc,
            semi_markov=semi_markov,
            test_prevalence=test_prevalence,
        )
    }

    print("\n=== CTMC suite: spectral clustered ===")
    outputs["spectral_clustered"] = run_clustered_only(
        test_events_path=test_events_path,
        sample_path=sample_path,
        output_dir=output_dir,
        models_dir=models_dir,
        n_clusters=n_clusters,
        use_spectral_clustering=True,
        max_journeys=max_journeys,
        test_prevalence=test_prevalence,
    )

    print("\n=== CTMC suite: piecewise time-varying timeout ===")
    outputs["temporal"] = run_temporal_ctmc(
        test_events_path=test_events_path,
        sample_path=sample_path,
        output_dir=output_dir,
        models_dir=models_dir,
        max_journeys=max_journeys,
        test_prevalence=test_prevalence,
    )

    for order in (2, 3):
        print(f"\n=== CTMC suite: higher-order order={order} ===")
        outputs[f"higher_order_{order}"] = run_higher_order_ctmc(
            test_events_path=test_events_path,
            sample_path=sample_path,
            output_dir=output_dir,
            models_dir=models_dir,
            order=order,
            max_journeys=max_journeys,
            test_prevalence=test_prevalence,
        )

    for use_spectral in (False, True):
        method = "spectral" if use_spectral else "kmeans"
        print(f"\n=== CTMC suite: truncated snapshot {method} ===")
        outputs[f"truncated_{method}"] = run_truncated_snapshot_ctmc(
            test_events_path=test_events_path,
            sample_path=sample_path,
            output_dir=output_dir,
            models_dir=models_dir,
            n_clusters=n_clusters,
            use_spectral_clustering=use_spectral,
            max_rows=max_journeys,
            test_prevalence=test_prevalence,
        )

    return outputs


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
        choices=["train", "predict", "all", "suite", "tune-k", "clustered", "truncated", "higher-order", "temporal"],
        help=(
            "train: fit and save models; "
            "predict: load models and write CSVs; "
            "all: train then predict; "
            "suite: run all retained CTMC submission modes; "
            "tune-k: Optuna search for best k, then all; "
            "clustered: fit/predict only GlobalCTMC + ClusteredCTMC; "
            "truncated: fit CTMC from realistic truncated snapshots; "
            "higher-order: fit an order-k augmented-state CTMC; "
            "temporal: fit a piecewise time-varying timeout CTMC"
        ),
    )
    parser.add_argument("--max-journeys",    type=int,   default=100_000)
    parser.add_argument("--n-clusters",      type=int,   default=3)
    parser.add_argument(
        "--markov-order",
        type=int,
        default=2,
        choices=[2, 3],
        help="History length for higher-order augmented-state CTMC.",
    )
    parser.add_argument(
        "--use-spectral-clustering",
        action="store_true",
        help="Use spectral clustering instead of KMeans inside ClusteredCTMC.",
    )
    parser.add_argument("--n-tune-trials",   type=int,   default=20)
    parser.add_argument("--k-min",           type=int,   default=2)
    parser.add_argument("--k-max",           type=int,   default=12)
    parser.add_argument(
        "--models-dir",
        type=_path_arg,
        default=DEFAULT_MODELS_DIR,
        help="Directory for cached .joblib models.",
    )
    parser.add_argument(
        "--output-dir",
        type=_path_arg,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSVs.",
    )
    parser.add_argument(
        "--test-events",
        type=_path_arg,
        default=DEFAULT_TEST_EVENTS,
        help="Open journey event CSV.",
    )
    parser.add_argument(
        "--sample",
        type=_path_arg,
        default=DEFAULT_SAMPLE,
        help="Kaggle sample/template CSV.",
    )
    parser.add_argument(
        "--test-prevalence",
        type=float,
        default=None,
        help=(
            "Override the prevalence target used for prior-probability calibration. "
            "For open_journeys2, all-zero Brier 0.04180 implies --test-prevalence 0.04180."
        ),
    )
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

    print("Resolved CTMC pipeline paths:")
    print(f"  test_events: {args.test_events}")
    print(f"  sample:      {args.sample}")
    print(f"  output_dir:  {args.output_dir}")
    print(f"  models_dir:  {args.models_dir}")
    if args.test_prevalence is not None:
        print(f"  test_prev:   {args.test_prevalence:.5f}")

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
            test_prevalence=args.test_prevalence,
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
            test_prevalence=args.test_prevalence,
        )
        return

    if args.mode == "higher-order":
        run_higher_order_ctmc(
            test_events_path=args.test_events,
            sample_path=args.sample,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            order=args.markov_order,
            max_journeys=args.max_journeys,
            test_prevalence=args.test_prevalence,
        )
        return

    if args.mode == "temporal":
        run_temporal_ctmc(
            test_events_path=args.test_events,
            sample_path=args.sample,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            max_journeys=args.max_journeys,
            test_prevalence=args.test_prevalence,
        )
        return

    if args.mode == "suite":
        run_suite(
            test_events_path=args.test_events,
            sample_path=args.sample,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            n_clusters=n_clusters,
            max_journeys=args.max_journeys,
            test_prevalence=args.test_prevalence,
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
        and _cached_models_are_timeout_aware(args.models_dir)
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
            test_prevalence=args.test_prevalence,
        )


if __name__ == "__main__":
    main()
