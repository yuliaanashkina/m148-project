"""
Advanced CTMC methods for the M148 capstone.

LaplaceGlobalCTMC  — Dirichlet-smoothed generator matrix (tames sparse-state noise)
CalibratedCTMC     — Post-hoc isotonic calibration wrapper (directly targets brier score)

These extend the baseline GlobalCTMC / ClusteredCTMC in ctmc.py and are compared
against them in notebooks/benchmarks/ctmc_exploration.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ctmc import GlobalCTMC, CTMCData
except ImportError:
    from ctmc import GlobalCTMC, CTMCData


# ---------------------------------------------------------------------------
# 1. LaplaceGlobalCTMC
# ---------------------------------------------------------------------------

class LaplaceGlobalCTMC(GlobalCTMC):
    """
    GlobalCTMC with Laplace (add-alpha) smoothing on transition counts.

    Adds alpha pseudo-counts to every N_ij before computing rates, which
    regularises the generator matrix for rarely-observed state pairs and
    prevents zero-probability transitions from dominating the absorption
    probability estimate.

    alpha=0 reproduces the unsmoothed GlobalCTMC exactly.
    alpha=1 is standard Laplace smoothing.
    alpha=0.5 (default) is the Jeffreys prior.
    """

    def __init__(self, alpha: float = 0.5, min_time: float = 1e-9) -> None:
        super().__init__(min_time=min_time)
        self.alpha = alpha

    def fit(self, transitions: pd.DataFrame) -> "LaplaceGlobalCTMC":
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
        # Laplace smoothing: add alpha to every off-diagonal cell.
        smoothed = counts.astype(float) + self.alpha
        np.fill_diagonal(smoothed.values, 0.0)

        time_in_state = transitions.groupby("state")["dt_seconds"].sum()
        time_in_state = time_in_state.reindex(states, fill_value=0.0).clip(lower=self.min_time)

        q_values = smoothed.div(time_in_state, axis=0).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(q_values, 0.0)
        q_values[np.diag_indices_from(q_values)] = -q_values.sum(axis=1)
        q = pd.DataFrame(q_values, index=states, columns=states)

        self.transition_counts_ = counts
        self.time_in_state_ = time_in_state
        self.Q_ = q
        return self


# ---------------------------------------------------------------------------
# 2. CalibratedCTMC
# ---------------------------------------------------------------------------

class CalibratedCTMC:
    """
    Post-hoc isotonic calibration wrapper for any CTMC predictor.

    Fits an isotonic regression mapping raw predicted probabilities to
    empirical success rates on a held-out calibration set.  Because isotonic
    regression minimises squared error, this directly reduces brier score —
    the Kaggle evaluation metric.

    Usage
    -----
    cal = CalibratedCTMC(base_model=global_ctmc)
    cal.fit_calibration(cal_features, cal_labels)
    probs = cal.predict(test_features)
    """

    def __init__(self, base_model: GlobalCTMC) -> None:
        self.base_model = base_model
        self.calibrator_ = None

    def _raw_probs(
        self,
        features: pd.DataFrame,
        success_state: int,
        horizon_seconds: float,
    ) -> np.ndarray:
        if hasattr(self.base_model, "predict_success_probability"):
            return self.base_model.predict_success_probability(
                features,
                success_state=success_state,
                horizon_seconds=horizon_seconds,
            )
        return self.base_model.absorption_probability(
            features["current_state"],
            success_state=success_state,
            horizon_seconds=horizon_seconds,
        )

    def fit_calibration(
        self,
        cal_features: pd.DataFrame,
        cal_labels: pd.Series | np.ndarray,
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> "CalibratedCTMC":
        from sklearn.isotonic import IsotonicRegression

        raw = self._raw_probs(cal_features, success_state, horizon_seconds)
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.calibrator_.fit(raw, np.asarray(cal_labels))
        return self

    def predict(
        self,
        features: pd.DataFrame,
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> np.ndarray:
        if self.calibrator_ is None:
            raise RuntimeError("Call fit_calibration() before predict().")
        raw = self._raw_probs(features, success_state, horizon_seconds)
        return np.clip(self.calibrator_.predict(raw), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Convenience: run all advanced models and return a comparison DataFrame
# ---------------------------------------------------------------------------

def compare_advanced_models(
    transitions: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    success_state: int = 28,
    horizon_seconds: float = 60 * 24 * 60 * 60,
    cal_frac: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit advanced CTMC variants and return a brier/AUC/logloss comparison table.

    Parameters
    ----------
    transitions : pd.DataFrame
        Output of CTMCData().transition_table().
    features : pd.DataFrame
        Journey-level feature table with 'current_state' and 'id' columns.
    labels : pd.Series
        Binary labels aligned to features (1 = successful journey).
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    idx_train, idx_cal = train_test_split(
        np.arange(len(features)), test_size=cal_frac, random_state=random_state, stratify=labels
    )
    feat_train = features.iloc[idx_train].reset_index(drop=True)
    feat_cal = features.iloc[idx_cal].reset_index(drop=True)
    lab_cal = labels.iloc[idx_cal].reset_index(drop=True)

    trans_train = transitions[transitions["id"].isin(feat_train["id"])]

    models: dict[str, np.ndarray] = {}

    laplace = LaplaceGlobalCTMC(alpha=0.5).fit(trans_train)
    models["laplace_global"] = laplace.absorption_probability(
        feat_cal["current_state"], success_state=success_state, horizon_seconds=horizon_seconds
    )

    base_global = GlobalCTMC().fit(trans_train)
    cal_global = CalibratedCTMC(base_global)
    cal_global.fit_calibration(feat_train, labels.iloc[idx_train], success_state, horizon_seconds)
    models["calibrated_global"] = cal_global.predict(feat_cal, success_state, horizon_seconds)

    cal_laplace = CalibratedCTMC(laplace)
    cal_laplace.fit_calibration(feat_train, labels.iloc[idx_train], success_state, horizon_seconds)
    models["calibrated_laplace"] = cal_laplace.predict(feat_cal, success_state, horizon_seconds)

    rows = []
    y = lab_cal.to_numpy()
    for name, probs in models.items():
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        rows.append({
            "model": name,
            "brier_score": brier_score_loss(y, probs),
            "log_loss": log_loss(y, probs, labels=[0, 1]),
            "roc_auc": roc_auc_score(y, probs),
            "average_precision": average_precision_score(y, probs),
        })

    return pd.DataFrame(rows).sort_values("brier_score")
