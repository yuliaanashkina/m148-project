"""
Advanced CTMC methods for the M148 capstone.

LaplaceGlobalCTMC      — Dirichlet-smoothed generator matrix (tames sparse-state noise)
WeibullSemiMarkovCTMC  — Semi-Markov model with per-state Weibull holding times;
                         absorption probability computed via Monte Carlo simulation
CalibratedCTMC         — Post-hoc isotonic calibration wrapper (directly targets brier score)

These extend the baseline GlobalCTMC / ClusteredCTMC in ctmc.py and are compared
against them in notebooks/benchmarks/ctmc_exploration.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
# 2. WeibullSemiMarkovCTMC
# ---------------------------------------------------------------------------

class WeibullSemiMarkovCTMC:
    """
    Semi-Markov model with per-state Weibull holding times.

    A standard CTMC constrains holding times to be exponential (memoryless).
    Real customer dwell times are typically lognormal or Weibull (shape != 1).
    This model:
      1. Estimates the embedded Markov chain P_ij = N_ij / N_i from observed
         transitions (direction of travel, independent of time).
      2. Fits a Weibull distribution to holding times per state.
      3. Estimates absorption probability via Monte Carlo simulation.

    States with fewer than min_obs observations fall back to an exponential
    (Weibull shape=1) parameterised by the global mean holding time.
    """

    def __init__(
        self,
        min_obs: int = 10,
        n_sims: int = 300,
        random_state: int = 42,
    ) -> None:
        self.min_obs = min_obs
        self.n_sims = n_sims
        self.random_state = random_state
        self.states_: list[int] = []
        self.embedded_P_: pd.DataFrame | None = None
        self.weibull_params_: dict[int, tuple[float, float]] = {}  # state -> (shape, scale)

    def fit(self, transitions: pd.DataFrame) -> "WeibullSemiMarkovCTMC":
        from scipy.stats import weibull_min

        df = transitions.copy()
        df = df[df["state"] != df["next_state"]]

        states = sorted(
            set(df["state"].astype(int)).union(set(df["next_state"].astype(int)))
        )
        self.states_ = states

        # Embedded Markov chain: row-normalised count matrix.
        counts = pd.crosstab(df["state"], df["next_state"])
        counts = counts.reindex(index=states, columns=states, fill_value=0)
        row_sums = counts.sum(axis=1).replace(0, 1)
        self.embedded_P_ = counts.div(row_sums, axis=0).astype(float)

        # Per-state Weibull fit on holding times.
        global_mean = float(df["dt_seconds"].mean()) or 1.0
        for state in states:
            times = df.loc[df["state"] == state, "dt_seconds"].to_numpy()
            times = times[times > 0]
            if len(times) >= self.min_obs:
                try:
                    shape, _, scale = weibull_min.fit(times, floc=0)
                    self.weibull_params_[state] = (float(shape), float(scale))
                except Exception:
                    self.weibull_params_[state] = (1.0, global_mean)
            else:
                self.weibull_params_[state] = (1.0, global_mean)

        return self

    def _simulate_one(
        self,
        start_state: int,
        success_state: int,
        horizon: float,
        rng: np.random.Generator,
    ) -> bool:
        from scipy.stats import weibull_min

        state = start_state
        t = 0.0
        while True:
            if state == success_state:
                return True
            shape, scale = self.weibull_params_.get(state, (1.0, 1.0))
            dt = float(weibull_min.rvs(shape, scale=scale, loc=0, random_state=rng))
            t += dt
            if t > horizon:
                return False
            # Sample next state from embedded chain.
            row = self.embedded_P_.loc[state] if state in self.embedded_P_.index else None
            if row is None or row.sum() == 0:
                return False
            state = int(rng.choice(self.states_, p=row.to_numpy()))

    def absorption_probability(
        self,
        current_states: Iterable[int],
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> np.ndarray:
        if self.embedded_P_ is None:
            raise RuntimeError("Call fit() before predicting absorption probabilities.")

        rng = np.random.default_rng(self.random_state)
        probs = []
        for state in current_states:
            hits = sum(
                self._simulate_one(int(state), success_state, horizon_seconds, rng)
                for _ in range(self.n_sims)
            )
            probs.append(hits / self.n_sims)
        return np.clip(np.asarray(probs), 0.0, 1.0)

    def predict_success_probability(
        self,
        features: pd.DataFrame,
        success_state: int = 28,
        horizon_seconds: float = 60 * 24 * 60 * 60,
    ) -> np.ndarray:
        return self.absorption_probability(
            features["current_state"],
            success_state=success_state,
            horizon_seconds=horizon_seconds,
        )


# ---------------------------------------------------------------------------
# 3. CalibratedCTMC
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

    def __init__(self, base_model: GlobalCTMC | WeibullSemiMarkovCTMC) -> None:
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
    weibull_n_sims: int = 300,
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

    # Split for calibration.
    idx_train, idx_cal = train_test_split(
        np.arange(len(features)), test_size=cal_frac, random_state=random_state, stratify=labels
    )
    feat_train = features.iloc[idx_train].reset_index(drop=True)
    feat_cal = features.iloc[idx_cal].reset_index(drop=True)
    lab_cal = labels.iloc[idx_cal].reset_index(drop=True)

    trans_train = transitions[transitions["id"].isin(feat_train["id"])]

    models: dict[str, np.ndarray] = {}

    # Laplace-smoothed global CTMC
    laplace = LaplaceGlobalCTMC(alpha=0.5).fit(trans_train)
    models["laplace_global"] = laplace.absorption_probability(
        feat_cal["current_state"], success_state=success_state, horizon_seconds=horizon_seconds
    )

    # Calibrated global CTMC
    base_global = GlobalCTMC().fit(trans_train)
    cal_global = CalibratedCTMC(base_global)
    cal_global.fit_calibration(feat_train, labels.iloc[idx_train], success_state, horizon_seconds)
    models["calibrated_global"] = cal_global.predict(feat_cal, success_state, horizon_seconds)

    # Calibrated Laplace CTMC
    cal_laplace = CalibratedCTMC(laplace)
    cal_laplace.fit_calibration(feat_train, labels.iloc[idx_train], success_state, horizon_seconds)
    models["calibrated_laplace"] = cal_laplace.predict(feat_cal, success_state, horizon_seconds)

    # Weibull semi-Markov (Monte Carlo — slower)
    print("Fitting WeibullSemiMarkovCTMC (Monte Carlo, may take a moment)...")
    weibull = WeibullSemiMarkovCTMC(n_sims=weibull_n_sims, random_state=random_state).fit(trans_train)
    models["weibull_semi_markov"] = weibull.absorption_probability(
        feat_cal["current_state"], success_state=success_state, horizon_seconds=horizon_seconds
    )

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
