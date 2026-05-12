"""
Optuna tuning for capstone prediction models.

Design choices:
- Optimize against mean CV log loss because Kaggle submissions are probabilities.
- Use StratifiedKFold CV so every trial sees the same variance from all folds.
- Tune XGBoost, histogram gradient boosting, and random forest.
- For CTMC clustering, use KFold over journey IDs with features precomputed once.
- Write all trial summaries and best parameters under results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
HORIZON_SECONDS = 60 * 24 * 60 * 60


def load_features(max_rows: int | None = 100_000) -> pd.DataFrame:
    try:
        from .ctmc import CTMCData
    except ImportError:
        from ctmc import CTMCData

    df = CTMCData().load_neural_rate_training_features(max_rows=max_rows)
    return df.drop(columns=["prefix_actions"], errors="ignore")


def score_probs(y_true, probs) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    probs = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
    }


def preprocess_arrays(x_train: pd.DataFrame, x_valid: pd.DataFrame):
    """Impute + scale; used by tune_neural_rate."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(imputer.fit_transform(x_train))
    x_valid_np = scaler.transform(imputer.transform(x_valid))
    return x_train_np.astype("float32"), x_valid_np.astype("float32")


# ---------------------------------------------------------------------------
# Tabular tuners — each takes raw (X_raw, y) and runs StratifiedKFold inside
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_raw: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    random_state: int,
    n_folds: int = 5,
    device: str = "cpu",
) -> optuna.Study:
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold

    X_np = X_raw.to_numpy(dtype="float64")
    y_np = y.to_numpy(dtype=int)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "device": device,
            "seed": random_state,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        }

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_losses = []

        for tr_idx, val_idx in cv.split(X_np, y_np):
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_np[tr_idx]).astype("float32")
            X_val = imputer.transform(X_np[val_idx]).astype("float32")

            dtrain = xgb.DMatrix(X_tr, label=y_np[tr_idx])
            dvalid = xgb.DMatrix(X_val, label=y_np[val_idx])

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=3000,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )
            best_iter = int(booster.best_iteration)
            probs = booster.predict(dvalid, iteration_range=(0, best_iter + 1))
            fold_losses.append(score_probs(y_np[val_idx], probs)["log_loss"])

        mean_loss = float(np.mean(fold_losses))
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("metrics", {"log_loss": mean_loss})
        return mean_loss

    return run_study("xgboost", objective, n_trials)


def tune_hist_gradient_boosting(
    X_raw: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    random_state: int,
    n_folds: int = 5,
) -> optuna.Study:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold

    X_np = X_raw.to_numpy(dtype="float64")
    y_np = y.to_numpy(dtype=int)

    def objective(trial: optuna.Trial) -> float:
        hgb_params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 120),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            random_state=random_state,
        )

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_losses = []

        for tr_idx, val_idx in cv.split(X_np, y_np):
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_np[tr_idx])
            X_val = imputer.transform(X_np[val_idx])

            model = HistGradientBoostingClassifier(**hgb_params)
            model.fit(X_tr, y_np[tr_idx])
            probs = model.predict_proba(X_val)[:, 1]
            fold_losses.append(score_probs(y_np[val_idx], probs)["log_loss"])

        mean_loss = float(np.mean(fold_losses))
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("metrics", {"log_loss": mean_loss})
        return mean_loss

    return run_study("hist_gradient_boosting", objective, n_trials)


def tune_random_forest(
    X_raw: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    random_state: int,
    n_folds: int = 5,
) -> optuna.Study:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold

    X_np = X_raw.to_numpy(dtype="float64")
    y_np = y.to_numpy(dtype=int)

    def objective(trial: optuna.Trial) -> float:
        rf_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 4, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            random_state=random_state,
            n_jobs=-1,
        )

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_losses = []

        for tr_idx, val_idx in cv.split(X_np, y_np):
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X_np[tr_idx])
            X_val = imputer.transform(X_np[val_idx])

            model = RandomForestClassifier(**rf_params)
            model.fit(X_tr, y_np[tr_idx])
            probs = model.predict_proba(X_val)[:, 1]
            fold_losses.append(score_probs(y_np[val_idx], probs)["log_loss"])

        mean_loss = float(np.mean(fold_losses))
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("metrics", {"log_loss": mean_loss})
        return mean_loss

    return run_study("random_forest", objective, n_trials)


# ---------------------------------------------------------------------------
# Neural rate tuner (legacy — not in default model list)
# ---------------------------------------------------------------------------

def make_torch_model(input_dim: int, layers: list[int], dropout: float):
    import torch
    from torch import nn

    modules: list[nn.Module] = []
    prev = input_dim
    for width in layers:
        modules.append(nn.Linear(prev, width))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = width
    modules.append(nn.Linear(prev, 1))
    return nn.Sequential(*modules)


def tune_neural_rate(x_train, x_valid, y_train, y_valid, n_trials: int, random_state: int):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    x_train_np, x_valid_np = preprocess_arrays(x_train, x_valid)
    y_train_np = y_train.to_numpy(dtype="float32")
    y_valid_np = y_valid.to_numpy(dtype="float32")

    torch.manual_seed(random_state)

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        layers = [width for _ in range(n_layers)]
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam", "rmsprop"])
        epochs = trial.suggest_int("epochs", 8, 40)

        model = make_torch_model(x_train_np.shape[1], layers, dropout)
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_np[:, None]))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best_loss = np.inf
        patience = 5
        stalled = 0
        probs = np.zeros(len(y_valid_np))
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(x_valid_np)).numpy().ravel()
            probs = 1.0 / (1.0 + np.exp(-logits))
            valid_loss = score_probs(y_valid_np, probs)["log_loss"]
            if valid_loss + 1e-5 < best_loss:
                best_loss = valid_loss
                stalled = 0
            else:
                stalled += 1
            if stalled >= patience:
                break

        trial.set_user_attr("metrics", score_probs(y_valid_np, probs))
        return best_loss

    return run_study("neural_rate_torch", objective, n_trials)


# ---------------------------------------------------------------------------
# CTMC clustering k-tuner — features precomputed once, KFold over journey IDs
# ---------------------------------------------------------------------------

def tune_ctmc_clustering(
    max_journeys: int = 100_000,
    n_trials: int = 20,
    k_min: int = 2,
    k_max: int = 12,
    random_state: int = 42,
    n_folds: int = 5,
) -> optuna.Study:
    """Optimize n_clusters for ClusteredCTMC against mean CV brier score.

    Journey features are computed once before the Optuna loop; each trial only
    does KMeans + k GlobalCTMC fits per fold, keeping per-trial time low.
    """
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from .ctmc import CTMCData, JourneyFeatureBuilder, GlobalCTMC
    except ImportError:
        from ctmc import CTMCData, JourneyFeatureBuilder, GlobalCTMC

    data = CTMCData()
    print("  Loading transitions...")
    transitions = data.transition_table(max_journeys=max_journeys)
    labels = data.load_binary_labels().set_index("id")["label"]

    # Precompute features once on all journeys — fit_transform on full set keeps
    # top_actions_ stable; cross-contamination is negligible for aggregate stats.
    print("  Building journey features (runs once)...")
    builder = JourneyFeatureBuilder()
    feats_all = builder.fit_transform(transitions)

    # Identify clustering columns (mirrors ClusteredCTMC logic).
    prop_cols  = [c for c in feats_all.columns if c.startswith("action_prop_")]
    state_cols = [c for c in ["first_state", "current_state"] if c in feats_all.columns]
    cluster_cols = prop_cols + state_cols

    # Journey IDs and their binary labels for stratified splitting.
    all_ids = feats_all["id"].to_numpy()
    all_id_labels = labels.reindex(all_ids).fillna(0).to_numpy(dtype=int)

    # Index feats_all by id for fast fold selection.
    feats_indexed = feats_all.set_index("id")

    def objective(trial: optuna.Trial) -> float:
        k = trial.suggest_int("n_clusters", k_min, k_max)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores = []

        for tr_idx, val_idx in cv.split(all_ids, all_id_labels):
            train_ids = all_ids[tr_idx]
            val_ids   = all_ids[val_idx]

            train_ids_set = set(train_ids.tolist())
            val_ids_set   = set(val_ids.tolist())

            trans_train = transitions[transitions["id"].isin(train_ids_set)]
            feats_train = feats_indexed.loc[feats_indexed.index.isin(train_ids_set)].reset_index()
            feats_val   = feats_indexed.loc[feats_indexed.index.isin(val_ids_set)].reset_index()

            x_train = feats_train.drop(columns=["id"])
            x_val   = feats_val.drop(columns=["id"])

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("cluster", KMeans(n_clusters=k, random_state=random_state, n_init=10)),
            ])
            cluster_labels_train = pipeline.fit_predict(x_train[cluster_cols])

            # Fit one GlobalCTMC per cluster.
            assignments = feats_train[["id"]].copy()
            assignments["cluster"] = cluster_labels_train
            merged = trans_train.merge(assignments, on="id", how="inner")
            global_fallback = GlobalCTMC().fit(trans_train)
            cluster_models: dict[int, GlobalCTMC] = {}
            for cid, g in merged.groupby("cluster"):
                cluster_models[int(cid)] = GlobalCTMC().fit(g)

            # Predict on val.
            for col in cluster_cols:
                if col not in x_val.columns:
                    x_val[col] = 0
            cluster_labels_val = pipeline.predict(x_val[cluster_cols])

            probs = np.zeros(len(feats_val))
            for cid in sorted(set(cluster_labels_val)):
                mask  = cluster_labels_val == cid
                model = cluster_models.get(int(cid), global_fallback)
                probs[mask] = model.absorption_probability(
                    feats_val.loc[mask, "current_state"]
                )

            val_labels = labels.reindex(feats_val["id"]).fillna(0).to_numpy()
            fold_scores.append(float(brier_score_loss(val_labels, np.clip(probs, 0.0, 1.0))))

        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("brier_score", mean_score)
        print(f"    trial {trial.number:>3}  k={k}  brier={mean_score:.5f}  folds={[f'{s:.5f}' for s in fold_scores]}")
        return mean_score

    study = run_study("ctmc_clustering_k", objective, n_trials)
    best_k = study.best_params["n_clusters"]
    print(f"Best n_clusters: {best_k}  (brier={study.best_value:.5f})")
    return study


# ---------------------------------------------------------------------------
# Time-stratified CTMC bin-edge tuner
# ---------------------------------------------------------------------------

def tune_time_stratified_ctmc(
    max_journeys: int = 100_000,
    n_trials: int = 20,
    random_state: int = 42,
    n_folds: int = 5,
) -> optuna.Study:
    """Optimize bin-edge boundaries for TimeStratifiedCTMC via CV brier score.

    Parameterization: edge1_hours (log-scale) and delta_hours (log-scale), so
    edge2 = edge1 + delta. This guarantees edge2 > edge1 without constraints.
    Features are precomputed once; fitting is cheap (3 GlobalCTMC fits per fold).
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold

    try:
        from .ctmc import CTMCData, JourneyFeatureBuilder, TimeStratifiedCTMC
    except ImportError:
        from ctmc import CTMCData, JourneyFeatureBuilder, TimeStratifiedCTMC

    data = CTMCData()
    print("  Loading transitions...")
    transitions = data.transition_table(max_journeys=max_journeys)
    labels = data.load_binary_labels().set_index("id")["label"]

    print("  Building journey features (runs once)...")
    builder = JourneyFeatureBuilder()
    feats_all = builder.fit_transform(transitions)

    all_ids = feats_all["id"].to_numpy()
    all_id_labels = labels.reindex(all_ids).fillna(0).to_numpy(dtype=int)
    feats_indexed = feats_all.set_index("id")

    def objective(trial: optuna.Trial) -> float:
        edge1_hours = trial.suggest_float("edge1_hours", 1.0, 72.0, log=True)
        delta_hours = trial.suggest_float("delta_hours", 12.0, 720.0, log=True)
        edge2_hours = edge1_hours + delta_hours
        bins = (edge1_hours * 3600.0, edge2_hours * 3600.0)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores = []

        for tr_idx, val_idx in cv.split(all_ids, all_id_labels):
            train_ids = set(all_ids[tr_idx].tolist())
            val_ids   = set(all_ids[val_idx].tolist())

            trans_train = transitions[transitions["id"].isin(train_ids)]
            feats_val   = feats_indexed.loc[feats_indexed.index.isin(val_ids)].reset_index()

            model = TimeStratifiedCTMC(bin_edges_seconds=bins)
            model.fit(trans_train)

            probs = model.predict_success_probability(feats_val)
            val_labels = labels.reindex(feats_val["id"]).fillna(0).to_numpy()
            fold_scores.append(float(brier_score_loss(val_labels, np.clip(probs, 0.0, 1.0))))

        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("brier_score", mean_score)
        trial.set_user_attr("edge2_hours", round(edge2_hours, 2))
        print(
            f"    trial {trial.number:>3}  "
            f"edges=[{edge1_hours:.1f}h, {edge2_hours:.1f}h]  "
            f"brier={mean_score:.5f}"
        )
        return mean_score

    study = run_study("time_stratified_ctmc", objective, n_trials)
    best = study.best_trial
    e1 = best.params["edge1_hours"]
    e2 = e1 + best.params["delta_hours"]
    print(f"Best edges: [{e1:.1f}h, {e2/24:.2f}d]  (brier={study.best_value:.5f})")
    return study


# ---------------------------------------------------------------------------
# Study helpers
# ---------------------------------------------------------------------------

def run_study(name: str, objective, n_trials: int) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", study_name=name, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def study_to_row(study: optuna.Study) -> dict:
    best = study.best_trial
    row = {
        "model": study.study_name,
        "best_value": best.value,
        **best.user_attrs.get("metrics", {}),
        "best_params": json.dumps(best.params, sort_keys=True),
    }
    return row


def save_trials(study: optuna.Study) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
            **trial.params,
            **trial.user_attrs.get("metrics", {}),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"optuna_{study.study_name}_trials.csv", index=False)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_tuning(
    max_rows: int,
    n_trials: int,
    models: list[str],
    random_state: int = 42,
    device: str = "cpu",
    n_folds: int = 5,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)

    tabular_needed = any(m in models for m in ("xgboost", "hgb", "hist_gradient_boosting", "rf", "random_forest"))
    if tabular_needed:
        df = load_features(max_rows=max_rows)
        y = df["label"].astype(int)
        X_raw = df.drop(
            columns=["id", "label", "final_outcome", "remaining_time_to_success_seconds"],
            errors="ignore",
        )

    studies = []
    if "xgboost" in models:
        studies.append(tune_xgboost(X_raw, y, n_trials, random_state, n_folds=n_folds, device=device))
    if "hgb" in models or "hist_gradient_boosting" in models:
        studies.append(tune_hist_gradient_boosting(X_raw, y, n_trials, random_state, n_folds=n_folds))
    if "rf" in models or "random_forest" in models:
        studies.append(tune_random_forest(X_raw, y, n_trials, random_state, n_folds=n_folds))
    if "ctmc_k" in models:
        studies.append(tune_ctmc_clustering(
            max_journeys=max_rows, n_trials=n_trials, random_state=random_state, n_folds=n_folds,
        ))
    if "time_stratified" in models:
        studies.append(tune_time_stratified_ctmc(
            max_journeys=max_rows, n_trials=n_trials, random_state=random_state, n_folds=n_folds,
        ))

    for study in studies:
        save_trials(study)

    summary = pd.DataFrame([study_to_row(study) for study in studies])
    if not summary.empty and "log_loss" in summary.columns:
        summary = summary.sort_values("log_loss")
    summary.to_csv(RESULTS_DIR / "optuna_tuning_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune project models with Optuna.")
    parser.add_argument("--max-rows",  type=int, default=100_000)
    parser.add_argument("--n-trials",  type=int, default=20)
    parser.add_argument("--n-folds",   type=int, default=5, help="StratifiedKFold splits per trial")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "hgb", "rf"],
        help="Any of: xgboost hgb rf ctmc_k time_stratified",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="XGBoost device: 'cpu' or 'cuda'",
    )
    args = parser.parse_args()

    summary = run_tuning(
        max_rows=args.max_rows,
        n_trials=args.n_trials,
        models=args.models,
        device=args.device,
        n_folds=args.n_folds,
    )
    print(summary)


if __name__ == "__main__":
    main()
