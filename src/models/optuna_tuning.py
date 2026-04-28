"""
Optuna tuning for capstone prediction models.

Design choices:
- Optimize against validation log loss because Kaggle submissions are
  probabilities.
- Use one stratified split for comparable trials.
- Tune the neural CTMC rate model with PyTorch + Adam/AdamW/RMSprop.
- Tune XGBoost, histogram gradient boosting, and random forest.
- Write all trial summaries and best parameters under results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import polars as pl


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


def split_xy(df: pd.DataFrame, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    y = df["label"].astype(int)
    x = df.drop(
        columns=["id", "label", "final_outcome", "remaining_time_to_success_seconds"],
        errors="ignore",
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    return x_train, x_valid, y_train, y_valid


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
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(imputer.fit_transform(x_train))
    x_valid_np = scaler.transform(imputer.transform(x_valid))
    return x_train_np.astype("float32"), x_valid_np.astype("float32")


def tune_xgboost(x_train, x_valid, y_train, y_valid, n_trials: int, random_state: int):
    from xgboost import XGBClassifier

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 150, 700),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 12.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_valid)[:, 1]
        trial.set_user_attr("metrics", score_probs(y_valid, probs))
        return score_probs(y_valid, probs)["log_loss"]

    return run_study("xgboost", objective, n_trials)


def tune_hist_gradient_boosting(x_train, x_valid, y_train, y_valid, n_trials: int, random_state: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def objective(trial: optuna.Trial) -> float:
        model = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 120),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_valid)[:, 1]
        trial.set_user_attr("metrics", score_probs(y_valid, probs))
        return score_probs(y_valid, probs)["log_loss"]

    return run_study("hist_gradient_boosting", objective, n_trials)


def tune_random_forest(x_train, x_valid, y_train, y_valid, n_trials: int, random_state: int):
    from sklearn.ensemble import RandomForestClassifier

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 4, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_valid)[:, 1]
        trial.set_user_attr("metrics", score_probs(y_valid, probs))
        return score_probs(y_valid, probs)["log_loss"]

    return run_study("random_forest", objective, n_trials)


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

        # Binary event probability at 60 days. This directly tunes the rate
        # network for the downstream Kaggle probability loss.
        loss_fn = nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_np[:, None]))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best_loss = np.inf
        patience = 5
        stalled = 0
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


def run_tuning(max_rows: int, n_trials: int, models: list[str], random_state: int = 42) -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_features(max_rows=max_rows)
    x_train, x_valid, y_train, y_valid = split_xy(df, random_state=random_state)

    studies = []
    if "xgboost" in models:
        studies.append(tune_xgboost(x_train, x_valid, y_train, y_valid, n_trials, random_state))
    if "hgb" in models or "hist_gradient_boosting" in models:
        studies.append(tune_hist_gradient_boosting(x_train, x_valid, y_train, y_valid, n_trials, random_state))
    if "rf" in models or "random_forest" in models:
        studies.append(tune_random_forest(x_train, x_valid, y_train, y_valid, n_trials, random_state))
    if "neural" in models or "neural_rate" in models:
        studies.append(tune_neural_rate(x_train, x_valid, y_train, y_valid, n_trials, random_state))

    for study in studies:
        save_trials(study)

    summary = pd.DataFrame([study_to_row(study) for study in studies]).sort_values("log_loss")
    summary.to_csv(RESULTS_DIR / "optuna_tuning_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune project models with Optuna.")
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "hgb", "rf", "neural"],
        help="Any of: xgboost hgb rf neural",
    )
    args = parser.parse_args()

    summary = run_tuning(max_rows=args.max_rows, n_trials=args.n_trials, models=args.models)
    print(summary)


if __name__ == "__main__":
    main()
