"""Plot CTMC structure and performance diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_generator_heatmap(global_ctmc, output_path: Path, title: str = "Global CTMC generator rates") -> None:
    q = global_ctmc.Q_.copy()
    for state in q.index:
        q.loc[state, state] = 0.0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log1p(q.to_numpy()), aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log(1 + rate)")
    ax.set_title(title)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    ax.set_xticks(range(len(q.columns)))
    ax.set_xticklabels(q.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(q.index)))
    ax.set_yticklabels(q.index, fontsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_transition_graph(global_ctmc, output_path: Path, n_edges: int = 25) -> None:
    edges = global_ctmc.top_rates(n_edges).copy()
    if edges.empty:
        return

    states = sorted(set(edges["from_state"]).union(set(edges["to_state"])))
    angles = np.linspace(0, 2 * np.pi, len(states), endpoint=False)
    positions = {
        state: (np.cos(angle), np.sin(angle))
        for state, angle in zip(states, angles)
    }

    fig, ax = plt.subplots(figsize=(9, 9))
    max_rate = edges["rate"].max()
    for _, row in edges.iterrows():
        x1, y1 = positions[row["from_state"]]
        x2, y2 = positions[row["to_state"]]
        width = 0.5 + 4.0 * row["rate"] / max_rate
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": "#4c566a",
                "alpha": 0.45,
                "lw": width,
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )

    for state, (x, y) in positions.items():
        ax.scatter([x], [y], s=850, color="#88c0d0", edgecolor="#2e3440", zorder=3)
        ax.text(x, y, str(state), ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title(f"Top {len(edges)} CTMC transition rates")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_absorption_by_state(
    global_ctmc,
    output_path: Path,
    success_state: int = 28,
    horizon_days: float = 60,
    top_n: int = 20,
) -> pd.DataFrame:
    horizon_seconds = horizon_days * 24 * 60 * 60
    probs = global_ctmc.absorption_probability(
        global_ctmc.states_,
        success_state=success_state,
        horizon_seconds=horizon_seconds,
    )
    df = (
        pd.DataFrame({"state": global_ctmc.states_, "success_probability": probs})
        .sort_values("success_probability", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["state"].astype(str), df["success_probability"], color="#5e81ac")
    ax.set_title(f"P(order_shipped within {horizon_days:g} days) by current state")
    ax.set_xlabel("current state")
    ax.set_ylabel("success probability")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return df


def plot_calibration(y_true, y_prob, output_path: Path, n_bins: int = 10, title: str = "Calibration plot") -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 0.0, 1.0)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True)
    bin_ids = np.clip(bin_ids, 1, n_bins)

    rows = []
    for bin_id in range(1, n_bins + 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_id,
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="perfect calibration")
    ax.plot(df["mean_predicted"], df["observed_rate"], marker="o", color="#bf616a", label="model")
    ax.set_title(title)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed success rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return df


def plot_predictions_by_state(
    models: dict,
    output_path: Path,
    success_state: int = 28,
    horizon_days: float = 60,
) -> pd.DataFrame:
    """Grouped bar chart of absorption probability per current state for multiple CTMC models."""
    horizon_seconds = horizon_days * 24 * 60 * 60

    all_states = sorted(
        set().union(*[set(m.states_) for m in models.values()])
    )
    all_states = [s for s in all_states if s != success_state]

    rows = []
    for name, model in models.items():
        probs = model.absorption_probability(
            all_states, success_state=success_state, horizon_seconds=horizon_seconds
        )
        for state, prob in zip(all_states, probs):
            rows.append({"state": state, "model": name, "success_probability": prob})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="state", columns="model", values="success_probability").fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot.bar(ax=ax, width=0.75)
    ax.set_title(f"P(order_shipped within {horizon_days:g} days) by current state — model comparison")
    ax.set_xlabel("current state")
    ax.set_ylabel("success probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return df


def plot_metric_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    cols = [c for c in ["roc_auc", "average_precision", "log_loss", "brier_score"] if c in comparison.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ordered = comparison.sort_values(col, ascending=col in {"log_loss", "brier_score"})
        ax.barh(ordered["model"], ordered[col], color="#a3be8c")
        ax.set_title(col)
        ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
