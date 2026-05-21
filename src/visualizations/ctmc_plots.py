"""Plot CTMC structure and performance diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _event_name_map(event_defs: pd.DataFrame | None = None) -> dict[int, str]:
    """Map event definition ids to readable names when definitions are available."""
    if event_defs is None or event_defs.empty:
        return {}

    id_col = "event_definition_id"
    name_col = "event_name"
    if id_col not in event_defs.columns or name_col not in event_defs.columns:
        return {}
    return {
        int(row[id_col]): str(row[name_col])
        for _, row in event_defs[[id_col, name_col]].dropna().iterrows()
    }


def _short_event_label(state: int, names: dict[int, str], max_chars: int = 22) -> str:
    if state == -1:
        return "60d inactivity\nfailure"
    label = names.get(int(state), str(state)).replace("_", " ")
    if len(label) > max_chars:
        label = label[: max_chars - 1] + "…"
    return f"{state}\n{label}"


def plot_aesthetic_state_space_graph(
    global_ctmc,
    output_path: Path,
    event_defs: pd.DataFrame | None = None,
    n_edges: int = 36,
    success_state: int = 28,
    title: str = "Global CTMC state-space graph",
) -> pd.DataFrame:
    """Draw a polished directed graph of the highest-rate global CTMC transitions.

    Node size reflects transition volume involving the state. Edge width and
    opacity reflect the fitted transition rate q_ij. The plot intentionally
    limits to the strongest transitions so the graph is legible in a notebook.
    """
    edges = global_ctmc.top_rates(n_edges).copy()
    if edges.empty:
        return edges

    names = _event_name_map(event_defs)
    states = sorted(
        set(getattr(global_ctmc, "states_", []))
        .union(set(edges["from_state"]))
        .union(set(edges["to_state"]))
    )

    try:
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(states)
        for _, row in edges.iterrows():
            graph.add_edge(int(row["from_state"]), int(row["to_state"]), weight=float(row["rate"]))
        pos = nx.spring_layout(graph, seed=7, k=1.15 / np.sqrt(max(len(states), 1)), weight="weight")
    except Exception:
        angles = np.linspace(0, 2 * np.pi, len(states), endpoint=False)
        pos = {state: (np.cos(angle), np.sin(angle)) for state, angle in zip(states, angles)}

    counts = getattr(global_ctmc, "transition_counts_", None)
    if counts is not None:
        volumes = {
            state: float(counts.loc[state].sum() + counts[state].sum())
            if state in counts.index and state in counts.columns
            else 1.0
            for state in states
        }
    else:
        volumes = {state: 1.0 for state in states}
    max_volume = max(volumes.values()) if volumes else 1.0

    fig, ax = plt.subplots(figsize=(13, 9), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    max_rate = float(edges["rate"].max())
    for _, row in edges.sort_values("rate").iterrows():
        source = int(row["from_state"])
        target = int(row["to_state"])
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        rate = float(row["rate"])
        width = 0.7 + 4.5 * np.sqrt(rate / max_rate)
        alpha = 0.18 + 0.55 * np.sqrt(rate / max_rate)
        color = "#64748b" if target != success_state else "#16a34a"
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "alpha": alpha,
                "lw": width,
                "shrinkA": 23,
                "shrinkB": 25,
                "connectionstyle": "arc3,rad=0.12",
                "mutation_scale": 14,
            },
            zorder=1,
        )

    for state in states:
        x, y = pos[state]
        volume_scale = np.sqrt(volumes.get(state, 1.0) / max_volume)
        size = 1_050 + 1_900 * volume_scale
        if state == success_state:
            face, edge, text = "#bbf7d0", "#15803d", "#14532d"
        elif state == -1:
            face, edge, text = "#fecaca", "#b91c1c", "#7f1d1d"
        else:
            face, edge, text = "#dbeafe", "#2563eb", "#0f172a"
        ax.scatter([x], [y], s=size, color=face, edgecolor=edge, linewidth=2.2, zorder=3)
        ax.text(
            x,
            y,
            _short_event_label(state, names),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="semibold",
            color=text,
            zorder=4,
        )

    ax.set_title(title, fontsize=18, fontweight="bold", color="#0f172a", pad=16)
    ax.text(
        0.5,
        -0.035,
        "Top fitted transition rates; green arrows point into order_shipped. Larger nodes appear in more observed transitions.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#475569",
    )
    ax.set_axis_off()
    ax.margins(0.18)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return edges


def plot_journey_success_probability_over_time(
    events: pd.DataFrame,
    global_ctmc,
    output_path: Path,
    event_defs: pd.DataFrame | None = None,
    journey_id=None,
    customer_states: set[int] | None = None,
    success_state: int = 28,
    horizon_days: float = 60,
    title: str | None = None,
) -> pd.DataFrame:
    """Plot one journey's global-CTMC success probability after each user action.

    The plotted value is P(order_shipped within the horizon | current state)
    immediately after the action has occurred. The stair-step is drawn with the
    action marker slightly before the vertical jump, so actions that trigger an
    increase are visually placed before the increase they cause.
    """
    if events.empty:
        raise ValueError("events is empty")

    names = _event_name_map(event_defs)
    df = events.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["ed_id"] = df["ed_id"].astype(int)

    if journey_id is None:
        if "id" not in df.columns:
            raise ValueError("events must contain an id column when journey_id is not provided")
        journey_id = df["id"].iloc[0]
    df = df[df["id"] == journey_id].sort_values(["event_timestamp", "ed_id"]).copy()
    if df.empty:
        raise ValueError(f"No events found for journey_id={journey_id!r}")

    if customer_states is not None:
        plot_df = df[df["ed_id"].isin(customer_states | {success_state})].copy()
    else:
        plot_df = df.copy()
    if plot_df.empty:
        raise ValueError(f"No plotted actions found for journey_id={journey_id!r}")

    horizon_seconds = horizon_days * 24 * 60 * 60
    plot_df["success_probability"] = global_ctmc.absorption_probability(
        plot_df["ed_id"].to_numpy(),
        success_state=success_state,
        horizon_seconds=horizon_seconds,
    )
    first_time = plot_df["event_timestamp"].iloc[0]
    plot_df["elapsed_days"] = (
        plot_df["event_timestamp"] - first_time
    ).dt.total_seconds() / 86400
    plot_df["event_label"] = plot_df["ed_id"].map(lambda x: names.get(int(x), str(int(x))))
    plot_df["delta_probability"] = plot_df["success_probability"].diff().fillna(0.0)

    x = plot_df["elapsed_days"].to_numpy(dtype=float)
    y = plot_df["success_probability"].to_numpy(dtype=float)
    min_gap = np.diff(np.unique(x)).min() if len(np.unique(x)) > 1 else 0.08
    epsilon = max(min_gap * 0.08, 0.01)

    xs = [x[0]]
    ys = [y[0]]
    prev = y[0]
    for xi, yi in zip(x[1:], y[1:]):
        # Action marker sits just before the model updates to the new state.
        xs.extend([max(x[0], xi - epsilon), xi, xi])
        ys.extend([prev, prev, yi])
        prev = yi
    if len(x) == 1:
        xs.append(x[0] + 0.05)
        ys.append(y[0])

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.plot(xs, ys, color="#2563eb", linewidth=3.0, solid_capstyle="round", zorder=2)
    ax.fill_between(xs, ys, 0, color="#93c5fd", alpha=0.18, zorder=1)

    colors = np.where(plot_df["delta_probability"] > 1e-6, "#16a34a", "#64748b")
    ax.scatter(
        np.maximum(x - epsilon, x[0]),
        y,
        s=90,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
    )
    for xi, yi, label, delta in zip(x, y, plot_df["event_label"], plot_df["delta_probability"]):
        label_text = str(label).replace("_", " ")
        if len(label_text) > 24:
            label_text = label_text[:23] + "…"
        ax.annotate(
            label_text,
            xy=(max(x[0], xi - epsilon), yi),
            xytext=(-4, 18 if delta >= 0 else -24),
            textcoords="offset points",
            ha="right",
            va="bottom" if delta >= 0 else "top",
            fontsize=8,
            color="#334155",
            rotation=28,
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "alpha": 0.65, "lw": 0.9},
        )

    ax.set_title(
        title or f"Global CTMC predicted success probability over time — journey {journey_id}",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )
    ax.set_xlabel("Days since first plotted user action")
    ax.set_ylabel(f"P(order_shipped within {horizon_days:g} days | current state)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.65, linewidth=0.8)
    ax.grid(axis="x", color="#e2e8f0", alpha=0.45, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94a3b8")
    ax.text(
        0.0,
        -0.20,
        "Markers are placed just before the vertical update: the action occurs first, then the global CTMC state changes.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#475569",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return plot_df[
        [
            "id",
            "event_timestamp",
            "elapsed_days",
            "ed_id",
            "event_label",
            "success_probability",
            "delta_probability",
        ]
    ].reset_index(drop=True)


def plotly_timeout_probability_over_time(
    events: pd.DataFrame,
    timeout_ctmc,
    event_defs: pd.DataFrame | None = None,
    journey_id=None,
    customer_states: set[int] | None = None,
    success_state: int = 28,
    horizon_days: float = 60,
    title: str | None = None,
):
    """Interactive Plotly journey path for the timeout-absorbing CTMC.

    The score is the timeout-aware probability of shipping before the absorbing
    inactivity-failure state, evaluated after each observed customer action.
    Each marker is labeled with both the state id and the action name from the
    event-definition key.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the interactive timeout probability plot. "
            "Install it with: python -m pip install plotly"
        ) from exc

    if events.empty:
        raise ValueError("events is empty")

    names = _event_name_map(event_defs)
    df = events.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    df["ed_id"] = df["ed_id"].astype(int)

    if journey_id is None:
        if "id" not in df.columns:
            raise ValueError("events must contain an id column when journey_id is not provided")
        journey_id = df["id"].iloc[0]

    df = df[df["id"] == journey_id].sort_values(["event_timestamp", "ed_id"]).copy()
    if df.empty:
        raise ValueError(f"No events found for journey_id={journey_id!r}")

    if customer_states is not None:
        df = df[df["ed_id"].isin(customer_states | {success_state})].copy()
    if df.empty:
        raise ValueError(f"No plotted actions found for journey_id={journey_id!r}")

    horizon_seconds = horizon_days * 24 * 60 * 60
    if hasattr(timeout_ctmc, "predict_success_probability"):
        probs = timeout_ctmc.predict_success_probability(
            df["ed_id"].to_numpy(),
            success_state=success_state,
            horizon_seconds=horizon_seconds,
        )
    else:
        probs = timeout_ctmc.absorption_probability(
            df["ed_id"].to_numpy(),
            success_state=success_state,
            horizon_seconds=horizon_seconds,
        )

    first_time = df["event_timestamp"].iloc[0]
    df["elapsed_days"] = (df["event_timestamp"] - first_time).dt.total_seconds() / 86400
    df["success_probability"] = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    df["action_name"] = df["ed_id"].map(lambda x: names.get(int(x), str(int(x))))
    df["state_action_label"] = df.apply(
        lambda row: f"{int(row['ed_id'])} — {str(row['action_name']).replace('_', ' ')}",
        axis=1,
    )
    df["delta_probability"] = df["success_probability"].diff().fillna(0.0)
    df["direction"] = np.where(
        df["delta_probability"] > 1e-6,
        "increase",
        np.where(df["delta_probability"] < -1e-6, "decrease", "unchanged"),
    )
    marker_colors = df["direction"].map({
        "increase": "#16a34a",
        "decrease": "#dc2626",
        "unchanged": "#64748b",
    }).to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_days"],
            y=df["success_probability"],
            mode="lines",
            line={
                "shape": "hv",
                "width": 4,
                "color": "#2563eb",
            },
            fill="tozeroy",
            fillcolor="rgba(147, 197, 253, 0.22)",
            name="timeout-aware success probability",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["elapsed_days"],
            y=df["success_probability"],
            mode="markers+text",
            marker={
                "size": 12,
                "color": marker_colors,
                "line": {"width": 1.5, "color": "white"},
            },
            text=df["ed_id"].astype(str),
            textposition="top center",
            customdata=np.stack(
                [
                    df["state_action_label"],
                    df["event_timestamp"].astype(str),
                    df["delta_probability"],
                    df["direction"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "time: %{customdata[1]}<br>"
                "days since first action: %{x:.2f}<br>"
                "P(success before timeout): %{y:.3f}<br>"
                "change from previous action: %{customdata[2]:+.3f}<br>"
                "direction: %{customdata[3]}"
                "<extra></extra>"
            ),
            name="user actions",
        )
    )

    # Add unobtrusive labels for every action below the curve. Hover gives the
    # full action name; these annotations keep the state/action key visible.
    for idx, row in df.iterrows():
        label = row["state_action_label"]
        if len(label) > 34:
            label = label[:33] + "…"
        fig.add_annotation(
            x=row["elapsed_days"],
            y=max(float(row["success_probability"]) - 0.08, 0.03),
            text=label,
            showarrow=False,
            textangle=-35,
            font={"size": 10, "color": "#334155"},
            align="right",
            opacity=0.86,
        )

    fig.update_layout(
        title={
            "text": title
            or f"Timeout-absorbing CTMC probability path — journey {journey_id}",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "color": "#0f172a"},
        },
        template="plotly_white",
        width=1100,
        height=620,
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        hovermode="closest",
        legend={"orientation": "h", "y": 1.06, "x": 0.0},
        margin={"l": 70, "r": 35, "t": 95, "b": 130},
        xaxis={
            "title": "Days since first plotted user action",
            "showgrid": True,
            "gridcolor": "#e2e8f0",
            "zeroline": False,
        },
        yaxis={
            "title": f"P(order_shipped before 60-day inactivity failure)",
            "range": [-0.03, 1.03],
            "tickformat": ".0%",
            "showgrid": True,
            "gridcolor": "#cbd5e1",
            "zeroline": False,
        },
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0,
        y=-0.23,
        text=(
            "Step changes occur after user actions. Green markers increase the timeout-aware "
            "success probability; red markers decrease it; gray markers leave it roughly unchanged."
        ),
        showarrow=False,
        align="left",
        font={"size": 12, "color": "#475569"},
    )
    return fig, df[
        [
            "id",
            "event_timestamp",
            "elapsed_days",
            "ed_id",
            "action_name",
            "state_action_label",
            "success_probability",
            "delta_probability",
            "direction",
        ]
    ].reset_index(drop=True)


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
