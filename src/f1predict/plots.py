"""Charts for the notebooks, README and app.

One visual system across the project: a validated colour-blind-safe categorical palette assigned
in fixed order, thin marks, hairline solid grids, one y-axis per chart, and text in ink colours
(never in series colours). Where a chart has many entities we highlight the ones that matter and
grey the rest instead of cycling through more hues.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from f1predict.config import FIGURES
from f1predict.features import FEATURE_LABELS

# Categorical slots, fixed order (light surface).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#c3c2b7"  # de-emphasis grey for "everything else"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def save(fig: plt.Figure, name: str, folder: Path = FIGURES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


# --------------------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------------------


def podium_rate_by_grid(df: pd.DataFrame, max_grid: int = 20) -> plt.Figure:
    rate = (
        df[df["grid"].between(1, max_grid)]
        .groupby("grid")["podium"]
        .mean()
        .mul(100)
        .reindex(range(1, max_grid + 1), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(rate.index, rate.values, width=0.6, color=PALETTE[0])
    for g in (1, 2, 3):
        ax.annotate(
            f"{rate[g]:.0f}%",
            (g, rate[g]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xticks(range(1, max_grid + 1))
    ax.set_xlabel("Starting grid position")
    ax.set_ylabel("Podium finishes (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Starting position matters - but not completely", pad=22)
    seasons = f"{df['season'].min()}-{df['season'].max()}"
    _subtitle(ax, f"Share of starts from each grid slot that ended on the podium, {seasons}")
    return fig


def win_share_by_team(df: pd.DataFrame, top: int = 4) -> plt.Figure:
    wins = df[df["win"] == 1]
    lineage_name = (
        df.sort_values("date").groupby("team_lineage")["constructor_name"].last().to_dict()
    )
    leaders = wins["team_lineage"].value_counts().index[:top].tolist()
    wins = wins.assign(team=wins["team_lineage"].where(wins["team_lineage"].isin(leaders), "other"))
    table = (
        wins.groupby(["season", "team"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=leaders + ["other"], fill_value=0)
    )
    share = table.div(table.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(share))
    colors = PALETTE[:top] + [OTHER]
    for team, color in zip(share.columns, colors, strict=True):
        label = "Other teams" if team == "other" else lineage_name.get(team, team)
        ax.bar(
            share.index,
            share[team],
            bottom=bottom,
            width=0.6,
            color=color,
            label=label,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        bottom += share[team].to_numpy()
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of race wins (%)")
    ax.set_xticks(share.index)
    ax.grid(axis="x", visible=False)
    ax.legend(
        ncol=top + 1,
        loc="upper left",
        bbox_to_anchor=(0, -0.08),
        fontsize=9,
        handlelength=1,
        columnspacing=1.2,
    )
    ax.set_title("Eras of dominance", pad=22)
    _subtitle(ax, "Race wins per season by team (current team names; rebrands grouped)")
    return fig


# --------------------------------------------------------------------------------------
# Model results
# --------------------------------------------------------------------------------------


def model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Small multiples of the pooled ('All') metrics; each panel highlights its best model."""
    overall = metrics[metrics["season"] == "All"].set_index("model")
    panels = [
        ("roc_auc", "Podium ROC-AUC", "{:.3f}"),
        ("top3_hit", "Podium drivers found (top-3 hit rate)", "{:.0%}"),
        ("winner_acc", "Winner picked correctly", "{:.0%}"),
    ]
    order = list(overall.index)[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (col, title, fmt) in zip(axes, panels, strict=True):
        vals = overall.loc[order, col]
        colors = [PALETTE[0] if v == vals.max() else OTHER for v in vals]
        ax.barh(order, vals, height=0.55, color=colors)
        lo = 0.5 if col == "roc_auc" else 0
        ax.set_xlim(lo, max(1.0, vals.max() * 1.1) if col != "roc_auc" else 1.0)
        for y, v in enumerate(vals):
            ax.text(v + (1 - lo) * 0.015, y, fmt.format(v), va="center", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Walk-forward test results, all test seasons pooled",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def metric_by_season(metrics: pd.DataFrame, col: str, ylabel: str) -> plt.Figure:
    per = metrics[metrics["season"] != "All"].copy()
    per["season"] = per["season"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for model, color in zip(per["model"].unique(), PALETTE, strict=False):
        m = per[per["model"] == model]
        ax.plot(
            m["season"],
            m[col],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=model,
        )
    ax.set_xticks(sorted(per["season"].unique()))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, fontsize=9)
    ax.set_title(f"{ylabel} by test season", pad=12)
    return fig


def calibration(pred: pd.DataFrame, col: str = "p_lgbm", bins: int = 10) -> plt.Figure:
    frac_pos, mean_pred = calibration_curve(
        pred["podium"], pred[col], n_bins=bins, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.plot(
        mean_pred,
        frac_pos,
        color=PALETTE[0],
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    ax.text(0.62, 0.55, "perfectly calibrated", rotation=45, color=MUTED, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted podium probability")
    ax.set_ylabel("Observed podium rate")
    ax.set_title("Are the probabilities honest?", pad=22)
    _subtitle(ax, "LightGBM, walk-forward test seasons, quantile bins")
    return fig


def importance(imp: pd.DataFrame, top: int = 12) -> plt.Figure:
    imp = imp.head(top).iloc[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top + 1.2))
    ax.barh(labels, imp["importance"], height=0.55, color=PALETTE[0])
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("What drives the podium prediction", pad=22)
    _subtitle(ax, "Permutation importance on held-out seasons")
    return fig


def race_probabilities(race: pd.DataFrame, col: str = "p_lgbm") -> plt.Figure:
    race = race.sort_values(col).reset_index(drop=True)
    labels = race["driver_code"].fillna(race["driver_name"])
    on_podium = (race["finish_position"] <= 3).to_numpy()
    y = np.arange(len(race))
    probs = race[col].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(6.5, 0.26 * len(race) + 1.4))
    ax.barh(y, probs, height=0.6, color=np.where(on_podium, PALETTE[0], OTHER))
    for yi, p in zip(y, probs, strict=True):
        if p >= 10:
            ax.text(p + 1, yi, f"{p:.0f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Predicted podium probability (%)")
    ax.grid(axis="y", visible=False)
    handles = [
        mpl.patches.Patch(color=PALETTE[0], label="Finished on the podium"),
        mpl.patches.Patch(color=OTHER, label="Did not finish on the podium"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    title = f"{race['season'].iloc[0]} {race['race_name'].iloc[0]}"
    ax.set_title(title, pad=22)
    _subtitle(ax, "Pre-race podium probability vs. what actually happened")
    return fig
