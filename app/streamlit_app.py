"""Interactive explorer for the F1 race outcome models.

Run with:  streamlit run app/streamlit_app.py
Reads only the processed files committed under data/processed/ (no API calls).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from f1predict.config import DATA_PROCESSED
from f1predict.model import SCORE_COLUMNS, race_ranking_metrics
from f1predict.plots import OTHER, PALETTE

st.set_page_config(page_title="F1 Race Outcome Predictor", page_icon="🏁", layout="wide")
MODEL_COLORS = dict(zip(SCORE_COLUMNS, PALETTE, strict=False))


@st.cache_data
def load():
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    pred = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    metrics = pd.read_csv(DATA_PROCESSED / "metrics.csv")
    imp = pd.read_csv(DATA_PROCESSED / "importance.csv")
    return feats, pred, metrics, imp


def tidy(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showline=True, gridwidth=1)
    fig.update_yaxes(gridwidth=1)
    return fig


feats, pred, metrics, imp = load()

st.title("F1 Race Outcome Predictor")
st.caption(
    "Podium probabilities and finishing order from pre-race information only: grid, qualifying "
    "pace, sprint result and rolling driver/team form. Every prediction shown is out of sample. "
    "Season *Y* was predicted by models trained only on seasons before *Y*."
)

tab_race, tab_perf, tab_imp, tab_era = st.tabs(
    ["Race predictions", "Model performance", "What drives it", "Era explorer"]
)

# ---------------------------------------------------------------------------------------
with tab_race:
    c1, c2, c3 = st.columns([1, 2, 2])
    seasons = sorted(pred["season"].unique(), reverse=True)
    season = c1.selectbox("Season", seasons)
    races = (
        pred[pred["season"] == season]
        .drop_duplicates("race_id")
        .sort_values("round", ascending=False)
    )
    race_label = c2.selectbox("Race", races["race_name"].tolist())
    model_name = c3.selectbox("Model", ["LightGBM classifier", "Logistic regression"])
    col = SCORE_COLUMNS[model_name]
    race = pred[(pred["season"] == season) & (pred["race_name"] == race_label)].copy()
    race["driver"] = race["driver_code"].fillna(race["driver_name"])

    rm = race_ranking_metrics(race, col)
    top3_pred = race.nlargest(3, col)["driver"].tolist()
    top3_actual = race.nsmallest(3, "finish_position")["driver"].tolist()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted podium", " · ".join(top3_pred))
    k2.metric("Actual podium", " · ".join(top3_actual))
    k3.metric("Podium drivers found", f"{rm['top3_hit'] * 3:.0f} of 3")
    k4.metric("Order correlation (Spearman)", f"{rm['spearman']:.2f}")

    race = race.sort_values(col)
    race["result"] = race["finish_position"].map(
        lambda p: "Finished on the podium" if p <= 3 else "Did not finish on the podium"
    )
    fig = px.bar(
        race,
        x=col,
        y="driver",
        orientation="h",
        color="result",
        color_discrete_map={
            "Finished on the podium": PALETTE[0],
            "Did not finish on the podium": OTHER,
        },
        category_orders={"driver": race["driver"].tolist()},
        hover_data={
            "driver_name": True,
            "constructor_name": True,
            "grid": ":.0f",
            "finish_position": ":.0f",
            col: ":.1%",
            "driver": False,
            "result": False,
        },
        labels={col: "Predicted podium probability", "driver": "", "result": ""},
        title=f"{season} {race_label}",
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(tidy(fig, 30 * len(race) + 120), use_container_width=True)

    with st.expander("Table view"):
        table = race.sort_values(col, ascending=False)[
            ["driver_name", "constructor_name", "grid", col, "finish_position", "status"]
        ].rename(columns={col: "podium_probability"})
        st.dataframe(
            table.style.format(
                {"podium_probability": "{:.1%}", "grid": "{:.0f}", "finish_position": "{:.0f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------------------------------
with tab_perf:
    metric_names = {
        "roc_auc": "Podium ROC-AUC",
        "top3_hit": "Top-3 hit rate",
        "winner_acc": "Winner accuracy",
        "spearman": "Spearman (full order)",
    }
    metric = st.radio("Metric", list(metric_names), format_func=metric_names.get, horizontal=True)
    per = metrics[metrics["season"] != "All"].copy()
    per["season"] = per["season"].astype(int)
    fig = px.line(
        per,
        x="season",
        y=metric,
        color="model",
        markers=True,
        color_discrete_map=MODEL_COLORS,
        labels={metric: metric_names[metric], "season": "Test season", "model": ""},
    )
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(tidy(fig), use_container_width=True)
    overall = metrics[metrics["season"] == "All"].drop(columns="season").set_index("model")
    st.dataframe(
        overall.style.format(
            {
                "roc_auc": "{:.3f}",
                "top3_hit": "{:.1%}",
                "winner_acc": "{:.1%}",
                "spearman": "{:.3f}",
                "log_loss": "{:.3f}",
                "brier": "{:.4f}",
            },
            na_rep="–",
        ),
        use_container_width=True,
    )
    st.caption("Grid order = baseline that predicts the finishing order equals the starting grid.")

# ---------------------------------------------------------------------------------------
with tab_imp:
    top = imp.head(15).iloc[::-1]
    fig = px.bar(
        top,
        x="importance",
        y="label",
        orientation="h",
        error_x="std",
        color_discrete_sequence=[PALETTE[0]],
        labels={"importance": "Drop in ROC-AUC when shuffled", "label": ""},
        title="Permutation importance (trained 2014–2024, scored on 2025–2026)",
    )
    st.plotly_chart(tidy(fig, 520), use_container_width=True)

# ---------------------------------------------------------------------------------------
with tab_era:
    wins = feats[feats["win"] == 1]
    leaders = wins["team_lineage"].value_counts().index[:4].tolist()
    names = feats.sort_values("date").groupby("team_lineage")["constructor_name"].last()
    wins = wins.assign(
        team=wins["team_lineage"].map(lambda t: names[t] if t in leaders else "Other teams")
    )
    share = wins.groupby(["season", "team"]).size().rename("wins").reset_index()
    order = [names[t] for t in leaders] + ["Other teams"]
    fig = px.bar(
        share,
        x="season",
        y="wins",
        color="team",
        category_orders={"team": order},
        color_discrete_map=dict(zip(order, PALETTE[:4] + [OTHER], strict=True)),
        labels={"wins": "Race wins", "season": "", "team": ""},
        title="Race wins per season by team",
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color="rgba(0,0,0,0)")
    fig.update_layout(bargap=0.35)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(tidy(fig), use_container_width=True)

    rate = feats[feats["grid"].between(1, 20)].groupby("grid")["podium"].mean().reset_index()
    fig = px.bar(
        rate,
        x="grid",
        y="podium",
        color_discrete_sequence=[PALETTE[0]],
        labels={"grid": "Starting grid position", "podium": "Podium rate"},
        title="How often each grid slot ends on the podium (2014–2026)",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(dtick=1)
    fig.update_layout(bargap=0.4)
    st.plotly_chart(tidy(fig, 380), use_container_width=True)
