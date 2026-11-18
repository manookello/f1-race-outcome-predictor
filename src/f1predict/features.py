"""Feature engineering.

Every feature describes information that is available **before the lights go out** on race day:
the starting grid, qualifying pace, the same weekend's sprint, and rolling form computed only from
earlier races (always ``shift(1)`` before any rolling window). ``tests/test_features.py`` guards
against leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1predict.config import CONSTRUCTOR_LINEAGE, STREET_CIRCUITS

FEATURES = [
    "grid",
    "pit_lane_start",
    "quali_position",
    "quali_gap_pct",
    "teammate_quali_delta",
    "sprint_position",
    "drv_form_3",
    "drv_form_5",
    "drv_points_form_5",
    "drv_podium_rate_10",
    "drv_dnf_rate_10",
    "drv_starts",
    "drv_champ_rank_before",
    "drv_points_share_before",
    "con_points_form_5",
    "con_best_finish_form_5",
    "con_champ_rank_before",
    "drv_circuit_avg_finish",
    "drv_circuit_starts",
    "round",
    "street_circuit",
]

FEATURE_LABELS = {
    "grid": "Grid position",
    "pit_lane_start": "Pit-lane start",
    "quali_position": "Qualifying position",
    "quali_gap_pct": "Qualifying gap to pole (%)",
    "teammate_quali_delta": "Quali gap vs teammate (pp)",
    "sprint_position": "Sprint finish (same weekend)",
    "drv_form_3": "Driver avg finish, last 3",
    "drv_form_5": "Driver avg finish, last 5",
    "drv_points_form_5": "Driver avg points, last 5",
    "drv_podium_rate_10": "Driver podium rate, last 10",
    "drv_dnf_rate_10": "Driver DNF rate, last 10",
    "drv_starts": "Driver career starts",
    "drv_champ_rank_before": "Driver championship rank",
    "drv_points_share_before": "Driver points vs leader",
    "con_points_form_5": "Team avg points, last 5",
    "con_best_finish_form_5": "Team best finish, last 5",
    "con_champ_rank_before": "Team championship rank",
    "drv_circuit_avg_finish": "Driver avg finish at circuit",
    "drv_circuit_starts": "Driver starts at circuit",
    "round": "Round of season",
    "street_circuit": "Street circuit",
}


def lap_time_to_seconds(value) -> float:
    """Convert '1:23.456' (or '83.456') to seconds; blanks become NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return np.nan
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def is_finished(status: str | None) -> bool:
    """Classified finishers: 'Finished', '+1 Lap', '+2 Laps', or Jolpica's newer 'Lapped'."""
    if not isinstance(status, str):
        return False
    return status == "Finished" or status.startswith("+") or status == "Lapped"


def _rolling_prev(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Rolling mean over the *previous* ``window`` observations (excludes the current row)."""
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


def build_features(
    races: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    sprints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per (race, driver) with targets and pre-race features."""
    df = results.merge(
        races[["season", "round", "race_name", "date", "circuit_id", "circuit_name", "country"]],
        on=["season", "round"],
        how="left",
    )
    df["date"] = pd.to_datetime(df["date"])
    df["race_id"] = df["season"].astype(str) + "-" + df["round"].astype(str).str.zfill(2)
    df = df.sort_values(["date", "position"], kind="stable").reset_index(drop=True)

    # ---- targets ---------------------------------------------------------------------
    df["finish_position"] = df["position"].astype(float)
    df["podium"] = (df["finish_position"] <= 3).astype(int)
    df["win"] = (df["finish_position"] == 1).astype(int)
    df["dnf"] = (~df["status"].map(is_finished)).astype(int)

    # ---- grid ------------------------------------------------------------------------
    df["pit_lane_start"] = (df["grid"].fillna(0) == 0).astype(int)
    n_cars = df.groupby("race_id")["driver_id"].transform("count")
    df["grid"] = np.where(df["pit_lane_start"] == 1, n_cars, df["grid"]).astype(float)
    df["n_entrants"] = n_cars

    # ---- qualifying ------------------------------------------------------------------
    q = qualifying.copy()
    for col in ("q1", "q2", "q3"):
        q[col + "_s"] = q[col].map(lap_time_to_seconds)
    q["best_quali_s"] = q[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    q["pole_s"] = q.groupby(["season", "round"])["best_quali_s"].transform("min")
    q["quali_gap_pct"] = (q["best_quali_s"] / q["pole_s"] - 1) * 100
    df = df.merge(
        q[["season", "round", "driver_id", "quali_position", "quali_gap_pct"]],
        on=["season", "round", "driver_id"],
        how="left",
    )
    grp = df.groupby(["race_id", "constructor_id"])["quali_gap_pct"]
    team_sum, team_cnt = grp.transform("sum"), grp.transform("count")
    mate_mean = (team_sum - df["quali_gap_pct"]) / (team_cnt - 1)
    df["teammate_quali_delta"] = df["quali_gap_pct"] - mate_mean.where(team_cnt > 1)

    # ---- sprint (same weekend, run before the Grand Prix) -----------------------------
    if sprints is not None and not sprints.empty:
        s = sprints[["season", "round", "driver_id", "position", "points"]].rename(
            columns={"position": "sprint_position", "points": "sprint_points"}
        )
        df = df.merge(s, on=["season", "round", "driver_id"], how="left")
    else:
        df["sprint_position"] = np.nan
        df["sprint_points"] = np.nan
    df["sprint_points"] = df["sprint_points"].fillna(0.0)

    # ---- driver rolling form (previous races only) ------------------------------------
    df = df.sort_values(["date", "driver_id"], kind="stable").reset_index(drop=True)
    by_drv = df.groupby("driver_id", sort=False)
    df["drv_form_3"] = by_drv["finish_position"].transform(lambda s: _rolling_prev(s, 3))
    df["drv_form_5"] = by_drv["finish_position"].transform(lambda s: _rolling_prev(s, 5))
    df["drv_points_form_5"] = by_drv["points"].transform(lambda s: _rolling_prev(s, 5))
    df["drv_podium_rate_10"] = by_drv["podium"].transform(lambda s: _rolling_prev(s, 10))
    df["drv_dnf_rate_10"] = by_drv["dnf"].transform(lambda s: _rolling_prev(s, 10, 3))
    df["drv_starts"] = by_drv.cumcount()

    # ---- championship standings before this weekend -----------------------------------
    df["weekend_points"] = df["points"] + df["sprint_points"]
    season_drv = df.groupby(["season", "driver_id"], sort=False)["weekend_points"]
    df["drv_points_before"] = season_drv.cumsum() - df["weekend_points"]
    leader = df.groupby("race_id")["drv_points_before"].transform("max")
    df["drv_points_share_before"] = np.where(leader > 0, df["drv_points_before"] / leader, np.nan)
    df["drv_champ_rank_before"] = df.groupby("race_id")["drv_points_before"].rank(
        ascending=False, method="min"
    )
    df.loc[df["round"] == 1, "drv_champ_rank_before"] = np.nan

    # ---- constructor form & standings -------------------------------------------------
    df["team_lineage"] = df["constructor_id"].map(CONSTRUCTOR_LINEAGE).fillna(df["constructor_id"])
    team = (
        df.groupby(["race_id", "season", "date", "constructor_id", "team_lineage"], as_index=False)
        .agg(
            team_points=("points", "sum"),
            team_weekend_points=("weekend_points", "sum"),
            team_best_finish=("finish_position", "min"),
        )
        .sort_values(["date", "constructor_id"], kind="stable")
    )
    by_con = team.groupby("team_lineage", sort=False)
    team["con_points_form_5"] = by_con["team_points"].transform(lambda s: _rolling_prev(s, 5))
    team["con_best_finish_form_5"] = by_con["team_best_finish"].transform(
        lambda s: _rolling_prev(s, 5)
    )
    team_season = team.groupby(["season", "constructor_id"], sort=False)["team_weekend_points"]
    team["con_points_before"] = team_season.cumsum() - team["team_weekend_points"]
    team["con_champ_rank_before"] = team.groupby("race_id")["con_points_before"].rank(
        ascending=False, method="min"
    )
    df = df.merge(
        team[
            [
                "race_id",
                "constructor_id",
                "con_points_form_5",
                "con_best_finish_form_5",
                "con_champ_rank_before",
            ]
        ],
        on=["race_id", "constructor_id"],
        how="left",
    )
    df.loc[df["round"] == 1, "con_champ_rank_before"] = np.nan

    # ---- driver history at this circuit ------------------------------------------------
    df = df.sort_values(["date", "driver_id"], kind="stable").reset_index(drop=True)
    by_dc = df.groupby(["driver_id", "circuit_id"], sort=False)["finish_position"]
    df["drv_circuit_avg_finish"] = by_dc.transform(lambda s: s.shift(1).expanding().mean())
    df["drv_circuit_starts"] = by_dc.cumcount()

    # ---- race context -------------------------------------------------------------------
    df["street_circuit"] = df["circuit_id"].isin(STREET_CIRCUITS).astype(int)

    return df.sort_values(["date", "finish_position"], kind="stable").reset_index(drop=True)
