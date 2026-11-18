import numpy as np
import pandas as pd
import pytest

from f1predict.features import build_features, is_finished, lap_time_to_seconds


def test_lap_time_to_seconds():
    assert lap_time_to_seconds("1:23.456") == pytest.approx(83.456)
    assert lap_time_to_seconds("59.1") == pytest.approx(59.1)
    assert np.isnan(lap_time_to_seconds(""))
    assert np.isnan(lap_time_to_seconds(None))


@pytest.mark.parametrize(
    "status, expected",
    [
        ("Finished", True),
        ("+1 Lap", True),
        ("+3 Laps", True),
        ("Lapped", True),
        ("Engine", False),
        ("Accident", False),
        ("Retired", False),
        (None, False),
    ],
)
def test_is_finished(status, expected):
    assert is_finished(status) is expected


def _synthetic(n_races: int = 6):
    """Two teams x two drivers over ``n_races`` races with a known finishing pattern."""
    drivers = [("a", "t1"), ("b", "t1"), ("c", "t2"), ("d", "t2")]
    races, results, quali = [], [], []
    rng = np.random.default_rng(0)
    for rnd in range(1, n_races + 1):
        races.append(
            {
                "season": 2020,
                "round": rnd,
                "race_name": f"GP {rnd}",
                "date": f"2020-0{1 + rnd // 4}-{(rnd * 5) % 28 + 1:02d}",
                "circuit_id": "x" if rnd % 2 else "y",
                "circuit_name": "X",
                "country": "Nowhere",
            }
        )
        order = rng.permutation(4)
        for pos, idx in enumerate(order, start=1):
            drv, team = drivers[idx]
            results.append(
                {
                    "season": 2020,
                    "round": rnd,
                    "driver_id": drv,
                    "driver_code": drv.upper(),
                    "driver_name": drv,
                    "constructor_id": team,
                    "constructor_name": team,
                    "grid": pos,
                    "position": pos,
                    "points": [25, 18, 15, 12][pos - 1],
                    "status": "Finished" if pos < 4 else "Engine",
                }
            )
            quali.append(
                {
                    "season": 2020,
                    "round": rnd,
                    "driver_id": drv,
                    "quali_position": pos,
                    "q1": f"1:{20 + pos}.000",
                    "q2": None,
                    "q3": None,
                }
            )
    races = pd.DataFrame(races)
    races["date"] = pd.to_datetime(races["date"]).dt.strftime("%Y-%m-%d")
    # make dates strictly increasing with round
    races["date"] = pd.date_range("2020-03-01", periods=n_races, freq="14D").strftime("%Y-%m-%d")
    return races, pd.DataFrame(results), pd.DataFrame(quali)


def test_rolling_form_uses_only_previous_races():
    races, results, quali = _synthetic()
    feats = build_features(races, results, quali)
    for drv in "abcd":
        d = feats[feats["driver_id"] == drv].sort_values("round")
        finishes = d["finish_position"].to_numpy()
        # first race: no history
        assert np.isnan(d["drv_form_3"].iloc[0])
        for i in range(1, len(d)):
            expected = finishes[max(0, i - 3) : i].mean()
            assert d["drv_form_3"].iloc[i] == pytest.approx(expected)
        # career starts counts previous races only
        assert d["drv_starts"].tolist() == list(range(len(d)))


def test_championship_points_exclude_current_race():
    races, results, quali = _synthetic()
    feats = build_features(races, results, quali)
    for drv in "abcd":
        d = feats[feats["driver_id"] == drv].sort_values("round")
        cum_before = d["points"].cumsum().shift(1).fillna(0).to_numpy()
        assert np.allclose(d["drv_points_before"].to_numpy(), cum_before)
    assert feats.loc[feats["round"] == 1, "drv_champ_rank_before"].isna().all()


def test_quali_gap_and_teammate_delta():
    races, results, quali = _synthetic(1)
    feats = build_features(races, results, quali).set_index("driver_id")
    pole = feats["quali_gap_pct"].min()
    assert pole == pytest.approx(0.0)
    # teammates' deltas are mirror images
    t1 = feats[feats["constructor_id"] == "t1"]["teammate_quali_delta"]
    assert t1.sum() == pytest.approx(0.0)


def test_targets():
    races, results, quali = _synthetic(2)
    feats = build_features(races, results, quali)
    assert feats.groupby("race_id")["podium"].sum().eq(3).all()
    assert feats.groupby("race_id")["win"].sum().eq(1).all()
    assert (feats.loc[feats["finish_position"] == 4, "dnf"] == 1).all()
