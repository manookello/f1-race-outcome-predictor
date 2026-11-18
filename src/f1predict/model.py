"""Models, walk-forward validation and evaluation metrics.

Four approaches are compared on exactly the same (race, driver) rows:

* **Grid order** – baseline: the car starting further forward finishes further forward.
* **Logistic regression** – linear podium classifier on the same features.
* **LightGBM classifier** – gradient-boosted podium probability.
* **LightGBM LambdaRank** – learns the full finishing order of each race directly.

Validation is *walk-forward by season*: to score season ``Y`` we train only on seasons ``< Y``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1predict.config import MODELS, TEST_SEASONS
from f1predict.features import FEATURES

SCORE_COLUMNS = {
    "Grid order": "s_grid",
    "Logistic regression": "p_logreg",
    "LightGBM classifier": "p_lgbm",
    "LightGBM LambdaRank": "s_rank",
}
PROBABILISTIC = {"Logistic regression", "LightGBM classifier"}

ID_COLUMNS = [
    "race_id",
    "season",
    "round",
    "race_name",
    "date",
    "circuit_id",
    "driver_id",
    "driver_code",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "grid",
    "quali_position",
    "finish_position",
    "status",
    "podium",
    "win",
]

MAX_RELEVANCE = 20

# Hyper-parameters were chosen on validation seasons 2019-2021 (training on earlier seasons
# only), i.e. before any of the 2022-2026 test seasons - see notebooks/03_modeling.ipynb.


def make_classifier(random_state: int = 7) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=2000)),
        ]
    )


def make_ranker(random_state: int = 7) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        label_gain=[i * i for i in range(MAX_RELEVANCE + 1)],
        lambdarank_truncation_level=20,
        random_state=random_state,
        verbose=-1,
    )


def relevance(finish_position: pd.Series) -> np.ndarray:
    """Winner -> 20, P2 -> 19, ... P21 and beyond -> 0 (integer labels for LambdaRank)."""
    return np.clip(
        MAX_RELEVANCE + 1 - finish_position.fillna(99).to_numpy(), 0, MAX_RELEVANCE
    ).astype(int)


def _group_sizes(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("race_id", sort=False).size().to_numpy()


def fit_all(train: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    """Fit the three learned models on ``train`` and return them in a dict."""
    train = train.sort_values(["date", "race_id"], kind="stable")
    X, y = train[features], train["podium"]
    return {
        "Logistic regression": make_logreg().fit(X, y),
        "LightGBM classifier": make_classifier().fit(X, y),
        "LightGBM LambdaRank": make_ranker().fit(
            X, relevance(train["finish_position"]), group=_group_sizes(train)
        ),
    }


def score(models: dict, df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Score rows with every model; returns ids + one score column per model."""
    out = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
    X = df[features]
    out["s_grid"] = -df["grid"].to_numpy()
    out["p_logreg"] = models["Logistic regression"].predict_proba(X)[:, 1]
    out["p_lgbm"] = models["LightGBM classifier"].predict_proba(X)[:, 1]
    out["s_rank"] = models["LightGBM LambdaRank"].predict(X)
    return out


def walk_forward(
    df: pd.DataFrame,
    test_seasons: list[int] = TEST_SEASONS,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """Out-of-sample predictions for each test season, training only on earlier seasons."""
    parts = []
    for season in test_seasons:
        train, test = df[df["season"] < season], df[df["season"] == season]
        if test.empty:
            continue
        parts.append(score(fit_all(train, features), test, features))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def race_ranking_metrics(race: pd.DataFrame, score_col: str) -> dict:
    """Top-3 hit rate, winner hit and Spearman rho for one race."""
    ordered = race.sort_values(score_col, ascending=False, kind="stable")
    predicted_top3 = set(ordered["driver_id"].iloc[:3])
    actual_top3 = set(race.loc[race["finish_position"] <= 3, "driver_id"])
    winner = set(race.loc[race["finish_position"] == 1, "driver_id"])
    rho = spearmanr(-race[score_col].to_numpy(), race["finish_position"].to_numpy()).statistic
    return {
        "top3_hit": len(predicted_top3 & actual_top3) / 3,
        "winner_hit": float(ordered["driver_id"].iloc[0] in winner),
        "spearman": float(rho),
    }


def evaluate(pred: pd.DataFrame, by: str | None = "season") -> pd.DataFrame:
    """Metrics per model (and per ``by`` group, plus an 'All' row)."""
    groups = [("All", pred)]
    if by is not None:
        groups += [(key, g) for key, g in pred.groupby(by)]
    rows = []
    for model, col in SCORE_COLUMNS.items():
        for key, g in groups:
            per_race = pd.DataFrame(
                [race_ranking_metrics(r, col) for _, r in g.groupby("race_id")]
            ).mean()
            row = {
                "model": model,
                by or "group": key,
                "races": g["race_id"].nunique(),
                "roc_auc": roc_auc_score(g["podium"], g[col]),
                "top3_hit": per_race["top3_hit"],
                "winner_acc": per_race["winner_hit"],
                "spearman": per_race["spearman"],
                "log_loss": np.nan,
                "brier": np.nan,
            }
            if model in PROBABILISTIC:
                row["log_loss"] = log_loss(g["podium"], g[col].clip(1e-6, 1 - 1e-6))
                row["brier"] = brier_score_loss(g["podium"], g[col])
            rows.append(row)
    return pd.DataFrame(rows)


def permutation_importance_table(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 7
) -> pd.DataFrame:
    """Drop in ROC-AUC when each feature is shuffled (on held-out data)."""
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# Final model persistence
# --------------------------------------------------------------------------------------


def train_final(df: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    """Train the podium classifier on every available race and return a saveable bundle."""
    model = make_classifier().fit(df[features], df["podium"])
    return {
        "model": model,
        "features": features,
        "trained_through": str(df["date"].max().date()),
        "n_rows": len(df),
    }


def save_bundle(bundle: dict, path: Path = MODELS / "podium_lgbm.joblib") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = MODELS / "podium_lgbm.joblib") -> dict:
    return joblib.load(path)
