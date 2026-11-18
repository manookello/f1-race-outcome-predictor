import pandas as pd
import pytest

from f1predict.model import evaluate, race_ranking_metrics, relevance


def _race(scores):
    return pd.DataFrame(
        {
            "race_id": "2024-01",
            "season": 2024,
            "driver_id": list("abcde"),
            "finish_position": [1, 2, 3, 4, 5],
            "podium": [1, 1, 1, 0, 0],
            "score": scores,
        }
    )


def test_perfect_ranking():
    m = race_ranking_metrics(_race([5, 4, 3, 2, 1]), "score")
    assert m == {"top3_hit": 1.0, "winner_hit": 1.0, "spearman": pytest.approx(1.0)}


def test_reversed_ranking():
    m = race_ranking_metrics(_race([1, 2, 3, 4, 5]), "score")
    assert m["winner_hit"] == 0.0
    assert m["top3_hit"] == pytest.approx(1 / 3)
    assert m["spearman"] == pytest.approx(-1.0)


def test_relevance_labels():
    labels = relevance(pd.Series([1, 2, 20, 21, 25, None]))
    assert labels.tolist() == [20, 19, 1, 0, 0, 0]


def test_evaluate_shapes():
    race = _race([5, 4, 3, 2, 1])
    pred = race.assign(
        s_grid=race["score"],
        p_logreg=race["score"] / 10,
        p_lgbm=race["score"] / 10,
        s_rank=race["score"],
    )
    metrics = evaluate(pred)
    assert set(metrics["model"]) == {
        "Grid order",
        "Logistic regression",
        "LightGBM classifier",
        "LightGBM LambdaRank",
    }
    assert (metrics["roc_auc"] == 1.0).all()
