"""
model.py
========
A small multinomial logistic-regression model predicting match outcome
(Home win / Draw / Away win) from the engineered features.

The spec asks for "3 separate models (home/draw/away)". Scikit-learn's
``LogisticRegression`` with ``multi_class='multinomial'`` fits exactly that:
one set of coefficients per class, with a softmax that guarantees the three
probabilities sum to 1 (i.e. 100%). This is the standard, well-calibrated way
to get three mutually-exclusive probabilities and avoids the normalisation
hacks a trio of independent binary models would need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, build_fixture_features, build_training_frame

CLASSES = ["H", "D", "A"]  # home / draw / away


@dataclass
class TrainedModel:
    """Bundle of the fitted pipeline, its training accuracy and row count."""
    pipeline: Pipeline
    accuracy: float
    n_matches: int


def train_model(historical: pd.DataFrame) -> TrainedModel:
    """
    Build features from historical matches and fit the logistic model.
    Returns a TrainedModel. Falls back to a class-prior 'model' if there is
    too little data to fit (keeps the app usable rather than crashing).
    """
    frame = build_training_frame(historical)
    if len(frame) < 30:
        # Not enough to learn anything; return a trivial baseline.
        return _baseline_model(frame)

    X = frame[FEATURE_COLUMNS].fillna(0.0)
    y = frame["result"]

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                # Recent scikit-learn (>=1.7) fits multinomial by default for
                # multi-class targets and dropped the explicit 'multi_class'
                # argument, so we no longer pass it.
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    accuracy = float(pipeline.score(X, y))
    return TrainedModel(pipeline=pipeline, accuracy=accuracy, n_matches=len(frame))


def predict_fixture(
    model: TrainedModel, historical: pd.DataFrame, home: str, away: str
) -> dict:
    """
    Predict a single fixture. Returns a dict with per-outcome probabilities
    (as percentages summing to 100), the predicted label and confidence.
    """
    features = build_fixture_features(historical, home, away)
    probs = _predict_proba(model, features)[0]

    prob_map = {cls: round(float(p) * 100, 1) for cls, p in zip(model_classes(model), probs)}
    # Ensure all three keys exist even if a class was unseen in training.
    for cls in CLASSES:
        prob_map.setdefault(cls, 0.0)

    best = max(prob_map, key=prob_map.get)
    label = {"H": home, "D": "Draw", "A": away}[best]
    return {
        "home": home,
        "away": away,
        "prob_home": prob_map["H"],
        "prob_draw": prob_map["D"],
        "prob_away": prob_map["A"],
        "predicted": label,
        "confidence": prob_map[best],
    }


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def model_classes(model: TrainedModel) -> list[str]:
    """Class order the fitted estimator uses (may differ from CLASSES)."""
    clf = model.pipeline.named_steps.get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        return list(clf.classes_)
    return CLASSES


def _predict_proba(model: TrainedModel, features: pd.DataFrame) -> np.ndarray:
    clf = model.pipeline.named_steps.get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        return model.pipeline.predict_proba(features[FEATURE_COLUMNS].fillna(0.0))
    # Baseline: broadcast stored priors to one row.
    return np.tile(model.pipeline.priors, (len(features), 1))  # type: ignore[attr-defined]


def _baseline_model(frame: pd.DataFrame) -> TrainedModel:
    """
    Degenerate fallback used when there isn't enough data to fit. Predicts the
    empirical class distribution (or a sensible default) for every fixture.
    """
    if frame.empty:
        priors = np.array([0.45, 0.27, 0.28])  # typical PL home/draw/away split
    else:
        counts = frame["result"].value_counts(normalize=True)
        priors = np.array([counts.get(c, 0.0) for c in CLASSES])
        if priors.sum() == 0:
            priors = np.array([0.45, 0.27, 0.28])
        priors = priors / priors.sum()

    class _PriorPipeline:
        """Minimal stand-in exposing the bits model.py touches."""
        named_steps: dict = {}
        priors = priors

    pipe = _PriorPipeline()
    pipe.named_steps = {}  # no 'clf' -> _predict_proba uses priors
    pipe.priors = priors
    return TrainedModel(pipeline=pipe, accuracy=0.0, n_matches=len(frame))  # type: ignore[arg-type]
