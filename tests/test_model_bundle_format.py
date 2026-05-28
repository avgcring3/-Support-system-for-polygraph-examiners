from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.predict import load_model_bundle


def _fit_dummy_model() -> DummyClassifier:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.2], [0.2, 0.9]])
    y = np.array([0, 1, 0, 1])
    model = DummyClassifier(strategy="prior")
    model.fit(x, y)
    return model


def test_load_model_bundle_accepts_single_model_format(tmp_path: Path) -> None:
    model = _fit_dummy_model()
    bundle = {
        "model": model,
        "feature_cols": ["f1", "f2"],
        "threshold": 0.5,
    }
    path = tmp_path / "single.pkl"
    joblib.dump(bundle, path)

    loaded = load_model_bundle(path)
    assert loaded["feature_cols"] == ["f1", "f2"]
    assert "model" in loaded


def test_load_model_bundle_accepts_ensemble_format(tmp_path: Path) -> None:
    model = _fit_dummy_model()
    bundle = {
        "model_kind": "ensemble",
        "models": [{"name": "dummy", "model": model, "weight": 1.0, "threshold": 0.5}],
        "feature_cols": ["f1", "f2"],
        "label_col": "label_train",
        "threshold": 0.45,
    }
    path = tmp_path / "ensemble.pkl"
    joblib.dump(bundle, path)

    loaded = load_model_bundle(path)
    assert "models" in loaded
    assert isinstance(loaded["models"], list)
    assert len(loaded["models"]) == 1


def test_load_model_bundle_rejects_missing_feature_cols(tmp_path: Path) -> None:
    bundle = {"model": _fit_dummy_model()}
    path = tmp_path / "bad_missing_features.pkl"
    joblib.dump(bundle, path)

    with pytest.raises(ValueError, match="feature_cols"):
        load_model_bundle(path)


def test_load_model_bundle_rejects_missing_model_and_models(tmp_path: Path) -> None:
    bundle = {"feature_cols": ["f1", "f2"], "threshold": 0.5}
    path = tmp_path / "bad_missing_model.pkl"
    joblib.dump(bundle, path)

    with pytest.raises(ValueError, match="Expected `model` or non-empty `models`"):
        load_model_bundle(path)


def test_load_model_bundle_rejects_empty_ensemble_list(tmp_path: Path) -> None:
    bundle = {
        "model_kind": "ensemble",
        "models": [],
        "feature_cols": ["f1", "f2"],
    }
    path = tmp_path / "bad_empty_ensemble.pkl"
    joblib.dump(bundle, path)

    with pytest.raises(ValueError, match="Expected `model` or non-empty `models`"):
        load_model_bundle(path)
