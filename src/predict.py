from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle format. Expected a dict.")

    feature_cols = bundle.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Invalid model bundle format. Expected non-empty `feature_cols` list.")

    has_single_model = "model" in bundle
    has_ensemble = (
        isinstance(bundle.get("models"), list)
        and len(bundle["models"]) > 0
        and all(isinstance(item, dict) and "model" in item for item in bundle["models"])
    )
    if not (has_single_model or has_ensemble):
        raise ValueError("Invalid model bundle format. Expected `model` or non-empty `models`.")
    return bundle


def prepare_feature_matrix(features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = features_df.copy()
    for col in feature_cols:
        if col not in x.columns:
            x[col] = np.nan
    x = x[feature_cols]
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x


def predict_df(features_df: pd.DataFrame, model_bundle: dict[str, Any]) -> pd.DataFrame:
    feature_cols: list[str] = model_bundle["feature_cols"]
    threshold = float(model_bundle.get("threshold", 0.5))
    x = prepare_feature_matrix(features_df, feature_cols)
    if "models" in model_bundle and isinstance(model_bundle["models"], list):
        numer = np.zeros(len(x), dtype=float)
        denom = 0.0
        for item in model_bundle["models"]:
            mdl = item["model"]
            w = float(item.get("weight", 1.0))
            numer += w * mdl.predict_proba(x)[:, 1]
            denom += w
        if denom <= 0:
            denom = float(len(model_bundle["models"])) if model_bundle["models"] else 1.0
        proba = numer / denom
    else:
        model = model_bundle["model"]
        proba = model.predict_proba(x)[:, 1]
    out = features_df.copy()
    out["deception_probability"] = proba
    out["pred_label"] = (out["deception_probability"] >= threshold).astype(int)
    return out


def summarize_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    overall = float(pred_df["deception_probability"].mean()) if len(pred_df) else float("nan")
    if overall >= 0.60:
        recommendation = "Вероятно ложь"
    elif overall <= 0.40:
        recommendation = "Вероятно правда"
    else:
        recommendation = "Требует внимания эксперта"
    return {"overall_score": overall, "recommendation": recommendation, "n_items": int(len(pred_df))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for polygraph DSS features.")
    parser.add_argument("--input-features", type=Path, default=Path("data/processed/polygram_features.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/best_model.pkl"))
    parser.add_argument("--output-path", type=Path, default=Path("reports/predictions_inference.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_features)
    bundle = load_model_bundle(args.model_path)
    pred_df = predict_df(df, bundle)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    summary = summarize_predictions(pred_df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[predict] saved: {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
