from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

try:
    from src.predict import load_model_bundle, predict_df, summarize_predictions
    from src.preprocess import build_feature_dataset
except ImportError as exc:
    raise RuntimeError("Run API from project root so `src` package is importable.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

app = FastAPI(title="Polygraph DSS API", version="0.1.0")
_model_bundle: dict[str, Any] | None = None


def _get_model_bundle() -> dict[str, Any]:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model_bundle = load_model_bundle(MODEL_PATH)
    return _model_bundle


def _read_uploaded_dataframe(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    raw = file.file.read()
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))


def _prepare_features_for_inference(df: pd.DataFrame, upload_name: str | None = None) -> pd.DataFrame:
    lower_map = {col.lower(): col for col in df.columns}
    has_feature_cols = any(col.startswith("i_") or col.startswith("ii_") for col in df.columns)
    if has_feature_cols:
        return df

    required_long = {"question_raw", "parameter"}
    if required_long.issubset(set(lower_map.keys())):
        frame = df.rename(
            columns={
                lower_map.get("question_raw", "question_raw"): "question_raw",
                lower_map.get("parameter", "parameter"): "parameter",
            }
        ).copy()
        if "metric_i" not in frame.columns:
            if "I" in frame.columns:
                frame["metric_i"] = frame["I"]
            elif "i" in frame.columns:
                frame["metric_i"] = frame["i"]
        if "metric_ii" not in frame.columns:
            if "II" in frame.columns:
                frame["metric_ii"] = frame["II"]
            elif "ii" in frame.columns:
                frame["metric_ii"] = frame["ii"]
        if "score_result" not in frame.columns:
            if "Результат" in frame.columns:
                frame["score_result"] = frame["Результат"]
            else:
                frame["score_result"] = np.nan
        if "test_id" not in frame.columns:
            frame["test_id"] = "uploaded_test"
        if "source_file" not in frame.columns:
            frame["source_file"] = upload_name if upload_name else "uploaded.csv"
        if "source_type" not in frame.columns:
            frame["source_type"] = "upload"
        if "question_code" not in frame.columns:
            frame["question_code"] = frame["question_raw"]
        if "question_text" not in frame.columns:
            frame["question_text"] = frame["question_raw"]
        if "question_type" not in frame.columns:
            frame["question_type"] = "other"
        return build_feature_dataset(frame, weak_label_threshold=0.55)

    raise ValueError(
        "Unsupported file schema. Upload `polygram_features.csv` or long format with columns question_raw + parameter."
    )


def _top_features(pred_df: pd.DataFrame) -> list[str]:
    candidate_cols = [col for col in pred_df.columns if col.startswith("i_") or col.startswith("ii_")]
    if not candidate_cols:
        return []
    importance = pred_df[candidate_cols].abs().mean().sort_values(ascending=False)
    return importance.head(3).index.tolist()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_polygram(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        raw_df = _read_uploaded_dataframe(file)
        feature_df = _prepare_features_for_inference(raw_df, upload_name=file.filename)
        bundle = _get_model_bundle()
        pred_df = predict_df(feature_df, bundle)
        summary = summarize_predictions(pred_df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to analyze file: {exc}") from exc

    cols_for_output = [col for col in ["question_code", "question_raw", "question_type", "deception_probability"] if col in pred_df.columns]
    records = pred_df[cols_for_output].to_dict(orient="records")
    return {
        "overall_score": summary["overall_score"],
        "recommendation": summary["recommendation"],
        "per_question_scores": records,
        "key_channels": _top_features(pred_df),
        "n_items": summary["n_items"],
    }
