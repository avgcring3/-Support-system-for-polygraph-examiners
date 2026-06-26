from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import parse_docx_table, parse_xdex_file  # noqa: E402
from src.predict import load_model_bundle, predict_df, summarize_predictions  # noqa: E402
from src.preprocess import build_feature_dataset, normalize_parameter_name, run_preprocess  # noqa: E402
from src.process_single_xdex import run_single_xdex  # noqa: E402
from src.train import train_models  # noqa: E402
from src.xdex_epp import run_xdex_epp_detailed  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
RUNS_DIR = PROJECT_ROOT / "reports" / "ui_runs"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "processed" / "feedback_labels.csv"
DEMO_FILE_PATH = PROJECT_ROOT / "data" / "raw" / "xdex" / "1.xdex"


class DemoUploadedFile:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: #f3f6fb;
            color: #0f172a;
          }

          .stApp h1,
          .stApp h2,
          .stApp h3,
          .stApp h4,
          .stApp li,
          .stApp label,
          .stApp [data-testid="stMarkdownContainer"],
          .stApp [data-testid="stMarkdownContainer"] * {
            color: #0f172a !important;
          }

          .stApp [data-testid="stMetricLabel"],
          .stApp [data-testid="stMetricValue"] {
            color: #0f172a !important;
          }

          .stApp [data-testid="stExpander"],
          .stApp [data-testid="stExpander"] * {
            background: #ffffff !important;
            color: #0f172a !important;
            border-color: #dbe3ee !important;
          }

          .stApp details,
          .stApp details summary,
          .stApp details summary * {
            background: #ffffff !important;
            color: #0f172a !important;
            border-color: #dbe3ee !important;
          }

          .stApp code {
            background: #eef2f7 !important;
            color: #111827 !important;
            border-radius: 6px !important;
            padding: 2px 6px !important;
          }

          .stApp pre,
          .stApp pre code,
          .stApp [data-testid="stMarkdownContainer"] code {
            background: #eef2f7 !important;
            color: #111827 !important;
          }

          .stButton > button,
          .stDownloadButton > button {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            text-shadow: none !important;
            opacity: 1 !important;
          }

          .stButton > button *,
          .stDownloadButton > button * {
            color: #000000 !important;
            fill: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
          }

          .stButton > button:hover,
          .stDownloadButton > button:hover {
            background: #f3f4f6 !important;
            color: #000000 !important;
            border: 1px solid #000000 !important;
            -webkit-text-fill-color: #000000 !important;
          }

          .main .block-container {
            max-width: 1400px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }

          .hero {
            border-radius: 16px;
            background: #0f172a;
            color: #f8fafc;
            padding: 20px 24px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.2);
            margin-bottom: 10px;
          }

          .hero h1 {
            margin: 0;
            font-size: 30px;
            line-height: 1.2;
            color: #f8fafc !important;
          }

          .hero p {
            margin-top: 8px;
            color: #cbd5e1;
            font-size: 15px;
            margin-bottom: 0;
          }

          .stApp .hero,
          .stApp .hero * {
            color: #f8fafc !important;
          }

          .stApp .hero p,
          .stApp .hero p * {
            color: #cbd5e1 !important;
          }

          .card {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
          }

          .metric {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 12px;
            padding: 12px 14px;
          }

          .metric .label {
            font-size: 12px;
            color: #64748b;
            margin-bottom: 4px;
          }

          .metric .value {
            font-size: 26px;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.1;
          }

          .epp-box {
            border-radius: 12px;
            border: 1px solid #dbe3ee;
            background: #ffffff;
            padding: 14px 16px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Polygraph DSS</h1>
          <p>
            Upload one file, run analysis, and get a complete EPP result:
            probabilities, per-question verdicts, recommendation, and XDEX overview.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        bootstrap_model_artifacts()
    if not MODEL_PATH.exists():
        return None
    return load_model_bundle(MODEL_PATH)


def bootstrap_model_artifacts() -> None:
    """Build generated model artifacts on hosts where git-ignored files are absent."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    docx_dir = PROJECT_ROOT / "data" / "raw" / "docx"
    xdex_dir = PROJECT_ROOT / "data" / "raw" / "xdex"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    xdex_input = xdex_dir if xdex_dir.exists() else None
    long_df, feat_df = run_preprocess(
        docx_dir=docx_dir,
        xdex_dir=xdex_input,
        processed_dir=processed_dir,
        weak_label_threshold=0.55,
    )
    if long_df.empty or feat_df.empty:
        raise ValueError("Не удалось собрать обучающий датасет из data/raw.")

    train_models(
        features_path=processed_dir / "polygram_features.csv",
        models_dir=models_dir,
        reports_dir=reports_dir,
        preferred_label=None,
        feedback_path=FEEDBACK_PATH,
    )


def save_uploaded(uploaded_file: Any, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / uploaded_file.name
    out_path.write_bytes(uploaded_file.getvalue())
    return out_path


def read_table(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))


def prepare_long_records_for_features(
    long_df: pd.DataFrame,
    *,
    file_name: str,
    source_type: str,
) -> pd.DataFrame:
    if long_df.empty:
        raise ValueError("The file does not contain metric rows for scoring.")

    frame = long_df.copy()
    frame["source_file"] = frame.get("source_file", file_name)
    frame["source_type"] = frame.get("source_type", source_type)
    frame["test_id"] = frame.get("test_id", Path(file_name).stem)
    frame["question_raw"] = frame.get("question_raw", frame.get("question_text", ""))
    frame["question_code"] = frame.get("question_code", frame["question_raw"])
    frame["question_text"] = frame.get("question_text", frame["question_raw"])
    frame["question_type"] = frame.get("question_type", "other")
    frame["parameter"] = frame.get("parameter", "")
    frame["metric_i"] = pd.to_numeric(frame.get("metric_i", np.nan), errors="coerce")
    frame["metric_ii"] = pd.to_numeric(frame.get("metric_ii", np.nan), errors="coerce")
    frame["score_result"] = pd.to_numeric(frame.get("score_result", np.nan), errors="coerce")
    frame["parameter_norm"] = frame["parameter"].map(normalize_parameter_name)

    if frame["parameter_norm"].isna().all():
        raise ValueError("Could not detect channels/parameters in the file.")
    if frame["metric_i"].isna().all() and frame["metric_ii"].isna().all():
        raise ValueError("No numeric I/II metrics were found in the file.")

    return frame


def features_from_uploaded(uploaded_file: Any) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".docx"):
        tmp_docx = save_uploaded(uploaded_file, PROJECT_ROOT / "data" / "tmp_uploads")
        long_df = parse_docx_table(tmp_docx)
        required_cols = {"question_raw", "parameter", "metric_i", "metric_ii"}
        if long_df.empty or not required_cols.issubset(set(long_df.columns)):
            raise ValueError("DOCX metrics table not found. Expected columns: question, parameter, I, II, result.")
        long_prepared = prepare_long_records_for_features(
            long_df,
            file_name=uploaded_file.name,
            source_type="docx",
        )
        return build_feature_dataset(long_prepared, weak_label_threshold=0.55)

    if file_name.endswith(".xdex") or file_name.endswith(".xdx"):
        tmp_xdex = save_uploaded(uploaded_file, PROJECT_ROOT / "data" / "tmp_uploads")
        long_df = parse_xdex_file(tmp_xdex)
        if long_df.empty:
            raise ValueError(
                "This XDEX does not contain tabular I/II metrics for ML mode. "
                "Direct score-based mode will be used instead."
            )
        long_prepared = prepare_long_records_for_features(
            long_df,
            file_name=uploaded_file.name,
            source_type="xdex",
        )
        return build_feature_dataset(long_prepared, weak_label_threshold=0.55)

    df = read_table(uploaded_file)

    if any(col.startswith("i_") or col.startswith("ii_") for col in df.columns):
        return df.copy()

    lower_cols = {str(c).lower(): c for c in df.columns}
    if "question_raw" not in lower_cols or "parameter" not in lower_cols:
        raise ValueError("Unsupported table format. Required columns: question_raw and parameter.")

    frame = df.rename(
        columns={
            lower_cols["question_raw"]: "question_raw",
            lower_cols["parameter"]: "parameter",
        }
    ).copy()

    if "metric_i" not in frame.columns:
        if "i" in lower_cols:
            frame["metric_i"] = frame[lower_cols["i"]]
        else:
            frame["metric_i"] = np.nan

    if "metric_ii" not in frame.columns:
        if "ii" in lower_cols:
            frame["metric_ii"] = frame[lower_cols["ii"]]
        else:
            frame["metric_ii"] = np.nan

    if "score_result" not in frame.columns:
        if "result" in lower_cols:
            frame["score_result"] = frame[lower_cols["result"]]
        elif "score" in lower_cols:
            frame["score_result"] = frame[lower_cols["score"]]
        else:
            frame["score_result"] = np.nan

    frame = prepare_long_records_for_features(
        frame,
        file_name=uploaded_file.name,
        source_type="upload",
    )
    return build_feature_dataset(frame, weak_label_threshold=0.55)


def build_final_html_report(
    output_path: Path,
    xdex_summary: dict[str, Any] | None,
    epp_summary: dict[str, Any] | None,
    epp_table: pd.DataFrame | None,
) -> Path:
    xdex_block = "<p>Нет данных XDEX.</p>"
    if xdex_summary:
        types_list = "".join(f"<li>{k}: {v}</li>" for k, v in xdex_summary.get("question_types", {}).items())
        xdex_block = f"""
        <ul>
          <li>Вопросов: {xdex_summary.get('questions_total', 0)}</li>
          <li>Допущено: {xdex_summary.get('questions_allowed', 0)}</li>
          <li>Каналов: {xdex_summary.get('channels_total', 0)}</li>
        </ul>
        <p><b>Типы вопросов:</b></p>
        <ul>{types_list}</ul>
        """

    epp_block = "<p>Нет ЭПП результата (не загружен файл метрик).</p>"
    if epp_summary and epp_table is not None:
        table_html = epp_table.to_html(index=False, border=0)
        method = epp_summary.get("method", "model_metrics")
        note = epp_summary.get("note", "")
        timing = epp_summary.get("timing_ms_total")
        score_prob = epp_summary.get("overall_score_probability")
        score_vote = epp_summary.get("overall_score_vote")
        score_prob_html = (
            f"<p><b>Score (probability branch):</b> {float(score_prob):.2%}</p>"
            if score_prob is not None and not pd.isna(score_prob)
            else ""
        )
        score_vote_html = (
            f"<p><b>Score (vote branch):</b> {float(score_vote):.2%}</p>"
            if score_vote is not None and not pd.isna(score_vote)
            else ""
        )
        timing_html = f"<p><b>Время расчета:</b> {timing} мс</p>" if timing is not None else ""
        note_html = f"<p><b>Комментарий:</b> {note}</p>" if note else ""
        epp_block = f"""
        <p><b>Итоговая вероятность:</b> {epp_summary.get('overall_score', 0):.2%}</p>
        {score_prob_html}
        {score_vote_html}
        <p><b>Рекомендация:</b> {epp_summary.get('recommendation', '')}</p>
        <p><b>Количество строк:</b> {epp_summary.get('n_items', 0)}</p>
        <p><b>Метод:</b> {method}</p>
        {timing_html}
        {note_html}
        {table_html}
        """

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Polygraph DSS - Финальный ЭПП отчет</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f9fc; color: #0f172a; }}
    .card {{ background: #fff; border: 1px solid #dbe3ee; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #dbe3ee; padding: 6px 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  <h1>Polygraph DSS - Финальный ЭПП отчет</h1>
  <div class="card">
    <h2>XDEX обзор</h2>
    {xdex_block}
  </div>
  <div class="card">
    <h2>ЭПП результат</h2>
    {epp_block}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_metric(label: str, value: Any) -> None:
    st.markdown(
        f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _first_notna(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def collapse_predictions_to_question_level(pred_df: pd.DataFrame, label_threshold: float = 0.5) -> pd.DataFrame:
    if pred_df.empty or "question_code" not in pred_df.columns:
        return pred_df

    keys = ["question_code"]
    if "test_id" in pred_df.columns:
        keys = ["test_id", "question_code"]

    df = pred_df.copy()
    if "stimulus_idx" in df.columns:
        df = df.sort_values(["question_code", "stimulus_idx"])

    agg: dict[str, Any] = {}
    first_cols = [
        "source_file",
        "source_type",
        "question_raw",
        "question_text",
        "question_type",
        "question_type_code",
        "question_type_ru",
        "is_allowed",
        "ref_c_question_code",
        "waiting_answer_raw",
        "waiting_answer_label",
        "waiting_answer_ru",
    ]
    for col in first_cols:
        if col in df.columns and col not in keys:
            agg[col] = _first_notna

    mean_cols = [
        "deception_probability",
        "raw_score",
        "evidence_score",
        "pair_evidence",
        "baseline_evidence",
        "empirical_scaled",
        "delta_resp_vs_c",
        "delta_eda_vs_c",
        "delta_cardio_vs_c",
        "delta_ppg_vs_c",
        "z_resp",
        "z_eda",
        "z_cardio",
        "z_ppg",
        "sig_resp",
        "sig_eda",
        "sig_cardio",
        "sig_ppg",
    ]
    for col in mean_cols:
        if col in df.columns:
            agg[col] = "mean"

    if agg:
        out = df.groupby(keys, dropna=False, as_index=False).agg(agg)
    else:
        out = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    if "deception_probability" in out.columns:
        out["pred_label"] = (
            pd.to_numeric(out["deception_probability"], errors="coerce").fillna(0.0) >= float(label_threshold)
        ).astype(int)
    elif "pred_label" in out.columns:
        out["pred_label"] = pd.to_numeric(out["pred_label"], errors="coerce").fillna(0).astype(int)

    return out.reset_index(drop=True)


def add_question_level_verdicts(
    pred_df: pd.DataFrame,
    verdict_threshold: float = 0.55,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pred_df.empty or "deception_probability" not in pred_df.columns:
        empty_eval = {
            "n_questions": 0,
            "lie_count": 0,
            "truth_count": 0,
            "lie_share": float("nan"),
            "recommendation": "Insufficient data",
        }
        return pred_df, empty_eval

    out = pred_df.copy()
    out["question_verdict_code"] = (out["deception_probability"] >= float(verdict_threshold)).astype(int)
    out["question_verdict"] = np.where(out["question_verdict_code"] == 1, "Вероятно ложь", "Вероятно правда")

    if "question_type_code" in out.columns:
        type_code = out["question_type_code"].astype(str).str.upper()
        if (type_code == "R").any():
            mask = type_code == "R"
        else:
            mask = type_code != "I"
    elif "question_type" in out.columns:
        mask = out["question_type"].astype(str).str.lower() != "neutral"
    else:
        mask = pd.Series(True, index=out.index)

    if "is_allowed" in out.columns:
        allowed_mask = out["is_allowed"].fillna(True).astype(bool)
        if int((mask & allowed_mask).sum()) > 0:
            mask = mask & allowed_mask

    if int(mask.sum()) == 0:
        mask = pd.Series(True, index=out.index)

    considered = out[mask]
    n_questions = int(len(considered))
    lie_count = int(considered["question_verdict_code"].sum())
    truth_count = int(n_questions - lie_count)
    lie_share = float(lie_count / n_questions) if n_questions else float("nan")

    if np.isnan(lie_share):
        recommendation = "Insufficient data"
    elif lie_share >= 0.65:
        recommendation = "Вероятно ложь"
    elif lie_share <= 0.35:
        recommendation = "Вероятно правда"
    else:
        recommendation = "Требует внимания эксперта"

    eval_summary = {
        "n_questions": n_questions,
        "lie_count": lie_count,
        "truth_count": truth_count,
        "lie_share": lie_share,
        "recommendation": recommendation,
    }
    return out, eval_summary


def _normalize_review_to_label(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"ложь", "lie", "1", "true", "yes", "да"}:
        return 1
    if text in {"правда", "truth", "0", "false", "no", "нет"}:
        return 0
    return None


def save_feedback_and_retrain(review_df: pd.DataFrame) -> dict[str, Any]:
    fb = review_df.copy()
    if "test_id" not in fb.columns:
        if "source_file" in fb.columns and len(fb):
            fb["test_id"] = fb["source_file"].astype(str).map(lambda x: Path(x).stem)
        else:
            fb["test_id"] = "unknown_test"
    if "question_code" not in fb.columns:
        if "question_raw" in fb.columns:
            fb["question_code"] = fb["question_raw"].astype(str)
        else:
            fb["question_code"] = [f"q{i+1}" for i in range(len(fb))]
    if "expert_review" not in fb.columns:
        raise ValueError("Missing required review column `expert_review` for feedback upload.")

    fb["expert_label"] = fb["expert_review"].map(_normalize_review_to_label)
    fb = fb.dropna(subset=["expert_label"])
    if fb.empty:
        raise ValueError("No expert labels selected. Mark at least one question.")

    fb["expert_label"] = fb["expert_label"].astype(int)
    fb["test_id"] = fb["test_id"].astype(str)
    fb["question_code"] = fb["question_code"].astype(str)
    fb = fb.drop_duplicates(subset=["test_id", "question_code"], keep="last")
    fb["created_at"] = datetime.now().isoformat(timespec="seconds")
    keep_cols = [
        c
        for c in [
            "created_at",
            "test_id",
            "question_code",
            "question_raw",
            "question_type",
            "deception_probability",
            "pred_label",
            "expert_review",
            "expert_label",
        ]
        if c in fb.columns
    ]
    fb = fb[keep_cols].copy()

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_PATH.exists():
        old = pd.read_csv(FEEDBACK_PATH)
        if "test_id" in old.columns:
            old["test_id"] = old["test_id"].astype(str)
        if "question_code" in old.columns:
            old["question_code"] = old["question_code"].astype(str)
        if "expert_label" in old.columns:
            old["expert_label"] = pd.to_numeric(old["expert_label"], errors="coerce")
        merged = pd.concat([old, fb], ignore_index=True)
    else:
        merged = fb
    if "test_id" in merged.columns:
        merged["test_id"] = merged["test_id"].astype(str)
    if "question_code" in merged.columns:
        merged["question_code"] = merged["question_code"].astype(str)
    if "expert_label" in merged.columns:
        merged["expert_label"] = pd.to_numeric(merged["expert_label"], errors="coerce")
        merged = merged.dropna(subset=["expert_label"])
        merged["expert_label"] = merged["expert_label"].astype(int).clip(0, 1)
    if {"test_id", "question_code"}.issubset(set(merged.columns)):
        sort_col = "created_at" if "created_at" in merged.columns else merged.columns[0]
        merged = merged.sort_values(sort_col).drop_duplicates(subset=["test_id", "question_code"], keep="last")
    merged.to_csv(FEEDBACK_PATH, index=False, encoding="utf-8-sig")

    summary = train_models(
        features_path=PROJECT_ROOT / "data" / "processed" / "polygram_features.csv",
        models_dir=PROJECT_ROOT / "models",
        reports_dir=PROJECT_ROOT / "reports",
        preferred_label=None,
        feedback_path=FEEDBACK_PATH,
    )
    get_model_bundle.clear()
    return {
        "saved_rows": int(len(fb)),
        "feedback_path": str(FEEDBACK_PATH),
        "train_summary": summary,
    }


def apply_question_based_summary(summary: dict[str, Any], q_eval: dict[str, Any]) -> dict[str, Any]:
    def _rec(prob: float) -> str:
        if prob >= 0.65:
            return "Вероятно ложь"
        if prob <= 0.35:
            return "Вероятно правда"
        return "Требует внимания эксперта"

    out = dict(summary)
    prob_score = float(out.get("overall_score", float("nan")))
    vote_score = float(q_eval.get("lie_share", float("nan")))
    out["overall_score_probability"] = prob_score
    out["overall_score_vote"] = vote_score
    out["recommendation_probability"] = out.get("recommendation", "")

    if not np.isnan(prob_score) and not np.isnan(vote_score):
        final_score = 0.5 * prob_score + 0.5 * vote_score
    elif not np.isnan(vote_score):
        final_score = vote_score
    else:
        final_score = prob_score

    out["overall_score"] = float(final_score)
    out["recommendation_vote"] = q_eval.get("recommendation", "")
    out["recommendation"] = _rec(float(final_score)) if not np.isnan(final_score) else q_eval.get("recommendation", "")
    out["question_vote"] = q_eval
    return out


def run_analysis(input_file: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"run_dir": run_dir, "timing_ms": {}}

    if input_file is None:
        raise ValueError("Input file is required.")

    file_name = str(getattr(input_file, "name", "")).lower()
    is_xdex = file_name.endswith(".xdex") or file_name.endswith(".xdx")
    is_metrics = file_name.endswith(".docx") or file_name.endswith(".csv") or file_name.endswith(".xls") or file_name.endswith(".xlsx")

    if not is_xdex and not is_metrics:
        raise ValueError("Unsupported format. Upload .xdex/.xdx or .docx/.csv/.xls/.xlsx")

    if is_xdex:
        t_xdex0 = time.perf_counter()
        xdex_temp = save_uploaded(input_file, PROJECT_ROOT / "data" / "tmp_uploads")
        xdex_outputs = run_single_xdex(xdex_temp, PROJECT_ROOT)
        xdex_summary = json.loads(Path(xdex_outputs["summary_json"]).read_text(encoding="utf-8"))
        xdex_questions = pd.read_csv(xdex_outputs["questions_csv"])
        t_xdex1 = time.perf_counter()
        result["xdex"] = {"outputs": xdex_outputs, "summary": xdex_summary, "questions": xdex_questions}
        result["timing_ms"]["xdex_overview"] = round((t_xdex1 - t_xdex0) * 1000, 2)

        model_bundle = get_model_bundle()
        xdex_copied = Path(result["xdex"]["outputs"]["xdex_copied_to"])
        model_error: str | None = None
        model_degenerate: str | None = None

        if model_bundle is not None:
            try:
                t_ml0 = time.perf_counter()
                feat_df = features_from_uploaded(input_file)
                t_ml1 = time.perf_counter()
                model_threshold = float(model_bundle.get("threshold", 0.5))
                pred_df = predict_df(feat_df, model_bundle)
                pred_df = collapse_predictions_to_question_level(pred_df, label_threshold=model_threshold)
                t_ml2 = time.perf_counter()
                if "deception_probability" in pred_df.columns and len(pred_df):
                    probs = pd.to_numeric(pred_df["deception_probability"], errors="coerce").fillna(0.0)
                    if (
                        probs.nunique(dropna=False) <= 1
                        or float(probs.std()) < 1e-9
                        or float(probs.max()) < 0.20
                        or float(probs.min()) > 0.85
                    ):
                        model_degenerate = "ML probabilities look weakly separated for this XDEX."

                epp_summary = summarize_predictions(pred_df)
                pred_df, q_eval = add_question_level_verdicts(pred_df, verdict_threshold=model_threshold)
                epp_summary = apply_question_based_summary(epp_summary, q_eval)
                epp_summary["method"] = "model_metrics"
                epp_summary["timing_ms_total"] = round((t_ml2 - t_ml0) * 1000, 2)
                epp_summary["timing_ms_split"] = {
                    "prepare_features": round((t_ml1 - t_ml0) * 1000, 2),
                    "model_predict": round((t_ml2 - t_ml1) * 1000, 2),
                }
                epp_summary["note"] = "XDEX was processed through the same ML pipeline as DOCX/CSV/XLSX."
                pred_path = run_dir / "predictions.csv"
                sum_path = run_dir / "summary.json"
                pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
                sum_path.write_text(json.dumps(epp_summary, ensure_ascii=False, indent=2), encoding="utf-8")
                result["epp"] = {
                    "pred_df": pred_df,
                    "summary": epp_summary,
                    "pred_path": pred_path,
                    "summary_path": sum_path,
                    "source": "model_metrics",
                }
                result["epp_model_debug"] = {
                    "pred_df": pred_df,
                    "summary": epp_summary,
                    "degenerate_note": model_degenerate,
                }
            except Exception as exc:
                model_error = str(exc)

        if "epp" not in result:
            pred_df, epp_summary, epp_details = run_xdex_epp_detailed(xdex_copied)
            pred_df = collapse_predictions_to_question_level(pred_df, label_threshold=0.55)
            if "deception_probability" in pred_df.columns:
                pred_df["pred_label"] = (
                    pd.to_numeric(pred_df["deception_probability"], errors="coerce").fillna(0.0) >= 0.55
                ).astype(int)
            pred_df, q_eval = add_question_level_verdicts(pred_df, verdict_threshold=0.55)
            epp_summary = apply_question_based_summary(epp_summary, q_eval)
            if model_error:
                epp_summary["note"] = (
                    f"{epp_summary.get('note', '')} ML mode is not available for this XDEX: {model_error}".strip()
                )
            if model_degenerate:
                epp_summary["note"] = (
                    f"{epp_summary.get('note', '')} {model_degenerate}".strip()
                )
            pred_path = run_dir / "predictions.csv"
            sum_path = run_dir / "summary.json"
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
            sum_path.write_text(json.dumps(epp_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            result["epp"] = {
                "pred_df": pred_df,
                "summary": epp_summary,
                "pred_path": pred_path,
                "summary_path": sum_path,
                "source": "xdex_direct",
                "details": epp_details,
            }
    elif is_metrics:
        model_bundle = get_model_bundle()
        if model_bundle is None:
            raise ValueError("Модель `models/best_model.pkl` не найдена.")
        t_ml0 = time.perf_counter()
        feat_df = features_from_uploaded(input_file)
        t_ml1 = time.perf_counter()
        model_threshold = float(model_bundle.get("threshold", 0.5))
        pred_df = predict_df(feat_df, model_bundle)
        pred_df = collapse_predictions_to_question_level(pred_df, label_threshold=model_threshold)
        t_ml2 = time.perf_counter()
        epp_summary = summarize_predictions(pred_df)
        pred_df, q_eval = add_question_level_verdicts(pred_df, verdict_threshold=model_threshold)
        epp_summary = apply_question_based_summary(epp_summary, q_eval)
        epp_summary["method"] = "model_metrics"
        epp_summary["timing_ms_total"] = round((t_ml2 - t_ml0) * 1000, 2)
        epp_summary["timing_ms_split"] = {
            "prepare_features": round((t_ml1 - t_ml0) * 1000, 2),
            "model_predict": round((t_ml2 - t_ml1) * 1000, 2),
        }
        pred_path = run_dir / "predictions.csv"
        sum_path = run_dir / "summary.json"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        sum_path.write_text(json.dumps(epp_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        result["epp"] = {
            "pred_df": pred_df,
            "summary": epp_summary,
            "pred_path": pred_path,
            "summary_path": sum_path,
            "source": "model_metrics",
        }

    xdex_summary = result.get("xdex", {}).get("summary")
    epp_summary = result.get("epp", {}).get("summary")
    epp_table = None
    if "epp" in result:
        pred_df = result["epp"]["pred_df"].copy()
        if "xdex" in result:
            qdf = result["xdex"].get("questions")
            if isinstance(qdf, pd.DataFrame) and "question_code" in qdf.columns:
                candidate_cols = [
                    c
                    for c in ["question_code", "waiting_answer_label", "waiting_answer_ru", "waiting_answer_raw"]
                    if c in qdf.columns
                ]
                missing_waiting_cols = [
                    c for c in ["waiting_answer_label", "waiting_answer_ru", "waiting_answer_raw"] if c not in pred_df.columns
                ]
                merge_cols = ["question_code"] + [c for c in missing_waiting_cols if c in candidate_cols]
                if len(merge_cols) >= 2:
                    pred_df = pred_df.merge(
                        qdf[merge_cols].drop_duplicates(subset=["question_code"]),
                        on="question_code",
                        how="left",
                    )
        result["epp"]["pred_df"] = pred_df
        if "question_raw" not in pred_df.columns and "question_text" in pred_df.columns:
            pred_df["question_raw"] = pred_df["question_text"]
        cols = [c for c in ["question_code", "question_raw", "deception_probability", "pred_label"] if c in pred_df.columns]
        epp_table = pred_df[cols].copy()

    final_html_path = run_dir / "final_epp_report.html"
    build_final_html_report(final_html_path, xdex_summary, epp_summary, epp_table)
    result["final_html"] = final_html_path
    result["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


def main() -> None:
    st.set_page_config(page_title="Polygraph DSS", layout="wide")
    inject_styles()
    show_header()

    st.markdown(
        """
        <div class="card">
          <b>Single input file:</b><br/>
          Upload one of: <b>XDEX/XDX</b> or <b>DOCX/CSV/XLSX</b>, then click run.<br/>
          Or click <b>Демонстрация</b> to load a built-in example with a full question route.
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_file = st.file_uploader(
        "Загрузите файл (.xdex/.xdx/.docx/.csv/.xls/.xlsx)",
        type=["xdex", "xdx", "docx", "csv", "xls", "xlsx"],
        key="unified_input",
    )

    action_col, demo_col = st.columns([2, 1])
    with action_col:
        run_clicked = st.button("Запустить полный анализ", use_container_width=True)
    with demo_col:
        demo_clicked = st.button("Демонстрация", use_container_width=True)

    if demo_clicked:
        if not DEMO_FILE_PATH.exists():
            st.error(f"Демо-файл не найден: {DEMO_FILE_PATH}")
        else:
            with st.spinner("Загружаю демонстрационный кейс..."):
                try:
                    st.session_state["analysis_result"] = run_analysis(DemoUploadedFile(DEMO_FILE_PATH))
                    st.session_state["analysis_source_name"] = DEMO_FILE_PATH.name
                    st.success(f"Демо-кейс загружен: {DEMO_FILE_PATH.name}")
                except Exception as exc:
                    st.error(f"Ошибка демонстрации: {exc}")

    if run_clicked:
        if input_file is None:
            st.error("Загрузите файл.")
        else:
            with st.spinner("Выполняю анализ..."):
                try:
                    st.session_state["analysis_result"] = run_analysis(input_file)
                    st.session_state["analysis_source_name"] = input_file.name
                except Exception as exc:
                    st.error(f"Ошибка анализа: {exc}")

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Загрузите файл и нажмите кнопку анализа или используйте демонстрационный кейс.")
        return

    source_name = st.session_state.get("analysis_source_name")
    if source_name:
        st.caption(f"Текущий кейс: {source_name}")

    if "epp" in result:
        st.markdown("## ЭПП результат")
        summary = result["epp"]["summary"]
        source = result["epp"].get("source", "unknown")
        p = float(summary["overall_score"])
        rec = summary["recommendation"]
        if p >= 0.65:
            color = "#dc2626"
            bg = "#fee2e2"
        elif p <= 0.35:
            color = "#15803d"
            bg = "#dcfce7"
        else:
            color = "#0369a1"
            bg = "#e0f2fe"

        st.markdown(
            f"""
            <div class="epp-box" style="background:{bg}; border-color:{color};">
              <div style="font-size:12px; color:#334155;">Итоговая вероятность</div>
              <div style="font-size:34px; font-weight:800; color:{color};">{p:.2%}</div>
              <div style="font-size:18px; font-weight:700; color:#0f172a;">Рекомендация: {rec}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        q_vote = summary.get("question_vote", {})
        if isinstance(q_vote, dict) and q_vote.get("n_questions", 0):
            q1, q2, q3 = st.columns(3)
            q1.metric("Questions used", int(q_vote.get("n_questions", 0)))
            q2.metric("Lie votes", int(q_vote.get("lie_count", 0)))
            q3.metric("Truth votes", int(q_vote.get("truth_count", 0)))
            st.caption(
                f"Vote score (lie share by question): {float(q_vote.get('lie_share', 0.0)):.2%}."
            )

        prob_score = summary.get("overall_score_probability")
        prob_rec = summary.get("recommendation_probability")
        if prob_score is not None and not pd.isna(prob_score):
            st.caption(f"Probability score: {float(prob_score):.2%} ({prob_rec}).")

        vote_score = summary.get("overall_score_vote")
        vote_rec = summary.get("recommendation_vote")
        if vote_score is not None and not pd.isna(vote_score):
            st.caption(f"Vote-based recommendation: {float(vote_score):.2%} ({vote_rec}).")
        st.caption("Final score = 50% probability score + 50% vote score.")

        if source == "xdex_direct":
            st.caption("Источник расчета: прямой скоринг из XDEX (`xdex_direct_scoring_v2_pairwise`).")
            st.warning("ML mode failed for this run, fallback to direct XDEX scoring is used.")
            note = summary.get("note")
            if note:
                st.info(note)
            timing = result["epp"].get("details", {}).get("timing_ms", {})
            if timing:
                st.markdown("### Тайминг обработки")
                tm1, tm2, tm3, tm4, tm5 = st.columns(5)
                tm1.metric("JSON", f"{timing.get('read_test_json', 0)} мс")
                tm2.metric("Mapping", f"{timing.get('build_maps', 0)} мс")
                tm3.metric("Scores", f"{timing.get('extract_scores', 0)} мс")
                tm4.metric("Scoring", f"{timing.get('compute_probabilities', 0)} мс")
                tm5.metric("Всего", f"{timing.get('total', 0)} мс")
        elif source == "model_metrics":
            st.caption("Source: ML model on tabular metrics (DOCX/CSV/XLSX, and XDEX when tabular parsing succeeds).")
            split = summary.get("timing_ms_split", {})
            if split:
                st.markdown("### Тайминг обработки")
                tm1, tm2, tm3 = st.columns(3)
                tm1.metric("Подготовка признаков", f"{split.get('prepare_features', 0)} мс")
                tm2.metric("Предсказание модели", f"{split.get('model_predict', 0)} мс")
                tm3.metric("Всего", f"{summary.get('timing_ms_total', 0)} мс")

        pred_df = result["epp"]["pred_df"].copy()
        if "question_raw" not in pred_df.columns and "question_text" in pred_df.columns:
            pred_df["question_raw"] = pred_df["question_text"]
        if "question_type" not in pred_df.columns and "question_type_ru" in pred_df.columns:
            pred_df["question_type"] = pred_df["question_type_ru"]

        if source == "xdex_direct":
            details = result["epp"].get("details", {})
            qmap_df = details.get("question_map_df")
            if isinstance(qmap_df, pd.DataFrame) and not qmap_df.empty:
                with st.expander("Как система поняла тип каждого вопроса"):
                    st.write(
                        "Тип берется из `questionList.questions[].type` в XDEX: "
                        "`I -> Нв`, `R -> Пв`, `C -> Вл`, `SR -> Жв`, `S -> Св`."
                    )
                    st.dataframe(qmap_df, use_container_width=True, hide_index=True)

            explain_cols = [
                c
                for c in [
                    "question_code",
                    "question_type_ru",
                    "ref_c_question_code",
                    "sig_resp",
                    "sig_eda",
                    "sig_cardio",
                    "sig_ppg",
                    "z_resp",
                    "z_eda",
                    "z_cardio",
                    "z_ppg",
                    "delta_resp_vs_c",
                    "delta_eda_vs_c",
                    "delta_cardio_vs_c",
                    "delta_ppg_vs_c",
                    "pair_evidence",
                    "baseline_evidence",
                    "empirical_scaled",
                    "evidence_score",
                    "raw_score",
                    "deception_probability",
                ]
                if c in pred_df.columns
            ]
            if explain_cols:
                with st.expander("Как считалась вероятность (формула и вклад)"):
                    st.write(
                        "Формула v2: "
                        "`pair = 0.45*О”EDA + 0.25*О”Cardio + 0.20*О”Resp + 0.10*О”PPG`, "
                        "где `Δ = z(вопрос) - z(ближайший C)`. "
                        "Для `R`: `evidence = pair + 0.20*empirical`; "
                        "для `C`: `evidence = -pair`; "
                        "для остальных типов: `evidence = 0.35*baseline`. "
                        "Вероятность: `p = sigmoid(1.45*evidence - 0.10)`."
                    )
                    st.dataframe(pred_df[explain_cols], use_container_width=True, hide_index=True)

        if "question_code" in pred_df.columns and "deception_probability" in pred_df.columns:
            st.markdown("### Вероятность по вопросам")
            chart_df = pred_df[["question_code", "deception_probability"]].dropna().set_index("question_code")
            st.bar_chart(chart_df)

        show_cols = [
            c
            for c in [
                "test_id",
                "question_code",
                "question_raw",
                "question_type",
                "deception_probability",
                "question_verdict",
                "question_verdict_code",
                "pred_label",
                "waiting_answer_label",
                "waiting_answer_ru",
            ]
            if c in pred_df.columns
        ]
        st.markdown("### Таблица предсказаний")
        st.dataframe(pred_df[show_cols], use_container_width=True, hide_index=True)

        if "deception_probability" in pred_df.columns and len(pred_df):
            pmin = float(pred_df["deception_probability"].min())
            pmax = float(pred_df["deception_probability"].max())
            pmean = float(pred_df["deception_probability"].mean())
            st.caption(f"Probability diagnostics: min={pmin:.6f}, max={pmax:.6f}, mean={pmean:.6f}")
            if pmax <= 1e-9:
                st.warning(
                    "All probabilities are ~0 for this file. This usually means the current model "
                    "has collapsed to class 0 on this feature pattern (not a UI bug). "
                    "Use expert review below and upload feedback to retrain."
                )

        st.markdown("### Expert Review For Retraining")
        st.caption(
            "Set your labels for each question, then click upload. "
            "These labels will be saved and used in the next retraining."
        )

        review_cols = [
            c
            for c in [
                "test_id",
                "question_code",
                "question_raw",
                "question_type",
                "deception_probability",
                "pred_label",
                "waiting_answer_label",
                "waiting_answer_raw",
                "waiting_answer_ru",
            ]
            if c in pred_df.columns
        ]
        review_df = pred_df[review_cols].copy()
        if "test_id" not in review_df.columns:
            default_test_id = None
            if "xdex" in result:
                try:
                    default_test_id = Path(result["xdex"]["outputs"]["xdex_copied_to"]).stem
                except Exception:
                    default_test_id = None
            if default_test_id is None and "source_file" in pred_df.columns and len(pred_df):
                default_test_id = Path(str(pred_df["source_file"].iloc[0])).stem
            if default_test_id is None:
                default_test_id = "unknown_test"
            review_df["test_id"] = str(default_test_id)
        else:
            review_df["test_id"] = review_df["test_id"].astype(str)

        if "question_code" not in review_df.columns:
            if "question_raw" in review_df.columns:
                review_df["question_code"] = (
                    review_df["question_raw"].astype(str).fillna("").replace({"": None})
                )
            else:
                review_df["question_code"] = [f"q{i+1}" for i in range(len(review_df))]
        review_df["question_code"] = review_df["question_code"].astype(str)
        if "waiting_answer_label" in review_df.columns:
            wa_lbl = review_df["waiting_answer_label"].astype(str).str.strip()
            review_df["expert_review"] = np.where(wa_lbl.eq("Lie"), "Lie", np.where(wa_lbl.eq("Truth"), "Truth", ""))
            if "pred_label" in review_df.columns:
                fallback = np.where(review_df["pred_label"].astype(int) == 1, "Lie", "Truth")
                review_df["expert_review"] = np.where(review_df["expert_review"].eq(""), fallback, review_df["expert_review"])
            else:
                review_df["expert_review"] = np.where(review_df["expert_review"].eq(""), "Truth", review_df["expert_review"])
        elif "waiting_answer_raw" in review_df.columns:
            wa_raw = review_df["waiting_answer_raw"].astype(str).str.strip().str.lower()
            review_df["expert_review"] = np.where(wa_raw.eq("no"), "Lie", np.where(wa_raw.eq("yes"), "Truth", ""))
            if "pred_label" in review_df.columns:
                fallback = np.where(review_df["pred_label"].astype(int) == 1, "Lie", "Truth")
                review_df["expert_review"] = np.where(review_df["expert_review"].eq(""), fallback, review_df["expert_review"])
            else:
                review_df["expert_review"] = np.where(review_df["expert_review"].eq(""), "Truth", review_df["expert_review"])
        elif "pred_label" in review_df.columns:
            review_df["expert_review"] = np.where(review_df["pred_label"].astype(int) == 1, "Lie", "Truth")
        else:
            review_df["expert_review"] = "Truth"

        edited_review_df = st.data_editor(
            review_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "expert_review": st.column_config.SelectboxColumn(
                    "Expert Label",
                    options=["Truth", "Lie"],
                    required=True,
                )
            },
            key=f"expert_review_editor_{result['run_dir'].name}",
        )

        if st.button("Upload Review And Retrain", use_container_width=True):
            try:
                with st.spinner("Saving review and retraining model..."):
                    feedback_result = save_feedback_and_retrain(edited_review_df)
                st.success(
                    f"Done. Saved labels: {feedback_result['saved_rows']}. "
                    f"File: {feedback_result['feedback_path']}"
                )
                st.json(feedback_result["train_summary"])
            except Exception as exc:
                st.error(f"Failed to upload review/retrain: {exc}")

        d1, d2 = st.columns(2)
        d1.download_button(
            "Скачать predictions.csv",
            data=result["epp"]["pred_path"].read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "Скачать summary.json",
            data=result["epp"]["summary_path"].read_bytes(),
            file_name="summary.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.warning("EPP result is not built. Upload XDEX or metrics file and run analysis.")

    if "xdex" in result:
        st.markdown("## XDEX обзор")
        xd = result["xdex"]
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric("Всего вопросов", xd["summary"].get("questions_total", 0))
        with m2:
            render_metric("Допущено в тест", xd["summary"].get("questions_allowed", 0))
        with m3:
            render_metric("Каналы", xd["summary"].get("channels_total", 0))

        cl, cr = st.columns([2, 1], gap="large")
        with cl:
            st.markdown("### Маршрут вопросов")
            st.image(xd["outputs"]["question_route_png"], use_container_width=True)
            st.markdown("### Таблица вопросов")
            st.dataframe(xd["questions"], use_container_width=True, hide_index=True)
        with cr:
            st.markdown("### Сводка XDEX")
            st.json(xd["summary"])
            dx1, dx2, dx3 = st.columns(3)
            dx1.download_button("HTML", Path(xd["outputs"]["report_html"]).read_bytes(), file_name="xdex_report.html", mime="text/html", use_container_width=True)
            dx2.download_button("CSV", Path(xd["outputs"]["questions_csv"]).read_bytes(), file_name="questions.csv", mime="text/csv", use_container_width=True)
            dx3.download_button("PNG", Path(xd["outputs"]["question_route_png"]).read_bytes(), file_name="route.png", mime="image/png", use_container_width=True)

    st.markdown("## Финальный HTML отчет")
    st.download_button(
        "Скачать final_epp_report.html",
        data=Path(result["final_html"]).read_bytes(),
        file_name="final_epp_report.html",
        mime="text/html",
        use_container_width=True,
    )
    st.success(f"Папка запуска: {result['run_dir']}")


if __name__ == "__main__":
    main()

