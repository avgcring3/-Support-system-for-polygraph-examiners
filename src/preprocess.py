from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .parser import load_raw_records
except ImportError:
    from parser import load_raw_records


PARAMETER_MAP = {
    "\u0414\u042b\u0425\u0410\u041d\u0418\u0415": "breath",
    "BREATH": "breath",
    "\u041a\u0413\u0420": "gsr",
    "EDA": "gsr",
    "GSR": "gsr",
    "\u0410\u0414": "bp",
    "AR": "bp",
    "BP": "bp",
    "\u041f\u0413": "ppg",
    "PLE": "ppg",
    "PPG": "ppg",
}


def normalize_parameter_name(value: str) -> str:
    raw = str(value).strip().upper()
    return PARAMETER_MAP.get(raw, raw.lower().replace(" ", "_"))


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator_safe = denominator.replace({0.0: np.nan})
    return numerator / denominator_safe


def _first_notna(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def build_long_dataset(docx_dir: Path, xdex_dir: Path | None = None) -> pd.DataFrame:
    df = load_raw_records(docx_dir=docx_dir, xdex_dir=xdex_dir)
    if df.empty:
        return df

    df = df.copy()
    df["parameter_norm"] = df["parameter"].map(normalize_parameter_name)
    df = df[~df["parameter_norm"].isin({"", "none", "nan", "параметр", "parameter"})].copy()
    df["question_code"] = df["question_code"].fillna(
        df.groupby(["test_id", "question_raw"], dropna=False).ngroup().astype(str).radd("q")
    )
    df["question_type"] = df["question_type"].fillna("other")
    df["score_result"] = pd.to_numeric(df["score_result"], errors="coerce")
    df["metric_i"] = pd.to_numeric(df["metric_i"], errors="coerce")
    df["metric_ii"] = pd.to_numeric(df["metric_ii"], errors="coerce")

    # Удаляем дубли, если один и тот же канал в одном вопросе встретился несколько раз.
    group_cols = [
        c
        for c in [
            "source_file",
            "source_type",
            "test_id",
            "question_raw",
            "question_code",
            "question_text",
            "question_type_code",
            "question_type",
            "is_allowed",
            "waiting_answer_raw",
            "waiting_answer_label",
            "waiting_answer_ru",
            "parameter",
            "parameter_norm",
        ]
        if c in df.columns
    ]
    df = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            metric_i=("metric_i", "mean"),
            metric_ii=("metric_ii", "mean"),
            score_result=("score_result", "max"),
        )
        .reset_index(drop=True)
    )
    return df


def build_feature_dataset(long_df: pd.DataFrame, weak_label_threshold: float = 0.55) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    id_cols = ["test_id", "source_file", "source_type", "question_code", "question_raw", "question_text", "question_type"]
    score_df = long_df.groupby(id_cols, dropna=False, as_index=False)["score_result"].max()
    meta_cols = [
        c
        for c in ["question_type_code", "is_allowed", "waiting_answer_raw", "waiting_answer_label", "waiting_answer_ru"]
        if c in long_df.columns
    ]
    if meta_cols:
        meta_df = long_df.groupby(id_cols, dropna=False, as_index=False)[meta_cols].agg(_first_notna)
        score_df = score_df.merge(meta_df, on=id_cols, how="left")

    metric_i = long_df.pivot_table(
        index=id_cols,
        columns="parameter_norm",
        values="metric_i",
        aggfunc="mean",
    )
    metric_i.columns = [f"i_{col}" for col in metric_i.columns]
    metric_i = metric_i.reset_index()

    metric_ii = long_df.pivot_table(
        index=id_cols,
        columns="parameter_norm",
        values="metric_ii",
        aggfunc="mean",
    )
    metric_ii.columns = [f"ii_{col}" for col in metric_ii.columns]
    metric_ii = metric_ii.reset_index()

    features = score_df.merge(metric_i, on=id_cols, how="left").merge(metric_ii, on=id_cols, how="left")

    for base_name in set(col.replace("i_", "") for col in features.columns if col.startswith("i_")):
        i_col = f"i_{base_name}"
        ii_col = f"ii_{base_name}"
        if i_col in features.columns and ii_col in features.columns:
            features[f"ratio_ii_i_{base_name}"] = _safe_div(features[ii_col], features[i_col])

    # Нормализация относительно нейтральных вопросов внутри каждого теста.
    i_cols = [col for col in features.columns if col.startswith("i_")]
    for col in i_cols:
        neutral_mean = (
            features[features["question_type"] == "neutral"]
            .groupby("test_id")[col]
            .mean()
            .rename(f"{col}_neutral_mean")
        )
        neutral_std = (
            features[features["question_type"] == "neutral"]
            .groupby("test_id")[col]
            .std()
            .replace({0.0: np.nan})
            .rename(f"{col}_neutral_std")
        )
        features = features.join(neutral_mean, on="test_id")
        features = features.join(neutral_std, on="test_id")
        features[f"z_{col}_vs_neutral"] = (
            (features[col] - features[f"{col}_neutral_mean"]) / features[f"{col}_neutral_std"]
        )

    qtype = features["question_type"].astype(str).str.lower()
    features["is_control_q"] = (qtype == "control").astype(int)
    features["is_probable_lie_q"] = (qtype == "probable_lie").astype(int)
    features["is_neutral_q"] = (qtype == "neutral").astype(int)
    features["is_sacrifice_q"] = (qtype == "sacrifice").astype(int)

    # Weak label comes only from score_result threshold; missing score remains missing label.
    score_numeric = pd.to_numeric(features["score_result"], errors="coerce")
    weak_label = pd.Series(pd.NA, index=features.index, dtype="Int64")
    has_score = score_numeric.notna()
    weak_label.loc[has_score] = (score_numeric.loc[has_score] >= weak_label_threshold).astype(int)

    # Question-template label comes from declared expected answer in question metadata.
    question_template_label = pd.Series(pd.NA, index=features.index, dtype="Int64")
    if "waiting_answer_label" in features.columns:
        wa_label = features["waiting_answer_label"].astype(str).str.strip().str.lower()
        mapped = np.where(wa_label.eq("lie"), 1, np.where(wa_label.eq("truth"), 0, pd.NA))
        question_template_label = pd.Series(mapped, index=features.index, dtype="Int64")
    if "waiting_answer_raw" in features.columns:
        wa_raw = features["waiting_answer_raw"].astype(str).str.strip().str.lower()
        mapped_raw = np.where(wa_raw.eq("no"), 1, np.where(wa_raw.eq("yes"), 0, pd.NA))
        mapped_raw = pd.Series(mapped_raw, index=features.index, dtype="Int64")
        question_template_label = question_template_label.where(question_template_label.notna(), mapped_raw)

    # Bootstrap label for training: question-template first, then weak score-based.
    label_bootstrap = question_template_label.where(question_template_label.notna(), weak_label).astype("Int64")
    label_source = pd.Series(
        np.where(
            question_template_label.notna(),
            "question-template",
            np.where(weak_label.notna(), "weak", "missing"),
        ),
        index=features.index,
        dtype="object",
    )

    features["label_weak"] = weak_label.astype("Int64")
    features["label_question_template"] = question_template_label.astype("Int64")
    features["label_bootstrap"] = label_bootstrap
    features["label_source"] = label_source
    return features


def save_processed_datasets(
    long_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    processed_dir: Path,
) -> tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    long_path = processed_dir / "polygram_long.csv"
    feature_path = processed_dir / "polygram_features.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
    return long_path, feature_path


def run_preprocess(
    docx_dir: Path,
    xdex_dir: Path | None,
    processed_dir: Path,
    weak_label_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = build_long_dataset(docx_dir=docx_dir, xdex_dir=xdex_dir)
    feature_df = build_feature_dataset(long_df=long_df, weak_label_threshold=weak_label_threshold)
    save_processed_datasets(long_df=long_df, feature_df=feature_df, processed_dir=processed_dir)
    return long_df, feature_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed polygraph datasets from raw DOCX/XDEX exports.")
    parser.add_argument("--docx-dir", type=Path, default=Path("data/raw/docx"))
    parser.add_argument("--xdex-dir", type=Path, default=Path("data/raw/xdex"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--weak-threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xdex_dir = args.xdex_dir if args.xdex_dir.exists() else None
    long_df, feature_df = run_preprocess(
        docx_dir=args.docx_dir,
        xdex_dir=xdex_dir,
        processed_dir=args.processed_dir,
        weak_label_threshold=args.weak_threshold,
    )
    print(f"[preprocess] long rows: {len(long_df)}")
    print(f"[preprocess] feature rows: {len(feature_df)}")
    print(f"[preprocess] saved to: {args.processed_dir.resolve()}")


if __name__ == "__main__":
    main()

