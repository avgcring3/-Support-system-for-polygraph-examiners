from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .metrics import compute_binary_metrics
except ImportError:
    from metrics import compute_binary_metrics


DEFAULT_EXCLUDED_COLS = {
    "label",
    "label_weak",
    "label_question_template",
    "label_bootstrap",
    "label_hybrid",
    "label_train",
    "label_source",
    "label_source_train",
    "score_result",
    "label_expert",
    "expert_label",
    "expert_review",
    "created_at",
    "source_file",
    "source_type",
    "test_id",
    "question_code",
    "question_raw",
    "question_text",
    "question_type",
}


def detect_label_column(df: pd.DataFrame, preferred_label: str | None = None) -> str:
    if preferred_label and preferred_label in df.columns:
        return preferred_label
    for col in ["label", "label_train", "label_bootstrap", "label_hybrid", "label_question_template", "label_weak"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Label column is missing. Expected one of: label, label_train, label_bootstrap, label_question_template, label_weak."
    )


def select_feature_columns(df: pd.DataFrame, label_col: str, excluded: set[str] | None = None) -> list[str]:
    excluded = set(excluded or set())
    excluded.add(label_col)
    candidates = [col for col in df.columns if col not in excluded]
    numeric_cols = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    min_non_null = max(5, int(0.15 * len(df)))
    return [col for col in numeric_cols if df[col].notna().sum() >= min_non_null]


def build_models() -> dict[str, Pipeline]:
    base_rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
        min_samples_leaf=2,
    )
    return {
        "logreg": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", base_rf),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "extra_trees_wide": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=900,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=1,
                        max_features=None,
                    ),
                ),
            ]
        ),
        "random_forest_wide": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=900,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=1,
                        max_features="sqrt",
                    ),
                ),
            ]
        ),
        "rf_calibrated": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=base_rf,
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def _build_cv(
    y: pd.Series,
    groups: pd.Series | None = None,
) -> tuple[StratifiedKFold | StratifiedGroupKFold, int, str]:
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        return StratifiedKFold(n_splits=2, shuffle=True, random_state=42), 1, "invalid"

    if groups is not None:
        grp = groups.astype(str)
        if grp.nunique() >= n_splits:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits, "stratified_group"
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits, "stratified"


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        mean_conf = float(np.mean(y_prob[mask]))
        mean_acc = float(np.mean(y_true[mask]))
        ece += abs(mean_conf - mean_acc) * (float(np.sum(mask)) / len(y_true))
    return float(ece)


def _mean_ci(values: list[float], z_score: float = 1.96) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0.0,
        }
    mean = float(np.mean(arr))
    if arr.size < 2:
        return {
            "mean": mean,
            "std": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "n": float(arr.size),
        }
    std = float(np.std(arr, ddof=1))
    half = float(z_score * std / np.sqrt(arr.size))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n": float(arr.size),
    }


def _compute_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(_expected_calibration_error(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["pr_auc"] = float("nan")
    return out


def _summarize_fold_metrics(fold_df: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ["f1", "pr_auc", "brier", "ece", "accuracy", "precision", "recall"]:
        ci = _mean_ci(fold_df[metric].astype(float).tolist())
        result[f"cv_{metric}"] = float(ci["mean"])
        result[f"cv_{metric}_std"] = float(ci["std"])
        result[f"cv_{metric}_ci_low"] = float(ci["ci_low"])
        result[f"cv_{metric}_ci_high"] = float(ci["ci_high"])
    return result


def evaluate_model_with_group_cv_detailed(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
) -> dict[str, Any]:
    cv, n_splits, cv_kind = _build_cv(y=y, groups=groups)
    if n_splits < 2:
        return {"summary": {}, "fold_metrics": pd.DataFrame(), "oof_proba": np.array([]), "oof_pred": np.array([])}

    x_df = x.reset_index(drop=True)
    y_arr = y.astype(int).to_numpy()
    groups_arr = groups.astype(str).to_numpy() if groups is not None else None

    splitter = cv.split(x_df, y_arr, groups_arr if cv_kind == "stratified_group" else None)
    oof_proba = np.full(len(x_df), np.nan, dtype=float)
    oof_pred = np.full(len(x_df), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter, start=1):
        model_fold = clone(model)
        model_fold.fit(x_df.iloc[train_idx], y_arr[train_idx])
        prob = model_fold.predict_proba(x_df.iloc[valid_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_proba[valid_idx] = prob
        oof_pred[valid_idx] = pred

        fold_metrics = _compute_eval_metrics(y_arr[valid_idx], pred, prob)
        fold_rows.append(
            {
                "fold": int(fold_idx),
                "n_valid": int(len(valid_idx)),
                "positive_rate": float(np.mean(y_arr[valid_idx])),
                **fold_metrics,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = _summarize_fold_metrics(fold_df)
    summary["cv_folds"] = float(n_splits)
    summary["cv_kind"] = cv_kind
    return {
        "summary": summary,
        "fold_metrics": fold_df,
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
    }


def compute_optimal_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    # Conservative search range to avoid pathological thresholds at boundaries.
    candidates = np.linspace(0.10, 0.90, 161)
    best_thr = 0.5
    best_f1 = -1.0
    y_int = y_true.astype(int).to_numpy()
    for thr in candidates:
        pred = (y_score >= thr).astype(int)
        val = f1_score(y_int, pred, zero_division=0)
        if val > best_f1:
            best_f1 = float(val)
            best_thr = float(thr)
    return best_thr, best_f1


def split_fixed_group_holdout(
    df: pd.DataFrame,
    label_col: str,
    group_col: str = "test_id",
    holdout_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[set[str], set[str]]:
    if group_col not in df.columns:
        raise ValueError(f"Group column `{group_col}` is missing.")
    group_df = (
        df[[group_col, label_col]]
        .dropna(subset=[label_col])
        .copy()
        .groupby(group_col, as_index=False)[label_col]
        .mean()
    )
    group_df[group_col] = group_df[group_col].astype(str)
    if len(group_df) < 4:
        raise ValueError("At least 4 unique groups are required for fixed group holdout.")

    group_df["strata"] = (group_df[label_col] >= 0.5).astype(int)
    n_groups = len(group_df)
    test_size = int(round(n_groups * holdout_frac))
    test_size = max(2, test_size)
    test_size = min(test_size, n_groups - 2)

    splitter = StratifiedShuffleSplit(n_splits=50, test_size=test_size, random_state=random_state)
    x_dummy = np.zeros((n_groups, 1), dtype=float)
    y_strata = group_df["strata"].to_numpy()
    all_groups = group_df[group_col].to_numpy()

    for train_idx, holdout_idx in splitter.split(x_dummy, y_strata):
        train_groups = set(all_groups[train_idx].tolist())
        holdout_groups = set(all_groups[holdout_idx].tolist())
        if not train_groups or not holdout_groups:
            continue

        train_mask = df[group_col].astype(str).isin(train_groups)
        holdout_mask = df[group_col].astype(str).isin(holdout_groups)
        y_train = df.loc[train_mask, label_col].dropna().astype(int)
        y_holdout = df.loc[holdout_mask, label_col].dropna().astype(int)
        if y_train.nunique() < 2 or y_holdout.nunique() < 2:
            continue
        return train_groups, holdout_groups

    raise ValueError("Failed to build a fixed holdout split with both classes in train and holdout.")


def extract_feature_importance(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        values = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        values = np.zeros(len(feature_cols))
    fi = pd.DataFrame({"feature": feature_cols, "importance": values})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


def predict_proba_from_bundle(
    bundle: dict[str, Any],
    x: pd.DataFrame,
) -> np.ndarray:
    if "models" in bundle and isinstance(bundle["models"], list):
        numer = np.zeros(len(x), dtype=float)
        denom = 0.0
        for item in bundle["models"]:
            mdl = item["model"]
            w = float(item.get("weight", 1.0))
            numer += w * mdl.predict_proba(x)[:, 1]
            denom += w
        if denom <= 0:
            denom = float(len(bundle["models"])) if bundle["models"] else 1.0
        return numer / denom

    model: Pipeline = bundle["model"]
    return model.predict_proba(x)[:, 1]


def _build_source_type_breakdown(
    eval_df: pd.DataFrame,
    *,
    model_name: str,
    eval_split: str,
) -> pd.DataFrame:
    if eval_df.empty or "source_type" not in eval_df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for source_type, part in eval_df.groupby("source_type", dropna=False):
        y_true = part["y_true"].astype(int).to_numpy()
        y_pred = part["y_pred"].astype(int).to_numpy()
        y_prob = part["y_prob"].astype(float).to_numpy()
        metrics = _compute_eval_metrics(y_true, y_pred, y_prob)
        rows.append(
            {
                "model": model_name,
                "eval_split": eval_split,
                "source_type": str(source_type),
                "n_samples": int(len(part)),
                "positive_rate": float(np.mean(y_true)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def train_models(
    features_path: Path,
    models_dir: Path,
    reports_dir: Path,
    preferred_label: str | None = None,
    feedback_path: Path | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(features_path)
    if "test_id" in df.columns:
        df["test_id"] = df["test_id"].astype(str)
    if "question_code" in df.columns:
        df["question_code"] = df["question_code"].astype(str)

    if "label_train" not in df.columns:
        if "label_bootstrap" in df.columns:
            df["label_train"] = df["label_bootstrap"]
        elif "label_weak" in df.columns:
            df["label_train"] = df["label_weak"]
        elif "label" in df.columns:
            df["label_train"] = df["label"]
    if "label_source_train" not in df.columns:
        if "label_source" in df.columns:
            df["label_source_train"] = df["label_source"]
        else:
            df["label_source_train"] = "missing"

    if feedback_path is not None and feedback_path.exists():
        fb = pd.read_csv(feedback_path)
        required = {"test_id", "question_code", "expert_label"}
        if required.issubset(set(fb.columns)):
            fb2 = fb.copy()
            fb2["test_id"] = fb2["test_id"].astype(str)
            fb2["question_code"] = fb2["question_code"].astype(str)
            fb2["expert_label"] = pd.to_numeric(fb2["expert_label"], errors="coerce")
            fb2 = fb2.dropna(subset=["expert_label"])
            fb2["expert_label"] = fb2["expert_label"].astype(int).clip(0, 1)
            fb2 = (
                fb2.sort_values("created_at" if "created_at" in fb2.columns else fb2.columns[0])
                .drop_duplicates(subset=["test_id", "question_code"], keep="last")
            )
            # Force merge keys to string in dedicated technical columns to avoid dtype conflicts.
            df["__merge_test_id"] = df["test_id"].astype(str)
            df["__merge_question_code"] = df["question_code"].astype(str)
            fb2["__merge_test_id"] = fb2["test_id"].astype(str)
            fb2["__merge_question_code"] = fb2["question_code"].astype(str)

            df = df.merge(
                fb2[["__merge_test_id", "__merge_question_code", "expert_label"]],
                on=["__merge_test_id", "__merge_question_code"],
                how="left",
            )
            df = df.drop(columns=["__merge_test_id", "__merge_question_code"], errors="ignore")
            base_label = None
            for candidate in ["label_train", "label_bootstrap", "label_hybrid", "label_weak", "label"]:
                if candidate in df.columns:
                    base_label = candidate
                    break

            df["label_expert"] = df["expert_label"]
            if base_label is not None:
                df["label_train"] = df["label_expert"].where(df["label_expert"].notna(), df[base_label])
            else:
                df["label_train"] = df["label_expert"]
            df["label_hybrid"] = df["label_train"]
            if "label_source_train" in df.columns:
                df["label_source_train"] = np.where(df["label_expert"].notna(), "expert", df["label_source_train"])
            else:
                df["label_source_train"] = np.where(df["label_expert"].notna(), "expert", "missing")

    preferred_label = preferred_label or ("label_train" if "label_train" in df.columns else None)
    label_col = detect_label_column(df, preferred_label=preferred_label)
    feature_cols = select_feature_columns(df, label_col=label_col, excluded=DEFAULT_EXCLUDED_COLS)
    if not feature_cols:
        raise ValueError("No numeric feature columns were found for training.")

    work_df = df.dropna(subset=[label_col]).copy()
    work_df[label_col] = work_df[label_col].astype(int)
    if "source_type" not in work_df.columns:
        work_df["source_type"] = "unknown"

    if "test_id" not in work_df.columns:
        raise ValueError("Column `test_id` is required for Group CV and fixed holdout.")
    if work_df[label_col].nunique() < 2:
        raise ValueError("Training requires at least two classes in labels.")

    train_groups, holdout_groups = split_fixed_group_holdout(
        df=work_df,
        label_col=label_col,
        group_col="test_id",
        holdout_frac=0.2,
        random_state=42,
    )
    dev_df = work_df[work_df["test_id"].astype(str).isin(train_groups)].copy().reset_index(drop=True)
    holdout_df = work_df[work_df["test_id"].astype(str).isin(holdout_groups)].copy().reset_index(drop=True)
    if dev_df.empty or holdout_df.empty:
        raise ValueError("Failed to create non-empty development and holdout splits.")
    if dev_df[label_col].nunique() < 2 or holdout_df[label_col].nunique() < 2:
        raise ValueError("Development and holdout splits must contain both classes.")

    x_dev = dev_df[feature_cols]
    y_dev = dev_df[label_col].astype(int)
    groups_dev = dev_df["test_id"].astype(str)
    x_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df[label_col].astype(int)

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models = build_models()

    rows: list[dict[str, Any]] = []
    fold_tables: list[pd.DataFrame] = []
    source_tables: list[pd.DataFrame] = []
    fitted_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    oof_prob_cache: dict[str, np.ndarray] = {}

    y_dev_arr = y_dev.to_numpy()
    for model_name, model in models.items():
        cv_detail = evaluate_model_with_group_cv_detailed(model=model, x=x_dev, y=y_dev, groups=groups_dev)
        cv_summary = cv_detail["summary"]
        fold_df = cv_detail["fold_metrics"]
        oof_proba = cv_detail["oof_proba"]
        if len(oof_proba) != len(y_dev_arr) or np.isnan(oof_proba).any():
            raise ValueError(f"OOF predictions are incomplete for model `{model_name}`.")

        threshold, cv_f1_opt = compute_optimal_threshold(y_dev, oof_proba)
        oof_pred = (oof_proba >= threshold).astype(int)
        oof_prob_std = float(np.std(oof_proba))

        model_fitted = clone(model)
        model_fitted.fit(x_dev, y_dev)
        train_score = model_fitted.predict_proba(x_dev)[:, 1]
        train_pred = (train_score >= threshold).astype(int)
        train_metrics = compute_binary_metrics(y_dev_arr, train_pred, train_score)

        holdout_score = model_fitted.predict_proba(x_holdout)[:, 1]
        holdout_pred = (holdout_score >= threshold).astype(int)
        holdout_metrics = _compute_eval_metrics(y_holdout.to_numpy(), holdout_pred, holdout_score)

        row = {
            "model": model_name,
            "n_samples_dev": int(len(dev_df)),
            "n_samples_holdout": int(len(holdout_df)),
            "n_features": int(len(feature_cols)),
            "threshold": float(threshold),
            "cv_f1_opt": float(cv_f1_opt),
            "oof_prob_std": oof_prob_std,
        }
        row.update(cv_summary)
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"holdout_{k}": v for k, v in holdout_metrics.items()})
        rows.append(row)

        fitted_models[model_name] = model_fitted
        thresholds[model_name] = float(threshold)
        oof_prob_cache[model_name] = oof_proba

        fold_df = fold_df.copy()
        fold_df["model"] = model_name
        fold_tables.append(fold_df)

        dev_eval_df = dev_df[["source_type"]].copy()
        dev_eval_df["y_true"] = y_dev_arr
        dev_eval_df["y_prob"] = oof_proba
        dev_eval_df["y_pred"] = oof_pred
        source_tables.append(_build_source_type_breakdown(dev_eval_df, model_name=model_name, eval_split="cv_oof"))

        holdout_eval_df = holdout_df[["source_type"]].copy()
        holdout_eval_df["y_true"] = y_holdout.to_numpy()
        holdout_eval_df["y_prob"] = holdout_score
        holdout_eval_df["y_pred"] = holdout_pred
        source_tables.append(_build_source_type_breakdown(holdout_eval_df, model_name=model_name, eval_split="holdout"))

        joblib.dump(
            {
                "model": model_fitted,
                "feature_cols": feature_cols,
                "label_col": label_col,
                "threshold": float(threshold),
            },
            models_dir / f"{model_name}.pkl",
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["cv_f1_opt", "cv_pr_auc", "holdout_f1"],
        ascending=False,
        na_position="last",
    )
    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    fold_metrics_path = reports_dir / "model_cv_fold_metrics.csv"
    if fold_tables:
        pd.concat(fold_tables, ignore_index=True).to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")

    source_breakdown_path = reports_dir / "model_source_type_breakdown.csv"
    if source_tables:
        pd.concat(source_tables, ignore_index=True).to_csv(source_breakdown_path, index=False, encoding="utf-8-sig")

    best_model_name = str(comparison_df.iloc[0]["model"])

    # Build a weighted ensemble from top models with close OOF F1.
    top_f1 = float(comparison_df.iloc[0]["cv_f1_opt"])
    candidate_df = comparison_df[comparison_df["cv_f1_opt"] >= (top_f1 - 0.02)].head(3).copy()
    if candidate_df.empty:
        candidate_df = comparison_df.head(1).copy()

    ensemble_items = []
    for _, row in candidate_df.iterrows():
        name = str(row["model"])
        weight = float(row["cv_f1_opt"]) if pd.notna(row["cv_f1_opt"]) else 0.0
        ensemble_items.append(
            {
                "name": name,
                "model": fitted_models[name],
                "weight": max(0.001, weight),
                "threshold": float(thresholds[name]),
            }
        )

    total_w = sum(item["weight"] for item in ensemble_items)
    if total_w <= 0:
        total_w = float(len(ensemble_items))
        for item in ensemble_items:
            item["weight"] = 1.0

    ensemble_threshold = float(sum(item["weight"] * item["threshold"] for item in ensemble_items) / total_w)
    best_bundle: dict[str, Any] = {
        "model_kind": "ensemble",
        "models": ensemble_items,
        "feature_cols": feature_cols,
        "label_col": label_col,
        "threshold": ensemble_threshold,
        "base_best_model": best_model_name,
    }
    joblib.dump(best_bundle, models_dir / "best_model.pkl")

    # Ensemble diagnostics on CV-OOF and fixed holdout.
    ensemble_oof = np.zeros(len(y_dev_arr), dtype=float)
    for item in ensemble_items:
        ensemble_oof += float(item["weight"]) * oof_prob_cache[item["name"]]
    ensemble_oof /= total_w
    ensemble_oof_pred = (ensemble_oof >= ensemble_threshold).astype(int)
    ensemble_cv_metrics = _compute_eval_metrics(y_dev_arr, ensemble_oof_pred, ensemble_oof)

    ensemble_holdout_prob = predict_proba_from_bundle(best_bundle, x_holdout)
    ensemble_holdout_pred = (ensemble_holdout_prob >= ensemble_threshold).astype(int)
    ensemble_holdout_metrics = _compute_eval_metrics(y_holdout.to_numpy(), ensemble_holdout_pred, ensemble_holdout_prob)

    ensemble_source_tables = []
    dev_eval_df = dev_df[["source_type"]].copy()
    dev_eval_df["y_true"] = y_dev_arr
    dev_eval_df["y_prob"] = ensemble_oof
    dev_eval_df["y_pred"] = ensemble_oof_pred
    ensemble_source_tables.append(_build_source_type_breakdown(dev_eval_df, model_name="ensemble", eval_split="cv_oof"))

    holdout_eval_df = holdout_df[["source_type"]].copy()
    holdout_eval_df["y_true"] = y_holdout.to_numpy()
    holdout_eval_df["y_prob"] = ensemble_holdout_prob
    holdout_eval_df["y_pred"] = ensemble_holdout_pred
    ensemble_source_tables.append(_build_source_type_breakdown(holdout_eval_df, model_name="ensemble", eval_split="holdout"))
    if ensemble_source_tables:
        ens_df = pd.concat(ensemble_source_tables, ignore_index=True)
        if source_breakdown_path.exists():
            base_df = pd.read_csv(source_breakdown_path)
            pd.concat([base_df, ens_df], ignore_index=True).to_csv(source_breakdown_path, index=False, encoding="utf-8-sig")
        else:
            ens_df.to_csv(source_breakdown_path, index=False, encoding="utf-8-sig")

    # Save predictions for all labeled rows with split markers.
    x_all = work_df[feature_cols]
    all_prob = predict_proba_from_bundle(best_bundle, x_all)
    pred_df = work_df[
        ["test_id", "question_code", "question_raw", "question_type", "source_type", "score_result", label_col]
    ].copy()
    pred_df["split"] = np.where(pred_df["test_id"].astype(str).isin(holdout_groups), "holdout", "dev")
    pred_df["deception_probability"] = all_prob
    pred_df["pred_label"] = (pred_df["deception_probability"] >= ensemble_threshold).astype(int)
    pred_df.to_csv(reports_dir / "predictions_train.csv", index=False, encoding="utf-8-sig")

    fi_df = extract_feature_importance(fitted_models[best_model_name], feature_cols)
    fi_df.to_csv(reports_dir / "feature_importance_best.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "split": {
            "dev_groups": sorted(list(train_groups)),
            "holdout_groups": sorted(list(holdout_groups)),
            "n_dev_rows": int(len(dev_df)),
            "n_holdout_rows": int(len(holdout_df)),
        },
        "ensemble_cv_oof_metrics": ensemble_cv_metrics,
        "ensemble_holdout_metrics": ensemble_holdout_metrics,
    }
    with open(reports_dir / "model_validation_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    summary = {
        "best_model": str(best_model_name),
        "serving_model": "ensemble",
        "ensemble_members": [item["name"] for item in ensemble_items],
        "label_col": label_col,
        "n_samples_total": int(len(work_df)),
        "n_samples_dev": int(len(dev_df)),
        "n_samples_holdout": int(len(holdout_df)),
        "n_features": int(len(feature_cols)),
        "threshold": float(ensemble_threshold),
        "comparison_table": str(comparison_path.as_posix()),
        "cv_fold_table": str(fold_metrics_path.as_posix()),
        "source_type_table": str(source_breakdown_path.as_posix()),
        "validation_diagnostics": str((reports_dir / "model_validation_diagnostics.json").as_posix()),
    }
    with open(reports_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models for polygraph DSS.")
    parser.add_argument("--features-path", type=Path, default=Path("data/processed/polygram_features.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--feedback-path", type=Path, default=Path("data/processed/feedback_labels.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_models(
        features_path=args.features_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        preferred_label=args.label_col,
        feedback_path=args.feedback_path,
    )
    print("[train] complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
