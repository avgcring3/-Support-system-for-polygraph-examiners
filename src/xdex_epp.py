from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUESTION_TYPE_RU = {
    "I": "\u041d\u0435\u0439\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439",
    "R": "\u041f\u0440\u043e\u0432\u0435\u0440\u043e\u0447\u043d\u044b\u0439",
    "C": "\u0412\u0435\u0440\u043e\u044f\u0442\u043d\u043e\u0439 \u043b\u0436\u0438",
    "SR": "\u0416\u0435\u0440\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u043f\u0440\u043e\u0432\u0435\u0440\u043e\u0447\u043d\u044b\u0439",
    "S": "\u0421\u0438\u043c\u043f\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0439",
    "B": "Вводный вопрос",
    "DL": "Вопрос управляемой лжи",
}

QUESTION_TYPE_CODE = {
    "I": "\u041d\u0432",
    "R": "\u041f\u0432",
    "C": "\u0412\u043b",
    "SR": "\u0416\u0432",
    "S": "\u0421\u0432",
    "B": "B",
    "DL": "DL",
}

QUESTION_TYPE_CANONICAL = {
    "B": "I",
    "DL": "C",
}


def _read_json_by_suffix(zf: zipfile.ZipFile, suffix: str) -> dict[str, Any]:
    names = [name for name in zf.namelist() if name.endswith(suffix)]
    if not names:
        return {}
    raw = zf.read(names[0])
    for enc in ("utf-8", "cp1251"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    return {}


def load_test_json(xdex_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(xdex_path, "r") as zf:
        return _read_json_by_suffix(zf, "/test.json")


def _extract_score_value(score_obj: dict[str, Any]) -> float | None:
    if not isinstance(score_obj, dict):
        return None

    if score_obj.get("isManuallyChanged", False):
        mv = score_obj.get("manualValue")
        if mv is not None:
            try:
                return float(mv)
            except (TypeError, ValueError):
                pass

    av = score_obj.get("autoValue")
    if av is not None:
        try:
            return float(av)
        except (TypeError, ValueError):
            pass

    vv = score_obj.get("value")
    if vv is not None:
        try:
            return float(vv)
        except (TypeError, ValueError):
            pass

    points = score_obj.get("automaticScorePoints")
    if isinstance(points, list):
        vals = []
        for item in points:
            if isinstance(item, dict) and item.get("value") is not None:
                try:
                    vals.append(float(item["value"]))
                except (TypeError, ValueError):
                    continue
        if vals:
            return float(np.mean(vals))
    return None


def _build_question_lookup(test_json: dict[str, Any]) -> dict[int, dict[str, Any]]:
    q_section = test_json.get("questionList", {})
    questions = q_section.get("questions", []) if isinstance(q_section, dict) else []
    counters: dict[str, int] = {}
    out: dict[int, dict[str, Any]] = {}
    for idx, q in enumerate(questions):
        q_type_raw = str(q.get("type", "UNK")).upper()
        q_type = QUESTION_TYPE_CANONICAL.get(q_type_raw, q_type_raw)
        counters[q_type_raw] = counters.get(q_type_raw, 0) + 1
        q_code = f"{QUESTION_TYPE_CODE.get(q_type_raw, q_type_raw)}{counters[q_type_raw]}"

        text = ""
        waiting_answer_raw = None
        waiting_answer_ru = None
        waiting_answer_label = None
        contents = q.get("contents", [])
        if isinstance(contents, list) and contents:
            first_content = contents[0] if isinstance(contents[0], dict) else {}
            text = str(first_content.get("text", "")).strip()
            waiting_answer_raw = first_content.get("waitingAnswer")
            if isinstance(waiting_answer_raw, str):
                wa = waiting_answer_raw.strip().lower()
                if wa == "yes":
                    waiting_answer_ru = "правда"
                    waiting_answer_label = "Truth"
                elif wa == "no":
                    waiting_answer_ru = "ложь"
                    waiting_answer_label = "Lie"

        out[idx] = {
            "question_type_code": q_type,
            "question_type_code_raw": q_type_raw,
            "question_type_ru": QUESTION_TYPE_RU.get(q_type_raw, QUESTION_TYPE_RU.get(q_type, "Прочее")),
            "question_code": q_code,
            "question_text": text,
            "is_allowed": bool(q.get("isAllowed", True)),
            "waiting_answer_raw": waiting_answer_raw,
            "waiting_answer_ru": waiting_answer_ru,
            "waiting_answer_label": waiting_answer_label,
        }
    return out


def extract_scores_dataframe(xdex_path: Path) -> pd.DataFrame:
    test_json = load_test_json(xdex_path)
    lookup = _build_question_lookup(test_json)
    groups = test_json.get("stimulusGroups", [])
    if not isinstance(groups, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for group_idx, group in enumerate(groups):
        stimuli = group.get("stimuli", []) if isinstance(group, dict) else []
        if not isinstance(stimuli, list):
            continue
        for stim_idx, stim in enumerate(stimuli):
            if not isinstance(stim, dict):
                continue
            interval = stim.get("analysisInterval", {})
            if not isinstance(interval, dict):
                continue
            if interval.get("isEmpty", False):
                continue

            q_idx = stim.get("questionIndex")
            if q_idx is None:
                continue
            q_info = lookup.get(int(q_idx), {})
            if not q_info:
                continue

            scores = stim.get("scores", {}) if isinstance(stim.get("scores"), dict) else {}

            def getv(key: str) -> float | None:
                return _extract_score_value(scores.get(key, {}))

            row = {
                "group_idx": group_idx,
                "stimulus_idx": stim_idx,
                "question_index": int(q_idx),
                "question_code": q_info.get("question_code"),
                "question_type_code": q_info.get("question_type_code"),
                "question_type_code_raw": q_info.get("question_type_code_raw"),
                "question_type_ru": q_info.get("question_type_ru"),
                "question_text": q_info.get("question_text"),
                "is_allowed": bool(q_info.get("is_allowed", True)),
                "waiting_answer_raw": q_info.get("waiting_answer_raw"),
                "waiting_answer_ru": q_info.get("waiting_answer_ru"),
                "waiting_answer_label": q_info.get("waiting_answer_label"),
                "analysis_min": interval.get("min"),
                "analysis_max": interval.get("max"),
                # Primary automatic scores
                "a_bv": getv("A_BV"),
                "a_eda": getv("A_EDA"),
                "a_eda2": getv("A_EDA2"),
                "a_ple": getv("A_PLE"),
                "a_arlength": getv("A_ARLENGTH"),
                "a_trlength": getv("A_TRLENGTH"),
                # Additional scores
                "counter_resp": getv("Countering_RESP"),
                "counter_trm": getv("Countering_TRM"),
                "counter_answer": getv("Countering_Answer"),
                "empirical_resp": getv("Empirical_RESP"),
                "empirical_bv": getv("Empirical_BV"),
                "empirical_eda": getv("Empirical_EDA"),
                "empirical_ple": getv("Empirical_PLE"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["group_idx", "stimulus_idx"]).reset_index(drop=True)
    return df


def _safe_scale(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0

    center = float(numeric.median())
    mad = float(np.median(np.abs(numeric - center)))
    robust = 1.4826 * mad
    if robust > 1e-8:
        return robust

    std = float(numeric.std(ddof=0))
    return std if std > 1e-8 else 1.0


def _robust_zscore(series: pd.Series, center_value: float, scale_value: float) -> pd.Series:
    if abs(scale_value) < 1e-8:
        scale_value = 1.0
    return ((series - center_value) / scale_value).clip(-5.0, 5.0)


def _find_nearest_reference_indices(
    question_order: pd.Series,
    reference_order: pd.Series,
    reference_rows: pd.Series,
) -> pd.Series:
    if reference_order.empty:
        return pd.Series([pd.NA] * len(question_order), index=question_order.index, dtype="Int64")

    ref_df = pd.DataFrame({"ref_order": reference_order, "ref_row": reference_rows}).sort_values("ref_order")
    ref_orders = ref_df["ref_order"].to_numpy(dtype=float)
    ref_rows = ref_df["ref_row"].to_numpy(dtype=int)

    out: list[int | None] = []
    for q in question_order.to_numpy(dtype=float):
        pos = int(np.searchsorted(ref_orders, q, side="left"))
        if pos <= 0:
            out.append(int(ref_rows[0]))
            continue
        if pos >= len(ref_orders):
            out.append(int(ref_rows[-1]))
            continue

        left_dist = abs(ref_orders[pos - 1] - q)
        right_dist = abs(ref_orders[pos] - q)
        if right_dist < left_dist:
            out.append(int(ref_rows[pos]))
        else:
            # Equal distance falls back to left (previous) reference.
            out.append(int(ref_rows[pos - 1]))

    return pd.Series(out, index=question_order.index, dtype="Int64")


def _recommendation_from_probability(prob: float) -> str:
    if prob >= 0.65:
        return "Вероятно ложь"
    if prob <= 0.35:
        return "Вероятно правда"
    return "Требует внимания эксперта"


def score_from_xdex_dataframe(scores_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if scores_df.empty:
        raise ValueError("No non-empty stimuli found in XDEX test.")

    df = scores_df.copy()
    numeric_cols = [
        "a_bv",
        "a_eda",
        "a_eda2",
        "a_ple",
        "a_arlength",
        "a_trlength",
        "counter_resp",
        "counter_trm",
        "counter_answer",
        "empirical_resp",
        "empirical_bv",
        "empirical_eda",
        "empirical_ple",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    allowed_mask = df.get("is_allowed", True)
    if isinstance(allowed_mask, pd.Series):
        allowed_mask = allowed_mask.fillna(True).astype(bool)
    else:
        allowed_mask = pd.Series(True, index=df.index)
    q_type = df["question_type_code"].astype(str).str.upper()

    # Channel blocks interpreted as reaction intensity:
    # RESP: suppression/shape changes, EDA: phasic increase, CARDIO: BP/cardio trace, PPG: vasomotor response.
    df["sig_resp"] = (-df["a_bv"].fillna(0.0)) + df["a_trlength"].abs().fillna(0.0)
    df["sig_eda"] = df["a_eda"].fillna(0.0) + 0.5 * df["a_eda2"].fillna(0.0)
    df["sig_cardio"] = df["a_arlength"].fillna(0.0)
    df["sig_ppg"] = df["a_ple"].abs().fillna(0.0)

    baseline = df[allowed_mask & q_type.isin(["I", "C"])]
    if baseline.empty:
        baseline = df[allowed_mask]
    if baseline.empty:
        baseline = df

    channel_map = {
        "resp": "sig_resp",
        "eda": "sig_eda",
        "cardio": "sig_cardio",
        "ppg": "sig_ppg",
    }

    for alias, col in channel_map.items():
        center = float(pd.to_numeric(baseline[col], errors="coerce").median())
        scale = _safe_scale(baseline[col])
        df[f"z_{alias}"] = _robust_zscore(pd.to_numeric(df[col], errors="coerce"), center, scale)

    # For each stimulus find nearest comparison (C) question; this gives pairwise R-vs-C contrast.
    c_mask = allowed_mask & q_type.eq("C")
    df["ref_c_row"] = _find_nearest_reference_indices(
        question_order=pd.to_numeric(df["question_index"], errors="coerce"),
        reference_order=pd.to_numeric(df.loc[c_mask, "question_index"], errors="coerce"),
        reference_rows=pd.Series(df.loc[c_mask].index, index=df.loc[c_mask].index),
    )
    ref_code_map = df["question_code"].to_dict()
    df["ref_c_question_code"] = df["ref_c_row"].map(ref_code_map)

    for alias in channel_map:
        ref_values = df["ref_c_row"].map(df[f"z_{alias}"].to_dict())
        df[f"delta_{alias}_vs_c"] = (df[f"z_{alias}"] - ref_values).fillna(0.0)

    # Literature-inspired weights: stronger emphasis on EDA, then cardio/resp, then PPG.
    channel_weights = {"eda": 0.45, "cardio": 0.25, "resp": 0.20, "ppg": 0.10}
    df["pair_evidence"] = sum(channel_weights[k] * df[f"delta_{k}_vs_c"] for k in channel_weights)
    df["baseline_evidence"] = sum(channel_weights[k] * df[f"z_{k}"] for k in channel_weights)

    empirical_raw = (
        0.10 * df["empirical_resp"].fillna(0.0)
        + 0.55 * df["empirical_eda"].fillna(0.0)
        + 0.25 * df["empirical_bv"].fillna(0.0)
        + 0.10 * df["empirical_ple"].fillna(0.0)
    )
    df["empirical_scaled"] = (empirical_raw / 2.0).clip(-2.0, 2.0)

    # Type logic:
    # - Relevant (R): stronger response than nearest C pushes score toward deception.
    # - Comparison (C): inverse sign, since stronger C is expected for truthful profile.
    # - Others: weak baseline-only contribution, mostly neutral around 0.5 probability.
    evidence = np.where(
        q_type.eq("R"),
        df["pair_evidence"] + 0.20 * df["empirical_scaled"],
        np.where(
            q_type.eq("C"),
            -df["pair_evidence"],
            0.35 * df["baseline_evidence"],
        ),
    )
    df["raw_score"] = evidence
    df["evidence_score"] = evidence

    logits = (1.45 * df["evidence_score"] - 0.10).clip(-20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    other_mask = ~q_type.isin(["R", "C"])
    probs = np.where(other_mask, 0.5 + (probs - 0.5) * 0.35, probs)
    df["deception_probability"] = np.clip(probs, 0.02, 0.98)
    df["pred_label"] = (df["deception_probability"] >= 0.55).astype(int)

    # Overall score focuses on relevant questions; fallback to allowed items if needed.
    relevant_mask = allowed_mask & q_type.eq("R")
    if int(relevant_mask.sum()) > 0:
        target_mask = relevant_mask
    else:
        target_mask = allowed_mask
        if int(target_mask.sum()) == 0:
            target_mask = pd.Series(True, index=df.index)

    overall = float(df.loc[target_mask, "deception_probability"].mean())
    recommendation = _recommendation_from_probability(overall)

    per_channel_mean_abs_delta = {
        k: float(df.loc[target_mask, f"delta_{k}_vs_c"].abs().mean()) for k in channel_weights
    }

    summary = {
        "overall_score": overall,
        "recommendation": recommendation,
        "n_items": int(len(df)),
        "n_relevant_questions": int((allowed_mask & q_type.eq("R")).sum()),
        "n_comparison_questions": int((allowed_mask & q_type.eq("C")).sum()),
        "method": "xdex_direct_scoring_v2_pairwise",
        "channel_weights": channel_weights,
        "mean_abs_pair_delta": per_channel_mean_abs_delta,
        "note": (
            "Оценка построена на парных контрастах R vs C, робастной нормализации каналов "
            "и логистической калибровке вероятности."
        ),
    }
    return df, summary


def run_xdex_epp(xdex_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores_df = extract_scores_dataframe(xdex_path)
    return score_from_xdex_dataframe(scores_df)


def build_question_map_dataframe(test_json: dict[str, Any]) -> pd.DataFrame:
    lookup = _build_question_lookup(test_json)
    rows = []
    for q_idx, info in sorted(lookup.items(), key=lambda x: x[0]):
        rows.append(
            {
                "question_index": q_idx,
                "question_code": info.get("question_code"),
                "question_type_code": info.get("question_type_code"),
                "question_type_code_raw": info.get("question_type_code_raw"),
                "question_type_ru": info.get("question_type_ru"),
                "question_text": info.get("question_text"),
                "is_allowed": info.get("is_allowed"),
            }
        )
    return pd.DataFrame(rows)


def _compute_structure_stats(test_json: dict[str, Any]) -> dict[str, int]:
    groups = test_json.get("stimulusGroups", [])
    groups_total = len(groups) if isinstance(groups, list) else 0
    stimuli_total = 0
    stimuli_nonempty = 0
    for group in groups if isinstance(groups, list) else []:
        stimuli = group.get("stimuli", []) if isinstance(group, dict) else []
        if not isinstance(stimuli, list):
            continue
        for stim in stimuli:
            if not isinstance(stim, dict):
                continue
            stimuli_total += 1
            interval = stim.get("analysisInterval", {})
            if isinstance(interval, dict) and not interval.get("isEmpty", False):
                stimuli_nonempty += 1
    return {
        "groups_total": groups_total,
        "stimuli_total": stimuli_total,
        "stimuli_nonempty": stimuli_nonempty,
        "stimuli_empty_or_skipped": stimuli_total - stimuli_nonempty,
    }


def run_xdex_epp_detailed(xdex_path: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    test_json = load_test_json(xdex_path)
    t1 = time.perf_counter()

    question_map_df = build_question_map_dataframe(test_json)
    structure_stats = _compute_structure_stats(test_json)
    t2 = time.perf_counter()

    scores_df = extract_scores_dataframe(xdex_path)
    t3 = time.perf_counter()
    pred_df, summary = score_from_xdex_dataframe(scores_df)
    t4 = time.perf_counter()

    details = {
        "question_map_df": question_map_df,
        "structure_stats": structure_stats,
        "timing_ms": {
            "read_test_json": round((t1 - t0) * 1000, 2),
            "build_maps": round((t2 - t1) * 1000, 2),
            "extract_scores": round((t3 - t2) * 1000, 2),
            "compute_probabilities": round((t4 - t3) * 1000, 2),
            "total": round((t4 - t0) * 1000, 2),
        },
    }
    summary["timing_ms_total"] = details["timing_ms"]["total"]
    return pred_df, summary, details
