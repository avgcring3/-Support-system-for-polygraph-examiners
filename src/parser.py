from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import numpy as np


QUESTION_TYPE_MAP = {
    "\u041d\u0432": "neutral",
    "\u041f\u0432": "control",
    "\u0412\u043b": "probable_lie",
    "\u0416\u0432": "sacrifice",
    "\u0421\u0432": "symptomatic",
}

XDEX_TYPE_CODE_MAP = {
    "I": "neutral",
    "R": "control",
    "C": "probable_lie",
    "SR": "sacrifice",
    "S": "symptomatic",
    "B": "other",
    "DL": "probable_lie",
}

RAW_COLUMNS = [
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
    "metric_i",
    "metric_ii",
    "score_result",
]


def _tag_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", maxsplit=1)[1].lower()
    return tag.lower()


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("\xa0", "").replace(" ", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _split_question_text(question_raw: str | None) -> tuple[str | None, str | None, str | None]:
    if not question_raw:
        return None, None, None
    text = question_raw.strip()
    code_match = re.match(r"^([\u0400-\u04FFA-Za-z]{1,3}\d+)", text)
    if not code_match:
        return None, text, None
    code = code_match.group(1)
    q_type = QUESTION_TYPE_MAP.get(code[:2], "other")
    description = text[len(code) :].strip()
    return code, description if description else None, q_type


def _read_docx_document_xml(docx_path: Path) -> ET.Element:
    with zipfile.ZipFile(docx_path, "r") as zf:
        with zf.open("word/document.xml") as xml_file:
            tree = ET.parse(xml_file)
    return tree.getroot()


def _cell_text(tc: ET.Element) -> str:
    texts = []
    for node in tc.iter():
        if _tag_local_name(node.tag) == "t" and node.text:
            texts.append(node.text)
    return "".join(texts).strip()


def parse_docx_table(docx_path: Path) -> pd.DataFrame:
    root = _read_docx_document_xml(docx_path)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    records: list[dict] = []
    current_question = None
    current_result = None

    for tbl in root.findall(".//w:tbl", ns):
        rows = tbl.findall(".//w:tr", ns)
        if not rows:
            continue

        header: list[str] | None = None
        for row in rows:
            cells = row.findall("w:tc", ns)
            values = [_cell_text(cell) for cell in cells]
            if not any(values):
                continue

            lower_values = [value.lower() for value in values]
            if "вопрос" in lower_values and "параметр" in lower_values:
                header = values
                continue

            # Fallback for corrupted header rows
            if header is None:
                header = ["Вопрос", "Параметр", "I", "II", "Результат"]

            row_map = {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}
            question_raw = row_map.get("Вопрос", "").strip() or current_question
            parameter = row_map.get("Параметр", "").strip()
            metric_i = _to_float(row_map.get("I"))
            metric_ii = _to_float(row_map.get("II"))
            result = _to_float(row_map.get("Результат"))

            if result is not None:
                current_result = result
            if question_raw:
                current_question = question_raw
            if not parameter:
                continue

            q_code, q_text, q_type = _split_question_text(current_question)
            test_id = docx_path.stem

            records.append(
                {
                    "source_file": docx_path.name,
                    "source_type": "docx",
                    "test_id": test_id,
                    "question_raw": current_question,
                    "question_code": q_code,
                    "question_text": q_text,
                    "question_type": q_type,
                    "parameter": parameter,
                    "metric_i": metric_i,
                    "metric_ii": metric_ii,
                    "score_result": result if result is not None else current_result,
                }
            )

    return pd.DataFrame.from_records(records)


def parse_docx_folder(docx_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted(docx_dir.glob("*.docx")):
        if path.name.startswith("~$"):
            continue
        frame = parse_docx_table(path)
        if not frame.empty:
            records.extend(frame.reindex(columns=RAW_COLUMNS).to_dict(orient="records"))
    if not records:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.DataFrame.from_records(records, columns=RAW_COLUMNS)


def _find_first_by_local_name(root: ET.Element, names: Iterable[str]) -> ET.Element | None:
    names_set = {name.lower() for name in names}
    for node in root.iter():
        if _tag_local_name(node.tag) in names_set:
            return node
    return None


def _extract_text_numeric(node: ET.Element, candidate_tags: Iterable[str]) -> float | None:
    tags = {tag.lower() for tag in candidate_tags}
    for child in node.iter():
        if _tag_local_name(child.tag) in tags and child.text:
            value = _to_float(child.text)
            if value is not None:
                return value
    return None


def parse_xdex_file(xdex_path: Path) -> pd.DataFrame:
    """
    Generic parser for Diana-like .xdex/.xdx archives.
    Works with the common structure:
    question -> channel -> metricI/metricII, plus question-level result/score.
    """
    records: list[dict] = []
    with zipfile.ZipFile(xdex_path, "r") as zf:
        xml_names = [name for name in zf.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            return _parse_xdex_via_scores(xdex_path)
        with zf.open(xml_names[0]) as xml_file:
            root = ET.parse(xml_file).getroot()

    question_nodes = [node for node in root.iter() if _tag_local_name(node.tag) in {"question", "q"}]
    if not question_nodes:
        return pd.DataFrame()

    for q_node in question_nodes:
        question_raw = (
            q_node.get("text")
            or q_node.get("name")
            or q_node.get("code")
            or (q_node.text.strip() if q_node.text else None)
        )
        q_code, q_text, q_type = _split_question_text(question_raw)
        result = _extract_text_numeric(q_node, ["result", "score", "index", "probability"])

        channel_nodes = [node for node in q_node.iter() if _tag_local_name(node.tag) in {"channel", "sensor", "parameter"}]
        if not channel_nodes:
            continue

        for ch_node in channel_nodes:
            parameter = ch_node.get("name") or ch_node.get("type") or ch_node.get("id")
            if not parameter:
                continue

            metric_i = _extract_text_numeric(ch_node, ["metrici", "i", "value1", "first"])
            metric_ii = _extract_text_numeric(ch_node, ["metricii", "ii", "value2", "second"])

            records.append(
                {
                    "source_file": xdex_path.name,
                    "source_type": "xdex",
                    "test_id": xdex_path.stem,
                    "question_raw": question_raw,
                    "question_code": q_code,
                    "question_text": q_text,
                    "question_type_code": None,
                    "question_type": q_type,
                    "is_allowed": True,
                    "waiting_answer_raw": None,
                    "waiting_answer_label": None,
                    "waiting_answer_ru": None,
                    "parameter": parameter,
                    "metric_i": metric_i,
                    "metric_ii": metric_ii,
                    "score_result": result,
                }
            )

    return pd.DataFrame.from_records(records)


def _parse_xdex_via_scores(xdex_path: Path) -> pd.DataFrame:
    """
    Fallback for modern Diana .xdex archives where the main payload is JSON (`test.json`)
    and no XML question/channel table exists.
    """
    try:
        from .xdex_epp import extract_scores_dataframe
    except ImportError:
        from xdex_epp import extract_scores_dataframe

    scores_df = extract_scores_dataframe(xdex_path)
    if scores_df.empty:
        return pd.DataFrame()

    channel_cols = [
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
    channel_cols = [c for c in channel_cols if c in scores_df.columns]

    records: list[dict] = []
    for _, row in scores_df.iterrows():
        q_code = row.get("question_code")
        q_text = row.get("question_text")
        q_type_code = str(row.get("question_type_code", "")).upper()
        q_type_ru = row.get("question_type_ru")
        q_type = "other"
        if isinstance(q_type_ru, str):
            low = q_type_ru.lower()
            if "нейтр" in low:
                q_type = "neutral"
            elif "контр" in low:
                q_type = "control"
            elif "провер" in low or "значим" in low:
                q_type = "probable_lie"

        q_type = XDEX_TYPE_CODE_MAP.get(q_type_code, q_type)

        score_result = row.get("counter_answer")
        if score_result is None or (isinstance(score_result, float) and np.isnan(score_result)):
            # If direct answer score is unavailable, use a small aggregate reaction proxy.
            reaction = [row.get(c) for c in ("a_bv", "a_eda", "a_ple") if c in row]
            reaction = [float(v) for v in reaction if v is not None and not pd.isna(v)]
            score_result = float(np.mean(np.abs(reaction))) if reaction else np.nan

        for ch in channel_cols:
            val = row.get(ch)
            if val is None or pd.isna(val):
                continue
            records.append(
                {
                    "source_file": xdex_path.name,
                    "source_type": "xdex",
                    "test_id": xdex_path.stem,
                    "question_raw": q_text,
                    "question_code": q_code,
                    "question_text": q_text,
                    "question_type_code": q_type_code,
                    "question_type": q_type,
                    "is_allowed": bool(row.get("is_allowed", True)),
                    "waiting_answer_raw": row.get("waiting_answer_raw"),
                    "waiting_answer_label": row.get("waiting_answer_label"),
                    "waiting_answer_ru": row.get("waiting_answer_ru"),
                    "parameter": ch,
                    "metric_i": float(val),
                    "metric_ii": np.nan,
                    "score_result": score_result,
                }
            )

    return pd.DataFrame.from_records(records)


def parse_xdex_folder(xdex_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted(list(xdex_dir.glob("*.xdex")) + list(xdex_dir.glob("*.xdx"))):
        frame = parse_xdex_file(path)
        if not frame.empty:
            records.extend(frame.reindex(columns=RAW_COLUMNS).to_dict(orient="records"))
    if not records:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.DataFrame.from_records(records, columns=RAW_COLUMNS)


def load_raw_records(docx_dir: Path, xdex_dir: Path | None = None) -> pd.DataFrame:
    frames = []
    docx_df = parse_docx_folder(docx_dir)
    if not docx_df.empty:
        frames.append(docx_df)
    if xdex_dir is not None and xdex_dir.exists():
        xdex_df = parse_xdex_folder(xdex_dir)
        if not xdex_df.empty:
            frames.append(xdex_df)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    if len(frames) == 1:
        out = frames[0].reset_index(drop=True)
    else:
        out = pd.concat(frames, ignore_index=True)
    out["parameter"] = out["parameter"].astype(str).str.strip()
    return out

