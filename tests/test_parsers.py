from __future__ import annotations

from pathlib import Path

from src.parser import RAW_COLUMNS
from src.parser import parse_docx_folder
from src.parser import parse_docx_table
from src.parser import parse_xdex_file
from src.parser import parse_xdex_folder


ROOT = Path(__file__).resolve().parents[1]
DOCX_DIR = ROOT / "data" / "raw" / "docx"
XDEX_DIR = ROOT / "data" / "raw" / "xdex"


def _docx_files() -> list[Path]:
    files = sorted(path for path in DOCX_DIR.glob("*.docx") if not path.name.startswith("~$"))
    assert files, "No DOCX test files found under data/raw/docx."
    return files


def _xdex_files() -> list[Path]:
    files = sorted(list(XDEX_DIR.glob("*.xdex")) + list(XDEX_DIR.glob("*.xdx")))
    assert files, "No XDEX/XDX test files found under data/raw/xdex."
    return files


def test_parse_docx_table_required_columns_and_non_empty() -> None:
    sample = _docx_files()[0]
    df = parse_docx_table(sample)

    required = {
        "source_file",
        "source_type",
        "test_id",
        "question_raw",
        "question_code",
        "question_type",
        "parameter",
        "metric_i",
        "metric_ii",
        "score_result",
    }
    assert not df.empty
    assert required.issubset(set(df.columns))
    assert set(df["source_type"].dropna().unique()) == {"docx"}
    assert df["metric_i"].notna().any()
    assert df["parameter"].astype(str).str.strip().ne("").all()


def test_parse_docx_folder_row_count_matches_sum_of_files() -> None:
    expected_rows = sum(len(parse_docx_table(path)) for path in _docx_files())
    folder_df = parse_docx_folder(DOCX_DIR)

    assert len(folder_df) == expected_rows
    assert set(folder_df.columns) == set(RAW_COLUMNS)


def test_parse_xdex_file_required_columns_and_types() -> None:
    sample = _xdex_files()[0]
    df = parse_xdex_file(sample)

    required = {
        "source_file",
        "source_type",
        "test_id",
        "question_raw",
        "question_code",
        "question_type",
        "parameter",
        "metric_i",
        "score_result",
    }
    allowed_question_types = {"neutral", "control", "probable_lie", "sacrifice", "symptomatic", "other"}

    assert not df.empty
    assert required.issubset(set(df.columns))
    assert set(df["source_type"].dropna().unique()) == {"xdex"}
    assert df["parameter"].notna().any()
    assert df["metric_i"].notna().any()
    assert set(df["question_type"].dropna().unique()).issubset(allowed_question_types)


def test_parse_xdex_folder_row_count_matches_sum_of_files() -> None:
    expected_rows = sum(len(parse_xdex_file(path)) for path in _xdex_files())
    folder_df = parse_xdex_folder(XDEX_DIR)

    assert len(folder_df) == expected_rows
    assert set(folder_df.columns) == set(RAW_COLUMNS)
