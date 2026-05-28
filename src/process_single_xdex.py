from __future__ import annotations

import argparse
import html
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

QUESTION_COLOR = {
    "I": "#A7C957",
    "R": "#3B82F6",
    "C": "#EE6C4D",
    "SR": "#7B2CBF",
    "S": "#577590",
    "B": "#9CA3AF",
    "DL": "#EE6C4D",
}

CHANNEL_MAP = {
    "ar": "АД (артериальное давление)",
    "bv": "Дыхание (грудной канал)",
    "absbv": "Дыхание (брюшной канал)",
    "eda": "КГР / EDA",
    "ple": "ПГ (плетизмограмма)",
    "tremor": "Тремор",
    "tr": "Двигательная активность",
    "sound": "Аудиоканал",
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


def extract_manifest(xdex_path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    with zipfile.ZipFile(xdex_path, "r") as zf:
        test_json = _read_json_by_suffix(zf, "/test.json")
        doc_json = _read_json_by_suffix(zf, "/document.json")

    questions_section = test_json.get("questionList", {})
    questions = questions_section.get("questions", []) if isinstance(questions_section, dict) else []

    counters: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for idx, question in enumerate(questions, start=1):
        q_type = str(question.get("type", "UNK"))
        counters[q_type] = counters.get(q_type, 0) + 1
        q_code = f"{QUESTION_TYPE_CODE.get(q_type, q_type)}{counters[q_type]}"

        contents = question.get("contents", [])
        text = ""
        waiting_answer_raw = None
        waiting_answer_ru = None
        waiting_answer_label = None
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

        rows.append(
            {
                "order": idx,
                "question_code": q_code,
                "question_type_code": q_type,
                "question_type_ru": QUESTION_TYPE_RU.get(q_type, "Прочее"),
                "question_text": text,
                "is_allowed": bool(question.get("isAllowed", True)),
                "waiting_answer_raw": waiting_answer_raw,
                "waiting_answer_ru": waiting_answer_ru,
                "waiting_answer_label": waiting_answer_label,
            }
        )

    question_df = pd.DataFrame(rows)
    channels_raw = []
    channel_infos = doc_json.get("channelInfos", {})
    if isinstance(channel_infos, dict):
        channels_raw = sorted(channel_infos.keys())
    channels_pretty = [CHANNEL_MAP.get(ch, ch) for ch in channels_raw]

    summary = {
        "test_type": test_json.get("testType"),
        "questions_total": int(len(question_df)),
        "questions_allowed": int(question_df["is_allowed"].sum()) if not question_df.empty else 0,
        "question_types": question_df["question_type_ru"].value_counts().to_dict() if not question_df.empty else {},
        "channels_total": int(len(channels_raw)),
        "channels_raw": channels_raw,
        "channels_pretty": channels_pretty,
        "can_score_directly_from_this_xdex": False,
        "why_not_direct_scoring": (
            "В этом формате .xdex хранятся сырые сигналы и структура теста, "
            "а готовые метрики I/II/Результат для каждого вопроса отсутствуют в явном табличном виде."
        ),
        "next_step": (
            "В Диане экспортировать таблицу метрик по вопросам в DOCX/CSV/XLSX, "
            "после чего загрузить файл в `data/raw/docx` и запустить `python src/run_pipeline.py`."
        ),
    }
    return question_df, channels_pretty, summary


def plot_question_route(question_df: pd.DataFrame, output_path: Path) -> Path:
    if question_df.empty:
        raise ValueError("No questions found in XDEX.")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title("Маршрут вопросов в тесте (по порядку)")
    ax.set_xlim(0.5, len(question_df) + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Порядковый номер вопроса")
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    for _, row in question_df.iterrows():
        x = row["order"]
        q_type = row["question_type_code"]
        color = QUESTION_COLOR.get(q_type, "#888888")
        ax.scatter([x], [0.5], s=420, c=[color], edgecolors="black", linewidths=0.6, zorder=3)
        ax.text(x, 0.5, str(row["question_code"]), ha="center", va="center", fontsize=8, color="black", weight="bold")

    handles = []
    labels = []
    for t_code, t_ru in QUESTION_TYPE_RU.items():
        if t_code in set(question_df["question_type_code"]):
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=QUESTION_COLOR[t_code], markeredgecolor="black", markersize=9)
            )
            labels.append(f"{QUESTION_TYPE_CODE.get(t_code, t_code)} вЂ” {t_ru}")
    if handles:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=min(4, len(handles)), frameon=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_html_report(
    xdex_input: Path,
    question_df: pd.DataFrame,
    channels_pretty: list[str],
    summary: dict[str, Any],
    route_png_path: Path,
    output_html: Path,
) -> Path:
    table_df = question_df.copy()
    table_df["is_allowed"] = table_df["is_allowed"].map({True: "Да", False: "Нет"})
    table_html = table_df.to_html(index=False, border=0, classes="questions-table", justify="left", escape=True)

    channels_html = "".join(f"<li>{html.escape(ch)}</li>" for ch in channels_pretty) if channels_pretty else "<li>Нет данных</li>"
    q_types_html = "".join(
        f"<li>{html.escape(k)}: {v}</li>" for k, v in summary.get("question_types", {}).items()
    ) or "<li>Нет данных</li>"

    html_text = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>Отчет по XDEX: {html.escape(xdex_input.name)}</title>
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      margin: 24px;
      background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
      color: #1e293b;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 16px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    .muted {{ color: #475569; }}
    .badge {{
      display: inline-block;
      background: #e2e8f0;
      color: #0f172a;
      padding: 4px 8px;
      border-radius: 8px;
      margin-right: 6px;
      margin-bottom: 6px;
      font-size: 12px;
    }}
    img {{
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
    }}
    .questions-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
      background: #fff;
    }}
    .questions-table th, .questions-table td {{
      border: 1px solid #e2e8f0;
      padding: 8px;
      text-align: left;
    }}
    .questions-table th {{
      background: #f1f5f9;
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Отчет по входному файлу XDEX</h1>
    <p class="muted">Файл: <code>{html.escape(str(xdex_input))}</code></p>
    <span class="badge">Вопросов: {summary.get('questions_total', 0)}</span>
    <span class="badge">Допущено в тест: {summary.get('questions_allowed', 0)}</span>
    <span class="badge">Каналов: {summary.get('channels_total', 0)}</span>
  </div>

  <div class="card">
    <h2>Маршрут Вопросов</h2>
    <img src="{html.escape(route_png_path.name)}" alt="Маршрут вопросов" />
  </div>

  <div class="card">
    <h2>Типы Вопросов</h2>
    <ul>{q_types_html}</ul>
  </div>

  <div class="card">
    <h2>Каналы Датчиков</h2>
    <ul>{channels_html}</ul>
  </div>

  <div class="card">
    <h2>Таблица Вопросов</h2>
    {table_html}
  </div>

  <div class="card">
    <h2>Что Получено На Выходе</h2>
    <p>{html.escape(summary.get('why_not_direct_scoring', ''))}</p>
    <p><strong>Следующий шаг:</strong> {html.escape(summary.get('next_step', ''))}</p>
  </div>
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")
    return output_html


def run_single_xdex(xdex_path: Path, project_root: Path) -> dict[str, str]:
    if not xdex_path.exists():
        raise FileNotFoundError(f"XDEX file not found: {xdex_path}")

    raw_xdex_dir = project_root / "data" / "raw" / "xdex"
    raw_xdex_dir.mkdir(parents=True, exist_ok=True)
    copied_path = raw_xdex_dir / xdex_path.name
    shutil.copy2(xdex_path, copied_path)

    question_df, channels_pretty, summary = extract_manifest(copied_path)

    run_dir = project_root / "reports" / "single_xdex" / copied_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    questions_csv = run_dir / "questions.csv"
    summary_json = run_dir / "summary.json"
    route_png = run_dir / "question_route.png"
    report_html = run_dir / "report.html"

    question_df.to_csv(questions_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_question_route(question_df, route_png)
    build_html_report(copied_path, question_df, channels_pretty, summary, route_png, report_html)

    return {
        "xdex_copied_to": str(copied_path),
        "questions_csv": str(questions_csv),
        "summary_json": str(summary_json),
        "question_route_png": str(route_png),
        "report_html": str(report_html),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="User-friendly processing of a single XDEX file.")
    parser.add_argument("--xdex-path", type=Path, required=True, help="Absolute or relative path to .xdex/.xdx file.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_single_xdex(xdex_path=args.xdex_path, project_root=args.project_root.resolve())
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
