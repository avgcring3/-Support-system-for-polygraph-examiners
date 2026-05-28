from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_plot(predictions_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(predictions_path)
    if "question_code" not in df.columns:
        raise ValueError("Column `question_code` is required for plotting.")
    if "deception_probability" not in df.columns:
        raise ValueError("Column `deception_probability` is required for plotting.")

    plot_df = df[["question_code", "deception_probability"]].dropna().copy()
    plot_df = plot_df.sort_values("deception_probability", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="question_code", y="deception_probability", color="#2f6690")
    plt.axhline(0.6, color="#c1121f", linestyle="--", linewidth=1.2, label="Порог 0.60")
    plt.axhline(0.4, color="#6d597a", linestyle="--", linewidth=1.2, label="Порог 0.40")
    plt.title("Вероятность реакции по вопросам")
    plt.xlabel("Код вопроса")
    plt.ylabel("Вероятность")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-question deception probability.")
    parser.add_argument("--predictions-path", type=Path, default=Path("reports/predictions_inference.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("reports/figures/per_question_scores.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_plot(args.predictions_path, args.output_path)
    print(f"[plot] saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
