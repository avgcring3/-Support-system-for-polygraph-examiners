from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .preprocess import run_preprocess
    from .train import train_models
except ImportError:
    from preprocess import run_preprocess
    from train import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full DSS pipeline: preprocess + train.")
    parser.add_argument("--docx-dir", type=Path, default=Path("data/raw/docx"))
    parser.add_argument("--xdex-dir", type=Path, default=Path("data/raw/xdex"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--weak-threshold", type=float, default=0.55)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--feedback-path", type=Path, default=Path("data/processed/feedback_labels.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xdex_dir = args.xdex_dir if args.xdex_dir.exists() else None
    long_df, feat_df = run_preprocess(
        docx_dir=args.docx_dir,
        xdex_dir=xdex_dir,
        processed_dir=args.processed_dir,
        weak_label_threshold=args.weak_threshold,
    )
    print(f"[pipeline] preprocess complete: long={len(long_df)} feature={len(feat_df)}")
    summary = train_models(
        features_path=args.processed_dir / "polygram_features.csv",
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        preferred_label=args.label_col,
        feedback_path=args.feedback_path,
    )
    print("[pipeline] training complete")
    print(summary)


if __name__ == "__main__":
    main()
