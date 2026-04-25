#!/usr/bin/env python3
"""Evaluate DeepDenoiser INT8 TFLite logits over a directory of NPZ files."""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from compare_tf_vs_int8_logits import (
    load_feature_from_waveform,
    run_tf_logits,
    run_tflite_int8_logits,
    softmax,
)


CSV_FIELDS = [
    "npz",
    "dt",
    "num_examples",
    "logits_shape",
    "logits_mae",
    "logits_rmse",
    "logits_max_abs",
    "logits_argmax_acc",
    "softmax_mae",
    "softmax_rmse",
    "softmax_max_abs",
    "softmax_argmax_acc",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TF checkpoint vs INT8 TFLite logits over NPZ files")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing TensorFlow checkpoint")
    parser.add_argument("--int8_model", required=True, help="INT8 TFLite model path")
    parser.add_argument("--npz_dir", required=True, help="Directory containing waveform .npz files")
    parser.add_argument("--max_files", type=int, default=0, help="Maximum NPZ files to evaluate; 0 means all")
    parser.add_argument("--depth", type=int, default=4, help="DeepDenoiser depth")
    parser.add_argument("--filters_root", type=int, default=4, help="DeepDenoiser filters_root")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for TF graph construction")
    parser.add_argument("--save_csv", default="eval_int8_logits_over_npz_dir.csv", help="CSV output path")
    return parser


def build_compare_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mode="pred",
        depth=int(args.depth),
        filters_root=int(args.filters_root),
        drop_rate=float(args.drop_rate),
        kernel_size=[3, 3],
        pool_size=[2, 2],
        dilation_rate=[1, 1],
        filters_cap=None,
        decoder_width_mult=1.0,
        skip_bottleneck_mult=1.0,
        use_skip_bottleneck=0,
        batch_size=20,
        class_weights=[1, 1],
        loss_type="cross_entropy",
        weight_decay=0.0,
        optimizer="adam",
        learning_rate=0.001,
        decay_step=-1,
        decay_rate=0.9,
        momentum=0.9,
        summary=True,
        distill_enable=0,
        distill_alpha=0.5,
        distill_beta=0.5,
        distill_temperature=2.0,
    )


def list_npz_files(npz_dir: Path, max_files: int) -> list[Path]:
    files = sorted(npz_dir.glob("*.npz"))
    if max_files > 0:
        files = files[:max_files]
    return files


def shape_text(shape) -> str:
    return "x".join(str(dim) for dim in shape)


def compute_metrics(tf_logits: np.ndarray, int8_logits: np.ndarray) -> dict[str, float]:
    logits_diff = int8_logits - tf_logits
    tf_cls = np.argmax(tf_logits, axis=-1)
    int8_cls = np.argmax(int8_logits, axis=-1)

    tf_prob = softmax(tf_logits, axis=-1).astype(np.float32)
    int8_prob = softmax(int8_logits, axis=-1).astype(np.float32)
    prob_diff = int8_prob - tf_prob

    return {
        "logits_mae": float(np.mean(np.abs(logits_diff))),
        "logits_rmse": float(np.sqrt(np.mean(logits_diff**2))),
        "logits_max_abs": float(np.max(np.abs(logits_diff))),
        "logits_argmax_acc": float(np.mean(tf_cls == int8_cls)),
        "softmax_mae": float(np.mean(np.abs(prob_diff))),
        "softmax_rmse": float(np.sqrt(np.mean(prob_diff**2))),
        "softmax_max_abs": float(np.max(np.abs(prob_diff))),
        "softmax_argmax_acc": float(
            np.mean(np.argmax(tf_prob, axis=-1) == np.argmax(int8_prob, axis=-1))
        ),
    }


def evaluate_one(npz_path: Path, checkpoint_dir: Path, int8_model: Path, compare_args) -> dict[str, object]:
    X_input, _batch_waveform, dt = load_feature_from_waveform(npz_path)
    tf_logits = run_tf_logits(compare_args, checkpoint_dir, X_input)
    _, int8_logits, _ = run_tflite_int8_logits(int8_model, X_input)
    metrics = compute_metrics(tf_logits, int8_logits)

    return {
        "npz": str(npz_path),
        "dt": float(dt),
        "num_examples": int(X_input.shape[0]),
        "logits_shape": shape_text(tf_logits.shape),
        **metrics,
    }


def print_summary(rows: list[dict[str, object]]) -> None:
    keys = [
        "logits_mae",
        "logits_rmse",
        "logits_argmax_acc",
        "softmax_mae",
        "softmax_rmse",
        "softmax_argmax_acc",
    ]
    print("\n[SUMMARY]")
    print(f"  files                    : {len(rows)}")
    for key in keys:
        print(f"  {key + ' mean':25}: {float(np.mean([row[key] for row in rows]))}")


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.max_files < 0:
        raise ValueError("--max_files must be >= 0")

    checkpoint_dir = Path(args.checkpoint_dir)
    int8_model = Path(args.int8_model)
    npz_dir = Path(args.npz_dir)
    save_csv = Path(args.save_csv)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir.resolve()}")
    if not int8_model.exists():
        raise FileNotFoundError(f"INT8 model not found: {int8_model.resolve()}")
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ dir not found: {npz_dir.resolve()}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"--npz_dir is not a directory: {npz_dir.resolve()}")

    npz_files = list_npz_files(npz_dir, args.max_files)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {npz_dir.resolve()}")

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    compare_args = build_compare_args(args)
    rows = []

    with save_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for index, npz_path in enumerate(npz_files, start=1):
            print(f"[{index}/{len(npz_files)}] {npz_path}")
            row = evaluate_one(npz_path, checkpoint_dir, int8_model, compare_args)
            rows.append(row)
            writer.writerow(row)
            fp.flush()
            print(
                "  logits_mae={logits_mae:.8f} logits_rmse={logits_rmse:.8f} "
                "logits_argmax_acc={logits_argmax_acc:.6f} "
                "softmax_mae={softmax_mae:.8f} softmax_rmse={softmax_rmse:.8f} "
                "softmax_argmax_acc={softmax_argmax_acc:.6f}".format(**row)
            )

    print_summary(rows)
    print(f"\n[OK] Wrote CSV: {save_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
