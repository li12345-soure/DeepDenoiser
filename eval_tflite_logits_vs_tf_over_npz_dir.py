#!/usr/bin/env python3
"""Evaluate float32 or int8 TFLite logits against a TF checkpoint over NPZ files."""

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from compare_tf_vs_int8_logits import (  # noqa: E402
    load_feature_from_waveform,
    run_tf_logits,
    run_tflite_int8_logits,
    softmax,
)


CSV_FIELDS = [
    "npz",
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
    parser = argparse.ArgumentParser(
        description="Evaluate TF checkpoint logits vs float32/int8 TFLite logits over NPZ files"
    )
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing TensorFlow checkpoint")
    parser.add_argument("--tflite_model", required=True, help="TFLite logits model path")
    parser.add_argument("--npz_dir", required=True, help="Directory containing waveform .npz files")
    parser.add_argument("--max_files", type=int, default=0, help="Maximum NPZ files to evaluate; 0 means all")
    parser.add_argument("--depth", type=int, default=6, help="DeepDenoiser depth")
    parser.add_argument("--filters_root", type=int, default=8, help="DeepDenoiser filters_root")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for TF graph construction")
    parser.add_argument("--save_csv", required=True, help="CSV output path")
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


def run_tflite_float_logits(model_path: Path, X_input: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            f"Expected 1 input and 1 output, got {len(input_details)} inputs / {len(output_details)} outputs"
        )

    interpreter.resize_tensor_input(
        input_details[0]["index"],
        [1, X_input.shape[1], X_input.shape[2], X_input.shape[3]],
        strict=False,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if input_details[0]["dtype"] != np.float32:
        raise RuntimeError(f"Expected float32 input, got {input_details[0]['dtype']}")
    if output_details[0]["dtype"] != np.float32:
        raise RuntimeError(f"Expected float32 output, got {output_details[0]['dtype']}")

    logits_list = []
    for i in range(X_input.shape[0]):
        x_i = X_input[i : i + 1].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], x_i)
        interpreter.invoke()
        y_i = interpreter.get_tensor(output_details[0]["index"]).astype(np.float32)
        logits_list.append(y_i)

    return np.concatenate(logits_list, axis=0).astype(np.float32)


def get_tflite_input_dtype(model_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    if len(input_details) != 1:
        raise RuntimeError(f"Expected 1 input, got {len(input_details)}")
    return input_details[0]["dtype"]


def run_tflite_logits(model_path: Path, X_input: np.ndarray, input_dtype) -> np.ndarray:
    if input_dtype == np.int8:
        _logits_q, logits_f, _meta = run_tflite_int8_logits(model_path, X_input)
        return logits_f.astype(np.float32)
    if input_dtype == np.float32:
        return run_tflite_float_logits(model_path, X_input)
    raise RuntimeError(f"Unsupported TFLite input dtype: {input_dtype}")


def compute_metrics(tf_logits: np.ndarray, tflite_logits: np.ndarray) -> dict[str, float]:
    if tf_logits.shape != tflite_logits.shape:
        raise RuntimeError(f"Logit shape mismatch: TF {tf_logits.shape} vs TFLite {tflite_logits.shape}")

    logits_diff = tflite_logits - tf_logits
    tf_cls = np.argmax(tf_logits, axis=-1)
    tflite_cls = np.argmax(tflite_logits, axis=-1)

    tf_prob = softmax(tf_logits, axis=-1).astype(np.float32)
    tflite_prob = softmax(tflite_logits, axis=-1).astype(np.float32)
    prob_diff = tflite_prob - tf_prob

    return {
        "logits_mae": float(np.mean(np.abs(logits_diff))),
        "logits_rmse": float(np.sqrt(np.mean(logits_diff**2))),
        "logits_max_abs": float(np.max(np.abs(logits_diff))),
        "logits_argmax_acc": float(np.mean(tf_cls == tflite_cls)),
        "softmax_mae": float(np.mean(np.abs(prob_diff))),
        "softmax_rmse": float(np.sqrt(np.mean(prob_diff**2))),
        "softmax_max_abs": float(np.max(np.abs(prob_diff))),
        "softmax_argmax_acc": float(
            np.mean(np.argmax(tf_prob, axis=-1) == np.argmax(tflite_prob, axis=-1))
        ),
    }


def evaluate_one(
    npz_path: Path,
    checkpoint_dir: Path,
    tflite_model: Path,
    compare_args: argparse.Namespace,
    tflite_input_dtype,
) -> dict[str, object]:
    X_input, _batch_waveform, _dt = load_feature_from_waveform(npz_path)
    tf_logits = run_tf_logits(compare_args, checkpoint_dir, X_input)
    tflite_logits = run_tflite_logits(tflite_model, X_input, tflite_input_dtype)
    metrics = compute_metrics(tf_logits, tflite_logits)
    return {"npz": str(npz_path), **metrics}


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
    tflite_model = Path(args.tflite_model)
    npz_dir = Path(args.npz_dir)
    save_csv = Path(args.save_csv)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir.resolve()}")
    if not tflite_model.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_model.resolve()}")
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ dir not found: {npz_dir.resolve()}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"--npz_dir is not a directory: {npz_dir.resolve()}")

    npz_files = list_npz_files(npz_dir, args.max_files)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {npz_dir.resolve()}")

    tflite_input_dtype = get_tflite_input_dtype(tflite_model)
    print(f"[INFO] TFLite input dtype: {tflite_input_dtype}")
    if tflite_input_dtype == np.int8:
        print("[INFO] Using run_tflite_int8_logits from compare_tf_vs_int8_logits.py")
    elif tflite_input_dtype == np.float32:
        print("[INFO] Using run_tflite_float_logits")
    else:
        raise RuntimeError(f"Unsupported TFLite input dtype: {tflite_input_dtype}")

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    compare_args = build_compare_args(args)
    rows = []

    with save_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for index, npz_path in enumerate(npz_files, start=1):
            print(f"[{index}/{len(npz_files)}] {npz_path}")
            row = evaluate_one(npz_path, checkpoint_dir, tflite_model, compare_args, tflite_input_dtype)
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
