#!/usr/bin/env python3
"""Batch waveform comparison for original TF DeepDenoiser vs INT8 TFLite logits."""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_waveform_original_vs_int8 import (  # noqa: E402
    CSV_FIELDS as WAVEFORM_CSV_FIELDS,
    build_compare_args,
    compute_row,
    finite_mean,
    load_feature_from_waveform,
    parse_components,
    reconstruct_waveform,
    run_tf_logits,
    run_tflite_int8_logits,
    softmax,
    waveform_to_model_input,
)


CSV_FIELDS = ["npz", *WAVEFORM_CSV_FIELDS]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch compare original float DeepDenoiser vs B_d4_r4 INT8 TFLite waveforms"
    )
    parser.add_argument("--teacher_checkpoint_dir", required=True)
    parser.add_argument("--teacher_depth", type=int, required=True)
    parser.add_argument("--teacher_filters_root", type=int, required=True)
    parser.add_argument("--teacher_drop_rate", type=float, required=True)
    parser.add_argument("--student_int8_model", required=True)
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--mask_channel", type=int, default=1)
    parser.add_argument("--component", default="all")
    parser.add_argument("--save_csv", required=True)
    return parser


def list_npz_files(npz_dir: Path, max_files: int) -> list[Path]:
    files = sorted(npz_dir.glob("*.npz"))
    if max_files > 0:
        files = files[:max_files]
    return files


def evaluate_one(
    npz_path: Path,
    checkpoint_dir: Path,
    int8_model: Path,
    compare_args: argparse.Namespace,
    component_arg: str,
    mask_channel: int,
) -> list[dict[str, object]]:
    X_input, batch_waveform, _dt = load_feature_from_waveform(npz_path)
    X_check, noisy_signal, _nbt, _nt, nch = waveform_to_model_input(batch_waveform)
    if X_check.shape != X_input.shape:
        raise RuntimeError(f"Feature shape mismatch for {npz_path}: {X_check.shape} vs {X_input.shape}")
    if not np.allclose(X_check, X_input, rtol=1e-5, atol=1e-6):
        print(f"[WARN] Recomputed model input differs slightly for {npz_path}")

    components = parse_components(component_arg, nch)
    tf_logits = run_tf_logits(compare_args, checkpoint_dir, X_input)
    _int8_logits_q, int8_logits, _meta = run_tflite_int8_logits(int8_model, X_input)

    if tf_logits.shape != int8_logits.shape:
        raise RuntimeError(f"Logit shape mismatch for {npz_path}: {tf_logits.shape} vs {int8_logits.shape}")
    if mask_channel < 0 or mask_channel >= tf_logits.shape[-1]:
        raise ValueError(f"--mask_channel {mask_channel} out of range for {tf_logits.shape[-1]} classes")

    tf_prob = softmax(tf_logits, axis=-1).astype(np.float32)
    int8_prob = softmax(int8_logits, axis=-1).astype(np.float32)
    target_len = int(batch_waveform.shape[1])

    rows = []
    for component in components:
        sample_index = component
        if sample_index >= tf_prob.shape[0]:
            raise ValueError(
                f"Component {component} maps to sample {sample_index}, but logits batch is {tf_prob.shape[0]}"
            )

        tf_mask = tf_prob[sample_index, :, :, mask_channel]
        int8_mask = int8_prob[sample_index, :, :, mask_channel]
        tf_wave = reconstruct_waveform(tf_mask, noisy_signal[sample_index], target_len)
        int8_wave = reconstruct_waveform(int8_mask, noisy_signal[sample_index], target_len)

        row = compute_row(
            component,
            tf_wave,
            int8_wave,
            tf_mask,
            int8_mask,
            tf_prob[sample_index],
            int8_prob[sample_index],
        )
        rows.append({"npz": str(npz_path), **row})

    return rows


def print_summary(files_count: int, rows: list[dict[str, object]]) -> None:
    print("\n[SUMMARY]")
    print(f"  files                    : {files_count}")
    print(f"  rows                     : {len(rows)}")
    print(f"  mean waveform_rmse       : {finite_mean(rows, 'waveform_rmse')}")
    print(f"  mean waveform_corrcoef   : {finite_mean(rows, 'waveform_corrcoef')}")
    print(f"  mean waveform_relative_l2: {finite_mean(rows, 'waveform_relative_l2')}")
    print(f"  mean mask_argmax_acc     : {finite_mean(rows, 'mask_argmax_acc')}")
    print(f"  mean softmax_mae         : {finite_mean(rows, 'softmax_mae')}")
    print(f"  mean softmax_rmse        : {finite_mean(rows, 'softmax_rmse')}")


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.max_files < 0:
        raise ValueError("--max_files must be >= 0")

    checkpoint_dir = Path(args.teacher_checkpoint_dir)
    int8_model = Path(args.student_int8_model)
    npz_dir = Path(args.npz_dir)
    save_csv = Path(args.save_csv)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Teacher checkpoint dir not found: {checkpoint_dir.resolve()}")
    if not int8_model.exists():
        raise FileNotFoundError(f"Student INT8 model not found: {int8_model.resolve()}")
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ dir not found: {npz_dir.resolve()}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"--npz_dir is not a directory: {npz_dir.resolve()}")

    npz_files = list_npz_files(npz_dir, args.max_files)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {npz_dir.resolve()}")

    compare_args = build_compare_args(args)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    with save_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for index, npz_path in enumerate(npz_files, start=1):
            print(f"[{index}/{len(npz_files)}] {npz_path}")
            rows = evaluate_one(
                npz_path,
                checkpoint_dir,
                int8_model,
                compare_args,
                args.component,
                args.mask_channel,
            )
            for row in rows:
                writer.writerow(row)
            fp.flush()
            all_rows.extend(rows)
            print(
                "  rows={rows} mean_waveform_rmse={rmse:.8g} mean_corrcoef={corr:.8g} "
                "mean_mask_argmax_acc={acc:.8g}".format(
                    rows=len(rows),
                    rmse=finite_mean(rows, "waveform_rmse"),
                    corr=finite_mean(rows, "waveform_corrcoef"),
                    acc=finite_mean(rows, "mask_argmax_acc"),
                )
            )

    print_summary(len(npz_files), all_rows)
    print(f"\n[OK] Wrote CSV: {save_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
