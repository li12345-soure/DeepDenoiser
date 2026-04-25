#!/usr/bin/env python3
"""Compare original DeepDenoiser TF checkpoint vs INT8 TFLite at waveform level."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from compare_tf_vs_int8_logits import (  # noqa: E402
    load_feature_from_waveform,
    run_tf_logits,
    run_tflite_int8_logits,
    softmax,
)
from run_tflite_on_npz_fixed import (  # noqa: E402
    FS,
    NFFT,
    NPERSEG,
    load_npz_waveform,
    waveform_to_model_input,
)


CSV_FIELDS = [
    "component",
    "waveform_mae",
    "waveform_rmse",
    "waveform_max_abs",
    "waveform_relative_l2",
    "waveform_corrcoef",
    "waveform_cosine",
    "mask_mae",
    "mask_rmse",
    "mask_argmax_acc",
    "softmax_mae",
    "softmax_rmse",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare original float DeepDenoiser vs B_d4_r4 INT8 TFLite waveforms"
    )
    parser.add_argument("--teacher_checkpoint_dir", required=True)
    parser.add_argument("--teacher_depth", type=int, required=True)
    parser.add_argument("--teacher_filters_root", type=int, required=True)
    parser.add_argument("--teacher_drop_rate", type=float, required=True)
    parser.add_argument("--student_int8_model", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_channel", type=int, default=1)
    parser.add_argument("--component", default="all")
    return parser


def build_compare_args(args):
    return SimpleNamespace(
        mode="pred",
        depth=int(args.teacher_depth),
        filters_root=int(args.teacher_filters_root),
        drop_rate=float(args.teacher_drop_rate),
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


def inspect_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
        print("[INFO] NPZ keys:", keys)
        for key in keys:
            value = z[key]
            shape = getattr(value, "shape", ())
            dtype = getattr(value, "dtype", None)
            print(f"[INFO]   {key}: shape={shape} dtype={dtype}")

    waveform, meta = load_npz_waveform(npz_path)
    dt = float(meta.get("dt", 0.01))
    print(f"[INFO] waveform shape: {waveform.shape} [nt, chn]")
    print(f"[INFO] dt={dt}")
    return waveform, dt


def parse_components(component_arg, nch):
    text = str(component_arg).strip().lower()
    if text == "all":
        return list(range(nch))

    components = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token.startswith("ch"):
            token = token[2:]
        if token.startswith("component_"):
            token = token[len("component_") :]
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            raise ValueError(f"Unsupported --component token: {token!r}. Use all, 0, 1, 2, or comma lists.")
        if idx < 0 or idx >= nch:
            raise ValueError(f"Component {idx} out of range for {nch} channels")
        if idx not in components:
            components.append(idx)

    if not components:
        raise ValueError("--component did not select any component")
    return components


def finite_mean(rows, key):
    vals = []
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def align_1d(x, n):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    return np.pad(x, (0, n - x.shape[0]), mode="constant").astype(np.float32)


def reconstruct_waveform(mask, noisy_signal_one, target_len):
    complex_spec = noisy_signal_one[..., 0] + 1j * noisy_signal_one[..., 1]
    _, waveform = scipy.signal.istft(
        complex_spec * mask,
        fs=FS,
        nperseg=NPERSEG,
        nfft=NFFT,
        boundary="zeros",
    )
    return align_1d(waveform, target_len)


def mae(diff):
    return float(np.mean(np.abs(diff)))


def rmse(diff):
    return float(np.sqrt(np.mean(diff**2)))


def safe_relative_l2(diff, ref):
    denom = float(np.linalg.norm(ref))
    if denom <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(diff) / denom)


def safe_corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape[0] < 2:
        return float("nan")
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 1.0 if np.allclose(a, b) else float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else float("nan")
    return float(np.dot(a, b) / denom)


def compute_row(component, tf_wave, int8_wave, tf_mask, int8_mask, tf_prob, int8_prob):
    wave_diff = int8_wave - tf_wave
    mask_diff = int8_mask - tf_mask
    prob_diff = int8_prob - tf_prob
    mask_argmax_acc = float(np.mean(np.argmax(tf_prob, axis=-1) == np.argmax(int8_prob, axis=-1)))

    return {
        "component": int(component),
        "waveform_mae": mae(wave_diff),
        "waveform_rmse": rmse(wave_diff),
        "waveform_max_abs": float(np.max(np.abs(wave_diff))),
        "waveform_relative_l2": safe_relative_l2(wave_diff, tf_wave),
        "waveform_corrcoef": safe_corrcoef(tf_wave, int8_wave),
        "waveform_cosine": safe_cosine(tf_wave, int8_wave),
        "mask_mae": mae(mask_diff),
        "mask_rmse": rmse(mask_diff),
        "mask_argmax_acc": mask_argmax_acc,
        "softmax_mae": mae(prob_diff),
        "softmax_rmse": rmse(prob_diff),
    }


def write_csv(rows, csv_path):
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_component(
    out_png,
    component,
    input_wave,
    tf_wave,
    int8_wave,
    tf_mask,
    int8_mask,
    dt,
):
    n = min(input_wave.shape[0], tf_wave.shape[0], int8_wave.shape[0])
    input_wave = input_wave[:n]
    tf_wave = tf_wave[:n]
    int8_wave = int8_wave[:n]
    residual = int8_wave - tf_wave
    time_axis = np.arange(n, dtype=np.float32) * float(dt)
    mask_diff = int8_mask - tf_mask

    fig, axes = plt.subplots(7, 1, figsize=(15, 18), sharex=False)
    axes[0].plot(time_axis, input_wave, color="black", linewidth=0.7)
    axes[0].set_title(f"Input waveform | component {component}")
    axes[0].set_ylabel("amp")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, tf_wave, color="#1f77b4", linewidth=0.7)
    axes[1].set_title("Original float reconstructed waveform")
    axes[1].set_ylabel("amp")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, int8_wave, color="#ff7f0e", linewidth=0.7)
    axes[2].set_title("B_d4_r4 INT8 reconstructed waveform")
    axes[2].set_ylabel("amp")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_axis, residual, color="#d62728", linewidth=0.7)
    axes[3].set_title("Residual = B_int8 - original")
    axes[3].set_ylabel("amp")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    im0 = axes[4].imshow(tf_mask, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[4].set_title("Original mask heatmap")
    axes[4].set_ylabel("freq bin")
    fig.colorbar(im0, ax=axes[4], fraction=0.018, pad=0.01)

    im1 = axes[5].imshow(int8_mask, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[5].set_title("B_d4_r4 INT8 mask heatmap")
    axes[5].set_ylabel("freq bin")
    fig.colorbar(im1, ax=axes[5], fraction=0.018, pad=0.01)

    max_abs = float(np.max(np.abs(mask_diff)))
    lim = max(max_abs, 1e-6)
    im2 = axes[6].imshow(
        mask_diff,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
    )
    axes[6].set_title("Mask difference heatmap = B_int8 - original")
    axes[6].set_ylabel("freq bin")
    axes[6].set_xlabel("frame")
    fig.colorbar(im2, ax=axes[6], fraction=0.018, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_summary(rows):
    lines = ["[SUMMARY]"]
    for row in rows:
        lines.append(
            "  component {component}: waveform_rmse={waveform_rmse:.8g} "
            "corrcoef={waveform_corrcoef:.8g} relative_l2={waveform_relative_l2:.8g}".format(**row)
        )
    lines.append(f"  mean waveform_rmse    : {finite_mean(rows, 'waveform_rmse'):.8g}")
    lines.append(f"  mean waveform_corrcoef: {finite_mean(rows, 'waveform_corrcoef'):.8g}")
    lines.append(f"  mean mask_argmax_acc  : {finite_mean(rows, 'mask_argmax_acc'):.8g}")
    return "\n".join(lines)


def main():
    args = build_arg_parser().parse_args()

    teacher_checkpoint_dir = Path(args.teacher_checkpoint_dir)
    student_int8_model = Path(args.student_int8_model)
    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)

    if not teacher_checkpoint_dir.exists():
        raise FileNotFoundError(f"Teacher checkpoint dir not found: {teacher_checkpoint_dir.resolve()}")
    if not student_int8_model.exists():
        raise FileNotFoundError(f"Student INT8 model not found: {student_int8_model.resolve()}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path.resolve()}")
    if args.mask_channel < 0:
        raise ValueError("--mask_channel must be >= 0")

    out_dir.mkdir(parents=True, exist_ok=True)

    inspect_npz(npz_path)
    X_input, batch_waveform, dt = load_feature_from_waveform(npz_path)
    print(f"[INFO] prepared waveform shape: {batch_waveform.shape} [batch, nt, chn]")
    print(f"[INFO] model input shape: {X_input.shape}")

    X_check, noisy_signal, nbt, nt, nch = waveform_to_model_input(batch_waveform)
    if X_check.shape != X_input.shape:
        raise RuntimeError(f"Feature shape mismatch: {X_check.shape} vs {X_input.shape}")
    if not np.allclose(X_check, X_input, rtol=1e-5, atol=1e-6):
        print("[WARN] Recomputed model input differs slightly from load_feature_from_waveform output")

    print(
        "[INFO] waveform reconstruction uses STFT params from existing code: "
        f"fs={FS}, nperseg={NPERSEG}, nfft={NFFT}, boundary=zeros"
    )
    if abs(float(dt) - (1.0 / float(FS))) > 1e-9:
        print(f"[INFO] input dt={dt}; prepared/reconstructed waveforms are plotted at effective dt={1.0 / FS}")
    plot_dt = 1.0 / float(FS)

    components = parse_components(args.component, nch)
    print(f"[INFO] selected components: {components}")

    compare_args = build_compare_args(args)
    tf_logits = run_tf_logits(compare_args, teacher_checkpoint_dir, X_input)
    print(f"[INFO] original TF logits shape: {tf_logits.shape}")

    _int8_logits_q, int8_logits, quant_meta = run_tflite_int8_logits(student_int8_model, X_input)
    print(f"[INFO] B_d4_r4 INT8 logits shape: {int8_logits.shape}")
    print(
        "[INFO] INT8 quantization: "
        f"input_scale={quant_meta['input_scale']} input_zero_point={quant_meta['input_zero_point']} "
        f"output_scale={quant_meta['output_scale']} output_zero_point={quant_meta['output_zero_point']}"
    )

    if tf_logits.shape != int8_logits.shape:
        raise RuntimeError(f"Logit shape mismatch: {tf_logits.shape} vs {int8_logits.shape}")
    if args.mask_channel >= tf_logits.shape[-1]:
        raise ValueError(f"--mask_channel {args.mask_channel} out of range for {tf_logits.shape[-1]} classes")

    tf_prob = softmax(tf_logits, axis=-1).astype(np.float32)
    int8_prob = softmax(int8_logits, axis=-1).astype(np.float32)

    rows = []
    png_paths = []
    target_len = int(batch_waveform.shape[1])
    for component in components:
        sample_index = component
        if sample_index >= tf_prob.shape[0]:
            raise ValueError(f"Component {component} maps to sample {sample_index}, but logits batch is {tf_prob.shape[0]}")

        tf_mask = tf_prob[sample_index, :, :, args.mask_channel]
        int8_mask = int8_prob[sample_index, :, :, args.mask_channel]
        tf_wave = reconstruct_waveform(tf_mask, noisy_signal[sample_index], target_len)
        int8_wave = reconstruct_waveform(int8_mask, noisy_signal[sample_index], target_len)
        input_wave = align_1d(batch_waveform[0, :, component], target_len)

        row = compute_row(
            component,
            tf_wave,
            int8_wave,
            tf_mask,
            int8_mask,
            tf_prob[sample_index],
            int8_prob[sample_index],
        )
        rows.append(row)

        out_png = out_dir / f"component_{component}.png"
        plot_component(out_png, component, input_wave, tf_wave, int8_wave, tf_mask, int8_mask, plot_dt)
        png_paths.append(out_png)
        print(
            f"[RESULT] component {component}: waveform_rmse={row['waveform_rmse']:.8g} "
            f"corrcoef={row['waveform_corrcoef']:.8g} relative_l2={row['waveform_relative_l2']:.8g} "
            f"mask_argmax_acc={row['mask_argmax_acc']:.8g}"
        )
        print(f"[OK] saved PNG: {out_png.resolve()}")

    csv_path = out_dir / "metrics.csv"
    write_csv(rows, csv_path)
    print(f"[OK] saved CSV: {csv_path.resolve()}")

    summary = format_summary(rows)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"[OK] saved summary: {summary_path.resolve()}")

    print("[OUTPUT]")
    for path in png_paths:
        print(f"  PNG: {path.resolve()}")
    print(f"  CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
