#!/usr/bin/env python3
"""Convert AI4EPS/DeepDenoiser TF1-style checkpoint to TFLite.

Usage examples:

1) Float TFLite
python convert_deepdenoiser_tflite.py \
  --repo_root /path/to/DeepDenoiser \
  --checkpoint_dir /path/to/DeepDenoiser/model/190614-104802 \
  --output_tflite deepdenoiser_float.tflite \
  --output_nodes preds

2) Dynamic-range quantization
python convert_deepdenoiser_tflite.py \
  --repo_root /path/to/DeepDenoiser \
  --checkpoint_dir /path/to/DeepDenoiser/model/190614-104802 \
  --output_tflite deepdenoiser_dynamic.tflite \
  --quant_mode dynamic \
  --output_nodes preds

3) Full INT8 quantization
python convert_deepdenoiser_tflite.py \
  --repo_root /path/to/DeepDenoiser \
  --checkpoint_dir /path/to/DeepDenoiser/model/190614-104802 \
  --output_tflite deepdenoiser_int8.tflite \
  --quant_mode int8 \
  --calib_dir /path/to/calib_features \
  --output_nodes logits \
  --int8_io
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Generator, Iterable, List

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DeepDenoiser checkpoint to TFLite")
    parser.add_argument("--repo_root", required=True, help="Path to DeepDenoiser repo root")
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing TensorFlow checkpoint, e.g. model/190614-104802",
    )
    parser.add_argument("--output_tflite", required=True, help="Output .tflite path")
    parser.add_argument(
        "--output_nodes",
        default="preds",
        choices=["preds", "logits"],
        help="Which graph output to export",
    )
    parser.add_argument(
        "--quant_mode",
        default="none",
        choices=["none", "dynamic", "int8"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--calib_dir",
        default=None,
        help="Directory of precomputed network input features (*.npy or *.npz) for representative dataset",
    )
    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=200,
        help="Maximum number of representative samples to use",
    )
    parser.add_argument(
        "--int8_io",
        action="store_true",
        help="Force int8 input/output tensors (recommended only when inference graph has a single real input X)",
    )
    parser.add_argument(
        "--allow_aux_inputs",
        action="store_true",
        help=(
            "Export auxiliary inputs is_training/drop_rate too. Use this if you did NOT patch model.py to make "
            "them constants in pred mode."
        ),
    )
    parser.add_argument(
        "--fixed_batch_size",
        type=int,
        default=1,
        help="Informational only. Representative samples will be expanded to this batch size if needed.",
    )
    return parser.parse_args()


def ensure_repo_importable(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    pkg_dir = repo_root / "deepdenoiser"
    if not pkg_dir.exists():
        raise FileNotFoundError(f"Cannot find deepdenoiser/ under {repo_root}")
    sys.path.insert(0, str(pkg_dir))
    sys.path.insert(0, str(repo_root))



def load_feature_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        obj = np.load(path)
        if "X" in obj:
            arr = obj["X"]
        else:
            first_key = list(obj.keys())[0]
            arr = obj[first_key]
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 4:
        raise ValueError(f"Representative sample must be rank-4 after batch expansion, got shape {arr.shape} from {path}")
    return arr



def iter_feature_files(calib_dir: Path) -> List[Path]:
    files = sorted([*calib_dir.glob("*.npy"), *calib_dir.glob("*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npy or .npz feature files found under {calib_dir}")
    return files



def representative_dataset(calib_dir: Path, max_samples: int) -> Generator[List[np.ndarray], None, None]:
    files = iter_feature_files(calib_dir)
    for path in files[:max_samples]:
        arr = load_feature_file(path)
        yield [arr.astype(np.float32)]



def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_tflite = Path(args.output_tflite)

    ensure_repo_importable(repo_root)

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    model_module = importlib.import_module("model")
    UNet = getattr(model_module, "UNet")

    model = UNet(mode="pred")

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=sess_config) as sess:
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        sess.run(tf.compat.v1.global_variables_initializer())

        latest_ckpt = tf.train.latest_checkpoint(str(checkpoint_dir))
        if latest_ckpt is None:
            raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
        print(f"[INFO] Restoring checkpoint: {latest_ckpt}")
        saver.restore(sess, latest_ckpt)

        output_tensor = model.preds if args.output_nodes == "preds" else model.logits

        input_tensors: List[tf.Tensor] = [model.X]
        if args.allow_aux_inputs:
            input_tensors.extend([model.is_training, model.drop_rate])

        converter = tf.compat.v1.lite.TFLiteConverter.from_session(
            sess=sess,
            input_tensors=input_tensors,
            output_tensors=[output_tensor],
        )

        if args.quant_mode == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if args.quant_mode == "int8":
            if not args.calib_dir:
                raise ValueError("--quant_mode int8 requires --calib_dir with precomputed network input features")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: representative_dataset(Path(args.calib_dir), args.max_calib_samples)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if args.int8_io:
                if args.allow_aux_inputs:
                    raise ValueError(
                        "--int8_io cannot be combined with --allow_aux_inputs. Patch model.py so pred mode uses constant "
                        "is_training/drop_rate, then export a single-input graph."
                    )
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        try:
            tflite_model = converter.convert()
        except Exception:
            print("[ERROR] Builtin-only TFLite conversion failed.")
            raise

    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    output_tflite.write_bytes(tflite_model)
    print(f"[OK] Wrote {output_tflite}")
    print(f"[INFO] Size: {len(tflite_model) / 1024:.2f} KB")

    analysis_txt = output_tflite.with_suffix(output_tflite.suffix + ".analysis.txt")
    try:
        from io import StringIO
        import contextlib

        sio = StringIO()
        with contextlib.redirect_stdout(sio):
            tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)
        analysis_txt.write_text(sio.getvalue(), encoding="utf-8")
        print(f"[OK] Wrote analyzer report to {analysis_txt}")
    except Exception as exc:
        print(f"[WARN] Analyzer failed: {exc}")

    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        print("[INFO] Input details:")
        for d in interpreter.get_input_details():
            print(f"  - name={d['name']} shape={d['shape']} dtype={d['dtype']}")
        print("[INFO] Output details:")
        for d in interpreter.get_output_details():
            print(f"  - name={d['name']} shape={d['shape']} dtype={d['dtype']}")
    except Exception as exc:
        print(f"[WARN] TFLite interpreter validation failed: {exc}")


if __name__ == "__main__":
    main()
