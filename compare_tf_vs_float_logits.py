import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from model import UNet
from run_tflite_on_npz_fixed import (
    load_npz_waveform,
    prepare_waveform,
    waveform_to_model_input,
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def load_feature_from_waveform(npz_path: Path, dt_override=None, target_nt=3000):
    data, meta = load_npz_waveform(npz_path)
    dt = dt_override if dt_override is not None else float(meta.get("dt", 0.01))
    batch_waveform = prepare_waveform(data, dt=dt, target_nt=target_nt)
    X_input, noisy_signal, nbt, nt, nch = waveform_to_model_input(batch_waveform)
    return X_input.astype(np.float32), batch_waveform, dt


def run_tf_logits(checkpoint_dir: Path, X_input: np.ndarray) -> np.ndarray:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    model = UNet(mode="pred")
    latest_ckpt = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir.resolve()}")

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    logits_list = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, latest_ckpt)

        for i in range(X_input.shape[0]):
            x_i = X_input[i:i + 1].astype(np.float32)
            y_i = sess.run(model.logits, feed_dict={model.X: x_i})
            logits_list.append(y_i)

    logits = np.concatenate(logits_list, axis=0).astype(np.float32)
    return logits


def run_tflite_float_logits(model_path: Path, X_input: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            f"Expected 1 input and 1 output, got {len(input_details)} inputs / {len(output_details)} outputs"
        )

    logits_list = []
    for i in range(X_input.shape[0]):
        x_i = X_input[i:i + 1].astype(np.float32)
        interpreter.resize_tensor_input(input_details[0]["index"], list(x_i.shape), strict=False)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if input_details[0]["dtype"] != np.float32:
            raise RuntimeError(f"Expected float32 input, got {input_details[0]['dtype']}")
        if output_details[0]["dtype"] != np.float32:
            raise RuntimeError(f"Expected float32 output, got {output_details[0]['dtype']}")

        interpreter.set_tensor(input_details[0]["index"], x_i)
        interpreter.invoke()
        y_i = interpreter.get_tensor(output_details[0]["index"]).astype(np.float32)
        logits_list.append(y_i)

    logits = np.concatenate(logits_list, axis=0).astype(np.float32)
    return logits


def main():
    parser = argparse.ArgumentParser(description="Compare original TF logits vs float builtin-only TFLite logits")
    parser.add_argument("--checkpoint_dir", required=True, help=r".\model\190614-104802")
    parser.add_argument("--float_model", required=True, help=r".\deepdenoiser_float_logits_builtin.tflite")
    parser.add_argument("--npz", required=True, help=r".\Dataset\pred\BK_BKS_2008110908041793.npz")
    parser.add_argument("--dt", type=float, default=None, help="Override dt if needed")
    parser.add_argument("--target_nt", type=int, default=3000, help="Waveform length before STFT")
    parser.add_argument("--save_npz", default="compare_tf_vs_float_logits.npz", help="Optional output npz")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    float_model = Path(args.float_model)
    npz_path = Path(args.npz)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir.resolve()}")
    if not float_model.exists():
        raise FileNotFoundError(f"Float model not found: {float_model.resolve()}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Waveform npz not found: {npz_path.resolve()}")

    X_input, batch_waveform, dt = load_feature_from_waveform(npz_path, dt_override=args.dt, target_nt=args.target_nt)
    print(f"[INFO] model_input shape: {X_input.shape}")
    print(f"[INFO] input waveform shape: {batch_waveform.shape}")
    print(f"[INFO] dt={dt}")

    tf_logits = run_tf_logits(checkpoint_dir, X_input)
    print(f"[INFO] TF logits shape: {tf_logits.shape}")

    float_logits = run_tflite_float_logits(float_model, X_input)
    print(f"[INFO] Float logits shape: {float_logits.shape}")

    diff = float_logits - tf_logits

    tf_cls = np.argmax(tf_logits, axis=-1)
    float_cls = np.argmax(float_logits, axis=-1)
    argmax_acc = float(np.mean(tf_cls == float_cls))

    tf_prob = softmax(tf_logits, axis=-1).astype(np.float32)
    float_prob = softmax(float_logits, axis=-1).astype(np.float32)
    prob_diff = float_prob - tf_prob
    prob_argmax_acc = float(np.mean(np.argmax(tf_prob, axis=-1) == np.argmax(float_prob, axis=-1)))

    print("\n[RESULT] logits")
    print("  mae         :", float(np.mean(np.abs(diff))))
    print("  rmse        :", float(np.sqrt(np.mean(diff ** 2))))
    print("  max_abs     :", float(np.max(np.abs(diff))))
    print("  argmax_acc  :", argmax_acc)

    print("\n[RESULT] softmax(logits)")
    print("  mae         :", float(np.mean(np.abs(prob_diff))))
    print("  rmse        :", float(np.sqrt(np.mean(prob_diff ** 2))))
    print("  max_abs     :", float(np.max(np.abs(prob_diff))))
    print("  argmax_acc  :", prob_argmax_acc)

    if args.save_npz:
        out_path = Path(args.save_npz)
        np.savez(
            out_path,
            model_input=X_input.astype(np.float32),
            tf_logits=tf_logits.astype(np.float32),
            float_logits=float_logits.astype(np.float32),
            tf_prob=tf_prob.astype(np.float32),
            float_prob=float_prob.astype(np.float32),
            input_waveform=batch_waveform[0].astype(np.float32),
            dt=np.float32(dt),
        )
        print(f"\n[OK] Saved compare npz to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
