import argparse
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

from deepdenoiser.model import ModelConfig, UNet
from deepdenoiser.train import build_arg_parser, set_config
from run_tflite_on_npz_fixed import (
    load_npz_waveform,
    prepare_waveform,
    waveform_to_model_input,
)


def set_arg_help(parser: argparse.ArgumentParser, dest: str, help_text: str) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.help = help_text
            return
    raise RuntimeError(f"Shared parser is missing expected argument: {dest}")


def quantize_int8(x_float: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Invalid input scale: {scale}")
    q = np.round(x_float / scale + zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def dequantize_int8(x_int8: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Invalid output scale: {scale}")
    return (x_int8.astype(np.float32) - zero_point) * scale


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


def build_script_arg_parser():
    parser = build_arg_parser()
    parser.description = "Compare original TF logits vs INT8 TFLite logits"
    parser.set_defaults(
        mode="pred",
        depth=4,
        filters_root=6,
        drop_rate=0.0,
        kernel_size=[3, 3],
        pool_size=[2, 2],
        dilation_rate=[1, 1],
        filters_cap=None,
        decoder_width_mult=1.0,
        skip_bottleneck_mult=1.0,
        use_skip_bottleneck=0,
    )
    set_arg_help(parser, "depth", "model depth (default: 4)")
    set_arg_help(parser, "filters_root", "filters root (default: 6)")
    set_arg_help(parser, "drop_rate", "drop out rate (default: 0.0)")
    parser.add_argument("--checkpoint_dir", required=True, help=r".\model\190614-104802")
    parser.add_argument("--int8_model", required=True, help=r".\deepdenoiser_int8_builtin.tflite")
    parser.add_argument("--npz", required=True, help=r".\Dataset\pred\BK_BKS_2008110908041793.npz")
    parser.add_argument("--dt", type=float, default=None, help="Override dt if needed")
    parser.add_argument("--target_nt", type=int, default=3000, help="Waveform length before STFT")
    parser.add_argument("--save_npz", default="compare_tf_vs_int8_logits.npz", help="Optional output npz")
    return parser


def build_pred_config(args):
    args.mode = "pred"
    data_reader = SimpleNamespace(
        X_shape=ModelConfig.X_shape,
        Y_shape=ModelConfig.Y_shape,
    )
    return set_config(args, data_reader)


def run_tf_logits(args, checkpoint_dir: Path, X_input: np.ndarray) -> np.ndarray:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    config = build_pred_config(args)
    model = UNet(config=config, mode="pred")

    latest_ckpt = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir.resolve()}")

    checkpoint_var_names = [name for name, _ in tf.train.list_variables(latest_ckpt)]
    graph_var_names = [var.op.name for var in tf.compat.v1.global_variables()]
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    logits_list = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        try:
            saver.restore(sess, latest_ckpt)
        except Exception:
            print(f"[DEBUG] Checkpoint path: {latest_ckpt}")
            print("[DEBUG] First 30 checkpoint variable names:")
            for name in checkpoint_var_names[:30]:
                print(f"  {name}")
            print("[DEBUG] First 30 graph global variable names:")
            for name in graph_var_names[:30]:
                print(f"  {name}")
            raise

        for i in range(X_input.shape[0]):
            x_i = X_input[i:i + 1].astype(np.float32)
            y_i = sess.run(model.logits, feed_dict={model.X: x_i})
            logits_list.append(y_i)

    logits = np.concatenate(logits_list, axis=0).astype(np.float32)
    return logits


def run_tflite_int8_logits(model_path: Path, X_input: np.ndarray):
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

    in_scale, in_zp = input_details[0]["quantization"]
    out_scale, out_zp = output_details[0]["quantization"]

    if input_details[0]["dtype"] != np.int8:
        raise RuntimeError(f"Expected int8 input, got {input_details[0]['dtype']}")
    if output_details[0]["dtype"] != np.int8:
        raise RuntimeError(f"Expected int8 output, got {output_details[0]['dtype']}")

    logits_q_list = []
    logits_f_list = []
    for i in range(X_input.shape[0]):
        x_i = X_input[i:i + 1].astype(np.float32)
        x_q = quantize_int8(x_i, in_scale, in_zp)

        interpreter.set_tensor(input_details[0]["index"], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(output_details[0]["index"]).astype(np.int8)
        y_f = dequantize_int8(y_q, out_scale, out_zp)

        logits_q_list.append(y_q)
        logits_f_list.append(y_f)

    logits_q = np.concatenate(logits_q_list, axis=0)
    logits_f = np.concatenate(logits_f_list, axis=0)
    meta = {
        "input_scale": float(in_scale),
        "input_zero_point": int(in_zp),
        "output_scale": float(out_scale),
        "output_zero_point": int(out_zp),
    }
    return logits_q, logits_f.astype(np.float32), meta


def main():
    args = build_script_arg_parser().parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    int8_model = Path(args.int8_model)
    npz_path = Path(args.npz)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir.resolve()}")
    if not int8_model.exists():
        raise FileNotFoundError(f"INT8 model not found: {int8_model.resolve()}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Waveform npz not found: {npz_path.resolve()}")

    X_input, batch_waveform, dt = load_feature_from_waveform(npz_path, dt_override=args.dt, target_nt=args.target_nt)
    print(f"[INFO] model_input shape: {X_input.shape}")
    print(f"[INFO] input waveform shape: {batch_waveform.shape}")
    print(f"[INFO] dt={dt}")

    tf_logits = run_tf_logits(args, checkpoint_dir, X_input)
    print(f"[INFO] TF logits shape: {tf_logits.shape}")

    int8_logits_q, int8_logits_f, meta = run_tflite_int8_logits(int8_model, X_input)
    print(f"[INFO] INT8 logits shape: {int8_logits_f.shape}")

    diff = int8_logits_f - tf_logits

    tf_cls = np.argmax(tf_logits, axis=-1)
    int8_cls = np.argmax(int8_logits_f, axis=-1)
    argmax_acc = float(np.mean(tf_cls == int8_cls))

    tf_prob = softmax(tf_logits, axis=-1).astype(np.float32)
    int8_prob = softmax(int8_logits_f, axis=-1).astype(np.float32)
    prob_diff = int8_prob - tf_prob
    prob_argmax_acc = float(np.mean(np.argmax(tf_prob, axis=-1) == np.argmax(int8_prob, axis=-1)))

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

    print("\n[QUANT]")
    print("  input_scale       :", meta["input_scale"])
    print("  input_zero_point  :", meta["input_zero_point"])
    print("  output_scale      :", meta["output_scale"])
    print("  output_zero_point :", meta["output_zero_point"])

    if args.save_npz:
        out_path = Path(args.save_npz)
        np.savez(
            out_path,
            model_input=X_input.astype(np.float32),
            tf_logits=tf_logits.astype(np.float32),
            int8_logits_q=int8_logits_q.astype(np.int8),
            int8_logits_f=int8_logits_f.astype(np.float32),
            tf_prob=tf_prob.astype(np.float32),
            int8_prob=int8_prob.astype(np.float32),
            input_waveform=batch_waveform[0].astype(np.float32),
            dt=np.float32(dt),
            input_scale=np.float32(meta["input_scale"]),
            input_zero_point=np.int32(meta["input_zero_point"]),
            output_scale=np.float32(meta["output_scale"]),
            output_zero_point=np.int32(meta["output_zero_point"]),
        )
        print(f"\n[OK] Saved compare npz to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
