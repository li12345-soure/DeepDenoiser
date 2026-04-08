import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare raw TF vs TFLite model_input/preds")
    parser.add_argument("--tf_npz", required=True, help="orig_raw_debug.npz")
    parser.add_argument("--tflite_npz", required=True, help="result_builtin_float.npz")
    args = parser.parse_args()

    tf_path = Path(args.tf_npz)
    tfl_path = Path(args.tflite_npz)

    if not tf_path.exists():
        raise FileNotFoundError(tf_path.resolve())
    if not tfl_path.exists():
        raise FileNotFoundError(tfl_path.resolve())

    with np.load(tf_path, allow_pickle=True) as a, np.load(tfl_path, allow_pickle=True) as b:
        print("[INFO] TF keys     :", a.files)
        print("[INFO] TFLite keys :", b.files)

        x_tf = np.asarray(a["model_input"], dtype=np.float32)
        p_tf = np.asarray(a["preds"], dtype=np.float32)

        x_tfl = np.asarray(b["model_input"], dtype=np.float32)
        p_tfl = np.asarray(b["preds"], dtype=np.float32)

    print("[INFO] model_input shape:", x_tf.shape, x_tfl.shape)
    print("[INFO] preds shape      :", p_tf.shape, p_tfl.shape)

    if x_tf.shape != x_tfl.shape:
        raise ValueError(f"model_input shape mismatch: {x_tf.shape} vs {x_tfl.shape}")
    if p_tf.shape != p_tfl.shape:
        raise ValueError(f"preds shape mismatch: {p_tf.shape} vs {p_tfl.shape}")

    dx = x_tfl - x_tf
    dp = p_tfl - p_tf

    print("\n[RESULT] model_input")
    print("  mae      :", float(np.mean(np.abs(dx))))
    print("  rmse     :", float(np.sqrt(np.mean(dx ** 2))))
    print("  max_abs  :", float(np.max(np.abs(dx))))

    print("\n[RESULT] preds")
    print("  mae      :", float(np.mean(np.abs(dp))))
    print("  rmse     :", float(np.sqrt(np.mean(dp ** 2))))
    print("  max_abs  :", float(np.max(np.abs(dp))))

    print("\n[RESULT] sample stats")
    print("  TF preds min/max :", float(p_tf.min()), float(p_tf.max()))
    print("  TFL preds min/max:", float(p_tfl.min()), float(p_tfl.max()))


if __name__ == "__main__":
    main()
