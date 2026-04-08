import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_2d(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x


def main():
    parser = argparse.ArgumentParser(description="Compare original DeepDenoiser output vs TFLite output")
    parser.add_argument("--orig", default=r".\output\results\BK_BKS_2008110908041793.npz", help="Original TF result npz")
    parser.add_argument("--tflite", default=r".\result_from_dataset.npz", help="TFLite result npz")
    parser.add_argument("--save", default=r".\compare_original_vs_tflite.png", help="Output plot path")
    args = parser.parse_args()

    orig_path = Path(args.orig)
    tflite_path = Path(args.tflite)

    if not orig_path.exists():
        raise FileNotFoundError(f"Original result not found: {orig_path.resolve()}")
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite result not found: {tflite_path.resolve()}")

    with np.load(orig_path, allow_pickle=True) as z:
        print("[INFO] Original keys:", z.files)
        if "data" not in z:
            raise KeyError("Expected key 'data' in original result npz")
        orig_data = ensure_2d(z["data"])
        orig_t0 = z["t0"] if "t0" in z else None

    with np.load(tflite_path, allow_pickle=True) as z:
        print("[INFO] TFLite keys:", z.files)
        if "input_waveform" not in z or "denoised_waveform" not in z:
            raise KeyError("Expected keys 'input_waveform' and 'denoised_waveform' in TFLite result npz")
        input_wave = ensure_2d(z["input_waveform"])
        tflite_data = ensure_2d(z["denoised_waveform"])
        dt = float(np.asarray(z["dt"]).reshape(-1)[0]) if "dt" in z else 0.01

    n = min(orig_data.shape[0], tflite_data.shape[0], input_wave.shape[0])
    c = min(orig_data.shape[1], tflite_data.shape[1], input_wave.shape[1])

    orig_data = orig_data[:n, :c]
    tflite_data = tflite_data[:n, :c]
    input_wave = input_wave[:n, :c]
    t = np.arange(n) * dt

    fig, axes = plt.subplots(c, 1, figsize=(12, 3.2 * c), sharex=True)
    if c == 1:
        axes = [axes]

    for i in range(c):
        axes[i].plot(t, input_wave[:, i], label="Input waveform", alpha=0.5)
        axes[i].plot(t, orig_data[:, i], label="Original TF output", linewidth=1.2)
        axes[i].plot(t, tflite_data[:, i], label="TFLite output", linewidth=1.2)
        axes[i].set_ylabel(f"Ch {i}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")

    title = "Original TF vs TFLite"
    if orig_t0 is not None:
        title += f" | t0={orig_t0}"
    fig.suptitle(title)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    save_path = Path(args.save)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved plot to: {save_path.resolve()}")

    diff = tflite_data - orig_data
    print("[INFO] Compare summary:")
    print("  orig shape   :", orig_data.shape)
    print("  tflite shape :", tflite_data.shape)
    print("  mae          :", float(np.mean(np.abs(diff))))
    print("  rmse         :", float(np.sqrt(np.mean(diff ** 2))))

    plt.show()


if __name__ == "__main__":
    main()
