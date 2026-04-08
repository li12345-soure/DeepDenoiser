import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot DeepDenoiser NPZ result")
    parser.add_argument("--npz", default="result_from_dataset.npz", help="Path to result npz")
    parser.add_argument("--save", default="result_plot.png", help="Output image path")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path.resolve()}")

    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
        print("[INFO] Keys in npz:", keys)

        if "input_waveform" not in z or "denoised_waveform" not in z:
            raise KeyError("Expected keys 'input_waveform' and 'denoised_waveform' in result npz")

        x = np.asarray(z["input_waveform"], dtype=np.float32)
        y = np.asarray(z["denoised_waveform"], dtype=np.float32)
        dt = float(np.asarray(z["dt"]).reshape(-1)[0]) if "dt" in z else 0.01

    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    n = min(x.shape[0], y.shape[0])
    c = min(x.shape[1], y.shape[1])
    x = x[:n, :c]
    y = y[:n, :c]

    t = np.arange(n) * dt

    fig, axes = plt.subplots(c, 1, figsize=(12, 3 * c), sharex=True)
    if c == 1:
        axes = [axes]

    for i in range(c):
        axes[i].plot(t, x[:, i], label="input_waveform")
        axes[i].plot(t, y[:, i], label="denoised_waveform")
        axes[i].set_ylabel(f"Ch {i}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("DeepDenoiser Result")
    fig.tight_layout()

    save_path = Path(args.save)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved plot to: {save_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
