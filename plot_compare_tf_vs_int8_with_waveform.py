import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot original waveform + TF logits/prob vs INT8 logits/prob from compare_tf_vs_int8_logits.npz"
    )
    parser.add_argument("--npz", required=True, help="Path to compare_tf_vs_int8_logits*.npz")
    parser.add_argument("--save", default="compare_tf_vs_int8_with_waveform.png", help="Output image path")
    parser.add_argument("--sample", type=int, default=0, help="Feature sample index in [0, batch*chn)")
    parser.add_argument("--row", type=int, default=15, help="Row index in [0, 31)")
    parser.add_argument("--wave_ch", type=int, default=0, help="Waveform channel index in [0, 3)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path.resolve())

    with np.load(npz_path, allow_pickle=True) as z:
        print("[INFO] keys:", z.files)
        tf_logits = np.asarray(z["tf_logits"], dtype=np.float32)
        int8_logits_f = np.asarray(z["int8_logits_f"], dtype=np.float32)
        tf_prob = np.asarray(z["tf_prob"], dtype=np.float32)
        int8_prob = np.asarray(z["int8_prob"], dtype=np.float32)
        input_waveform = np.asarray(z["input_waveform"], dtype=np.float32)
        dt = float(np.asarray(z["dt"]).reshape(-1)[0]) if "dt" in z else 0.01

    s = args.sample
    r = args.row
    wch = args.wave_ch

    if input_waveform.ndim == 1:
        input_waveform = input_waveform[:, np.newaxis]
    elif input_waveform.ndim == 3 and input_waveform.shape[0] == 1:
        input_waveform = input_waveform[0]

    if wch < 0 or wch >= input_waveform.shape[1]:
        raise ValueError(f"wave_ch out of range: {wch}, available channels={input_waveform.shape[1]}")

    n = input_waveform.shape[0]
    t = np.arange(n) * dt

    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=False)

    axes[0].plot(t, input_waveform[:, wch], label=f"Input waveform ch{wch}")
    axes[0].set_title(f"Original waveform | wave_ch={wch}")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tf_logits[s, r, :, 0], label="TF logits ch0")
    axes[1].plot(int8_logits_f[s, r, :, 0], label="INT8 logits ch0", alpha=0.8)
    axes[1].set_title(f"Logits channel 0 | sample={s}, row={r}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tf_logits[s, r, :, 1], label="TF logits ch1")
    axes[2].plot(int8_logits_f[s, r, :, 1], label="INT8 logits ch1", alpha=0.8)
    axes[2].set_title("Logits channel 1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(tf_prob[s, r, :, 0], label="TF prob ch0")
    axes[3].plot(int8_prob[s, r, :, 0], label="INT8 prob ch0", alpha=0.8)
    axes[3].set_title("Softmax prob channel 0")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(np.abs(int8_prob[s, r, :, 0] - tf_prob[s, r, :, 0]), label="|prob diff ch0|")
    axes[4].set_title("Absolute probability difference")
    axes[4].set_xlabel("Feature width index")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    fig.suptitle("Original waveform + Original TF vs INT8 TFLite", fontsize=16)
    fig.tight_layout()

    save_path = Path(args.save)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] saved: {save_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
