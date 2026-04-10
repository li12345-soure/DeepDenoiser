import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot original TF logits/prob vs float builtin-only TFLite logits/prob"
    )
    parser.add_argument("--npz", required=True, help="Path to compare_tf_vs_float_logits*.npz")
    parser.add_argument("--save", default="compare_tf_vs_float_logits.png", help="Output image path")
    parser.add_argument("--sample", type=int, default=0, help="Feature sample index in [0, batch*chn)")
    parser.add_argument("--row", type=int, default=15, help="Row index in [0, 31)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path.resolve())

    with np.load(npz_path, allow_pickle=True) as z:
        print("[INFO] keys:", z.files)
        tf_logits = np.asarray(z["tf_logits"], dtype=np.float32)
        float_logits = np.asarray(z["float_logits"], dtype=np.float32)
        tf_prob = np.asarray(z["tf_prob"], dtype=np.float32)
        float_prob = np.asarray(z["float_prob"], dtype=np.float32)

    s = args.sample
    r = args.row

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(tf_logits[s, r, :, 0], label="TF logits ch0")
    axes[0].plot(float_logits[s, r, :, 0], label="Float TFLite logits ch0", alpha=0.8)
    axes[0].set_title(f"Logits channel 0 | sample={s}, row={r}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tf_logits[s, r, :, 1], label="TF logits ch1")
    axes[1].plot(float_logits[s, r, :, 1], label="Float TFLite logits ch1", alpha=0.8)
    axes[1].set_title("Logits channel 1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tf_prob[s, r, :, 0], label="TF prob ch0")
    axes[2].plot(float_prob[s, r, :, 0], label="Float TFLite prob ch0", alpha=0.8)
    axes[2].set_title("Softmax prob channel 0")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(np.abs(float_prob[s, r, :, 0] - tf_prob[s, r, :, 0]), label="|prob diff ch0|")
    axes[3].set_title("Absolute probability difference")
    axes[3].set_xlabel("Feature width index")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(args.save)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] saved: {save_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
