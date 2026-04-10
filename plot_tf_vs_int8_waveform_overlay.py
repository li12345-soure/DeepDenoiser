import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

FS = 100
NPERSEG = 30
NFFT = 60


def ensure_2d_waveform(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x


def waveform_to_noisy_signal(batch_waveform):
    # batch_waveform: [batch, nt, chn]
    nbt, nt, nch = batch_waveform.shape
    vec = np.transpose(batch_waveform, [0, 2, 1])   # [batch, chn, nt]
    vec = np.reshape(vec, [nbt * nch, nt])          # [batch*chn, nt]

    _, _, tmp_signal = scipy.signal.stft(
        vec, fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary="zeros"
    )
    noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1).astype(np.float32)
    return noisy_signal, nbt, nt, nch


def mask_to_waveform(prob_or_mask, noisy_signal, nbt, nt, nch):
    # prob_or_mask: [batch*chn, nf, nt_frames, 2] or [batch*chn, nf, nt_frames]
    if prob_or_mask.ndim == 4:
        mask = prob_or_mask[..., 0]
    else:
        mask = prob_or_mask

    complex_spec = noisy_signal[..., 0] + 1j * noisy_signal[..., 1]
    masked = complex_spec * mask
    _, denoised_signal = scipy.signal.istft(
        masked, fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary="zeros"
    )
    denoised_signal = np.reshape(denoised_signal, [nbt, nch, nt])
    denoised_signal = np.transpose(denoised_signal, [0, 2, 1])  # [batch, nt, chn]
    return denoised_signal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Plot input waveform + TF output + INT8 output from compare_tf_vs_int8_logits*.npz"
    )
    parser.add_argument("--npz", required=True, help="Path to compare_tf_vs_int8_logits*.npz")
    parser.add_argument("--save", default="compare_tf_vs_int8_waveform_overlay.png", help="Output image path")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path.resolve())

    with np.load(npz_path, allow_pickle=True) as z:
        print("[INFO] keys:", z.files)
        input_waveform = ensure_2d_waveform(z["input_waveform"])
        tf_prob = np.asarray(z["tf_prob"], dtype=np.float32)
        int8_prob = np.asarray(z["int8_prob"], dtype=np.float32)
        dt = float(np.asarray(z["dt"]).reshape(-1)[0]) if "dt" in z else 0.01

    batch_waveform = input_waveform[np.newaxis, ...]   # [1, nt, chn]
    noisy_signal, nbt, nt, nch = waveform_to_noisy_signal(batch_waveform)

    tf_wave = mask_to_waveform(tf_prob, noisy_signal, nbt, nt, nch)[0]
    int8_wave = mask_to_waveform(int8_prob, noisy_signal, nbt, nt, nch)[0]

    n = min(input_waveform.shape[0], tf_wave.shape[0], int8_wave.shape[0])
    c = min(input_waveform.shape[1], tf_wave.shape[1], int8_wave.shape[1])
    input_waveform = input_waveform[:n, :c]
    tf_wave = tf_wave[:n, :c]
    int8_wave = int8_wave[:n, :c]
    t = np.arange(n) * dt

    fig, axes = plt.subplots(c, 1, figsize=(14, 3.2 * c), sharex=True)
    if c == 1:
        axes = [axes]

    for i in range(c):
        axes[i].plot(t, input_waveform[:, i], label="Input waveform", alpha=0.5)
        axes[i].plot(t, tf_wave[:, i], label="Original TF output", linewidth=1.2)
        axes[i].plot(t, int8_wave[:, i], label="INT8 output", linewidth=1.2)
        axes[i].set_ylabel(f"Ch {i}")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")

    fig.suptitle("Original TF vs INT8 TFLite")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    save_path = Path(args.save)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[OK] saved: {save_path.resolve()}")
    print("[INFO] Compare summary:")
    diff = int8_wave - tf_wave
    print("  tf shape   :", tf_wave.shape)
    print("  int8 shape :", int8_wave.shape)
    print("  mae        :", float(np.mean(np.abs(diff))))
    print("  rmse       :", float(np.sqrt(np.mean(diff ** 2))))

    plt.show()


if __name__ == "__main__":
    main()
