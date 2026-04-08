import argparse
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
import tensorflow as tf


FS = 100
NPERSEG = 30
NFFT = 60


def normalize_batch(data, window=200):
    """
    data: [nbt, nf, nt, 2]
    Reimplements DeepDenoiser deepdenoiser/app.py normalize_batch().
    """
    assert len(data.shape) == 4
    shift = window // 2
    nbt, nf, nt, nimg = data.shape

    data_pad = np.pad(
        data,
        ((0, 0), (0, 0), (window // 2, window // 2), (0, 0)),
        mode="reflect",
    )
    t = np.arange(0, nt + shift - 1, shift, dtype="int")
    std = np.zeros([nbt, len(t)], dtype=np.float32)
    mean = np.zeros([nbt, len(t)], dtype=np.float32)

    for i in range(std.shape[1]):
        sl = data_pad[:, :, i * shift : i * shift + window, :]
        std[:, i] = np.std(sl, axis=(1, 2, 3))
        mean[:, i] = np.mean(sl, axis=(1, 2, 3))

    if std.shape[1] >= 2:
        std[:, -1], mean[:, -1] = std[:, -2], mean[:, -2]
        std[:, 0], mean[:, 0] = std[:, 1], mean[:, 1]

    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, kind="slinear")(t_interp)
    std_interp[std_interp == 0] = 1.0
    mean_interp = interp1d(t, mean, kind="slinear")(t_interp)

    data = (data - mean_interp[:, np.newaxis, :, np.newaxis]) / std_interp[:, np.newaxis, :, np.newaxis]
    if len(t) > 3:
        data /= 2.0
    return data.astype(np.float32)


def load_npz_waveform(npz_path: Path):
    """
    Tries common keys used in seismic npz files.
    Returns waveform as [nt, chn] float32.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
        arr = None
        for key in ["data", "vec", "waveform", "x"]:
            if key in z:
                arr = z[key]
                break
        if arr is None:
            # Fallback: first ndarray with ndim >= 1
            for key in keys:
                if isinstance(z[key], np.ndarray) and z[key].ndim >= 1:
                    arr = z[key]
                    break
        if arr is None:
            raise ValueError(f"Could not find waveform array in {npz_path}. Keys={keys}")

        arr = np.asarray(arr, dtype=np.float32)

        # NEW: support 1D waveform [nt] as single-channel input.
        if arr.ndim == 1:
            data = arr[:, np.newaxis]

        # Prefer [nt, chn]
        elif arr.ndim == 2:
            if arr.shape[1] in (1, 2, 3):
                data = arr
            elif arr.shape[0] in (1, 2, 3):
                data = arr.T
            else:
                # Heuristic: longer dimension is time axis
                data = arr if arr.shape[0] >= arr.shape[1] else arr.T

        elif arr.ndim == 3:
            # Try squeeze batch dim if present
            if arr.shape[0] == 1:
                arr = arr[0]
                if arr.ndim == 1:
                    data = arr[:, np.newaxis]
                elif arr.shape[1] in (1, 2, 3):
                    data = arr
                elif arr.shape[0] in (1, 2, 3):
                    data = arr.T
                else:
                    data = arr if arr.shape[0] >= arr.shape[1] else arr.T
            else:
                raise ValueError(f"Unsupported 3D waveform shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported waveform ndim={arr.ndim}, shape={arr.shape}")

        # Keep up to 3 channels
        if data.shape[1] > 3:
            data = data[:, :3]

        meta = {
            "keys": keys,
        }
        if "dt" in z:
            try:
                meta["dt"] = float(np.asarray(z["dt"]).reshape(-1)[0])
            except Exception:
                pass
        if "sampling_rate" in z and "dt" not in meta:
            try:
                sr = float(np.asarray(z["sampling_rate"]).reshape(-1)[0])
                if sr > 0:
                    meta["dt"] = 1.0 / sr
            except Exception:
                pass

    return data.astype(np.float32), meta


def prepare_waveform(data, dt=0.01, target_nt=3000):
    """
    Match app.py behavior as closely as practical:
    - resample to 100 Hz if dt != 0.01
    - pad/trim to target_nt
    - ensure 3 channels where possible
    Returns [1, nt, chn]
    """
    if data.ndim != 2:
        raise ValueError(f"Expected [nt, chn], got shape={data.shape}")

    nt, chn = data.shape
    if chn == 1:
        data = np.repeat(data, 3, axis=1)
    elif chn == 2:
        data = np.concatenate([data, data[:, -1:]], axis=1)
    elif chn > 3:
        data = data[:, :3]

    if abs(dt - 0.01) > 1e-9:
        t = np.linspace(0, 1, data.shape[0], dtype=np.float32)
        new_len = int(np.round(data.shape[0] * dt * FS))
        new_t = np.linspace(0, 1, new_len, dtype=np.float32)
        data = interp1d(t, data, axis=0, kind="slinear")(new_t).astype(np.float32)

    if data.shape[0] < target_nt:
        pad = target_nt - data.shape[0]
        data = np.pad(data, ((0, pad), (0, 0)), mode="constant")
    elif data.shape[0] > target_nt:
        data = data[:target_nt]

    return data[np.newaxis, ...].astype(np.float32)


def waveform_to_model_input(batch_waveform):
    """
    batch_waveform: [batch, nt, chn]
    Returns:
      X_input: [batch*chn, 31, 201, 2]
      noisy_signal: complex split before normalization
      batch, nt, chn
    """
    nbt, nt, nch = batch_waveform.shape
    vec = np.transpose(batch_waveform, [0, 2, 1])  # [batch, chn, nt]
    vec = np.reshape(vec, [nbt * nch, nt])         # [batch*chn, nt]

    _, _, tmp_signal = scipy.signal.stft(
        vec, fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary="zeros"
    )
    noisy_signal = np.stack([tmp_signal.real, tmp_signal.imag], axis=-1).astype(np.float32)
    noisy_signal[np.isnan(noisy_signal)] = 0.0
    noisy_signal[np.isinf(noisy_signal)] = 0.0

    X_input = normalize_batch(noisy_signal)
    return X_input, noisy_signal, nbt, nt, nch


def run_tflite(model_path, X_input):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise RuntimeError(f"Expected 1 input, got {len(input_details)}")

    interpreter.resize_tensor_input(input_details[0]["index"], list(X_input.shape), strict=False)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = X_input.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])
    return y, input_details, output_details


def mask_to_waveform(preds, noisy_signal, nbt, nt, nch):
    """
    preds: [batch*chn, nf, nt_frames, 2]
    """
    complex_spec = noisy_signal[..., 0] + 1j * noisy_signal[..., 1]
    masked = complex_spec * preds[..., 0]
    _, denoised_signal = scipy.signal.istft(
        masked, fs=FS, nperseg=NPERSEG, nfft=NFFT, boundary="zeros"
    )
    denoised_signal = np.reshape(denoised_signal, [nbt, nch, nt])
    denoised_signal = np.transpose(denoised_signal, [0, 2, 1])  # [batch, nt, chn]
    return denoised_signal.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run DeepDenoiser TFLite model on Dataset/pred or your own .npz")
    parser.add_argument("--model", default="deepdenoiser_float.tflite", help="Path to TFLite model")
    parser.add_argument("--npz", required=True, help="Path to one .npz waveform file")
    parser.add_argument("--dt", type=float, default=None, help="Override sample interval in seconds (default: infer or 0.01)")
    parser.add_argument("--target_nt", type=int, default=3000, help="Target waveform length before STFT")
    parser.add_argument("--save_npz", default=None, help="Optional output .npz path")
    args = parser.parse_args()

    model_path = Path(args.model)
    npz_path = Path(args.npz)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path.resolve()}")

    data, meta = load_npz_waveform(npz_path)
    dt = args.dt if args.dt is not None else float(meta.get("dt", 0.01))

    print(f"[INFO] Loaded NPZ: {npz_path}")
    print(f"[INFO] Available keys: {meta.get('keys', [])}")
    print(f"[INFO] Raw waveform shape: {data.shape} [nt, chn]")
    print(f"[INFO] Using dt={dt}")

    batch_waveform = prepare_waveform(data, dt=dt, target_nt=args.target_nt)
    print(f"[INFO] Prepared waveform shape: {batch_waveform.shape} [batch, nt, chn]")

    X_input, noisy_signal, nbt, nt, nch = waveform_to_model_input(batch_waveform)
    print(f"[INFO] Model input feature shape: {X_input.shape} [batch*chn, 31, 201, 2]")

    preds, in_details, out_details = run_tflite(model_path, X_input)
    print(f"[INFO] TFLite output shape: {preds.shape}")

    denoised = mask_to_waveform(preds, noisy_signal, nbt, nt, nch)
    print(f"[OK] Denoised waveform shape: {denoised.shape} [batch, nt, chn]")

    if args.save_npz:
        out_path = Path(args.save_npz)
        np.savez(
            out_path,
            input_waveform=batch_waveform[0],
            denoised_waveform=denoised[0],
            model_input=X_input,
            preds=preds,
            dt=dt,
        )
        print(f"[OK] Saved results to: {out_path.resolve()}")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
