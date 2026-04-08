import argparse
import sys
from pathlib import Path

import numpy as np

# 让仓库根目录 / deepdenoiser 可导入
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from run_tflite_on_npz_fixed import (
    load_npz_waveform,
    prepare_waveform,
    waveform_to_model_input,
)


def iter_input_files(input_dir: Path):
    for p in sorted(input_dir.glob("*.npz")):
        yield p


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration feature files for DeepDenoiser INT8 TFLite conversion"
    )
    parser.add_argument(
        "--input_dir",
        default=r".\Dataset\train",
        help=r"Directory of raw waveform .npz files, default: .\Dataset\train",
    )
    parser.add_argument(
        "--output_dir",
        default=r".\calib_features",
        help=r"Directory to save calibration feature files, default: .\calib_features",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=50,
        help="Maximum number of raw waveform files to process",
    )
    parser.add_argument(
        "--target_nt",
        type=int,
        default=3000,
        help="Waveform length before STFT",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override sample interval in seconds; default: infer from file or 0.01",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {input_dir.resolve()}")

    files = files[: args.max_files]

    written = 0
    skipped = 0
    failed = 0

    print(f"[INFO] Input dir : {input_dir.resolve()}")
    print(f"[INFO] Output dir: {output_dir.resolve()}")
    print(f"[INFO] Raw files : {len(files)}")

    for idx, npz_path in enumerate(files, start=1):
        try:
            data, meta = load_npz_waveform(npz_path)
            dt = args.dt if args.dt is not None else float(meta.get("dt", 0.01))

            batch_waveform = prepare_waveform(data, dt=dt, target_nt=args.target_nt)
            X_input, noisy_signal, nbt, nt, nch = waveform_to_model_input(batch_waveform)

            # X_input shape: [batch*chn, 31, 201, 2]
            stem = npz_path.stem
            for ch in range(X_input.shape[0]):
                out_path = output_dir / f"{stem}_ch{ch}.npz"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                np.savez(
                    out_path,
                    X=X_input[ch].astype(np.float32),  # rank-3; converter will expand to batch-1
                    source_file=str(npz_path),
                    channel=np.int32(ch),
                    dt=np.float32(dt),
                )
                written += 1

            print(
                f"[OK] {idx}/{len(files)} {npz_path.name} -> {X_input.shape[0]} feature samples "
                f"(shape per sample: {tuple(X_input[0].shape)})"
            )

        except Exception as exc:
            failed += 1
            print(f"[FAIL] {npz_path.name}: {type(exc).__name__}: {exc}")

    print("\n[SUMMARY]")
    print(f"  written : {written}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")
    print(f"  out dir : {output_dir.resolve()}")

    sample_files = sorted(output_dir.glob("*.npz"))[:3]
    for p in sample_files:
        with np.load(p, allow_pickle=True) as z:
            shape = np.asarray(z["X"]).shape if "X" in z else None
            print(f"[SAMPLE] {p.name} | keys={list(z.keys())} | X.shape={shape}")


if __name__ == "__main__":
    main()
