import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


WAVE_KEYS = [
    "denoised_waveform",
    "pred_waveform",
    "waveform",
]
REF_KEYS = [
    "clean",
    "clean_waveform",
    "target",
    "signal",
    "reference",
    "y_true",
]


def load_npz_array(npz_path: Path, candidate_keys: List[str]) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    for key in candidate_keys:
        if key in keys:
            return np.asarray(data[key], dtype=np.float32)
    raise KeyError(f"{npz_path} does not contain any of keys: {candidate_keys}")



def load_waveform_from_npz(npz_path: Path) -> np.ndarray:
    return load_npz_array(npz_path, WAVE_KEYS)



def load_reference_from_npz(npz_path: Path) -> np.ndarray:
    return load_npz_array(npz_path, REF_KEYS)



def sample_id_from_path(path: Path) -> str:
    return path.stem



def flatten_waveform(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)



def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))



def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))



def peak_normalized_rmse(a: np.ndarray, b: np.ndarray) -> float:
    peak = float(max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12))
    return float(np.sqrt(np.mean((a - b) ** 2)) / peak)



def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    noise = clean - test
    p_signal = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 1e-12:
        return float("inf")
    if p_signal <= 1e-12:
        return float("nan")
    return float(10.0 * np.log10(p_signal / p_noise))



def snr_improvement(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    snr_noisy = snr_db(clean, noisy)
    snr_denoised = snr_db(clean, denoised)
    if math.isnan(snr_noisy) or math.isnan(snr_denoised):
        return float("nan")
    return float(snr_denoised - snr_noisy)



def collect_npz_map(folder: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for p in sorted(folder.glob("*.npz")):
        result[sample_id_from_path(p)] = p
    return result



def try_load_noisy_input(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    for key in ["input_waveform", "noisy_waveform", "waveform_in", "x"]:
        if key in keys:
            return np.asarray(data[key], dtype=np.float32)
    raise KeyError



def write_csv(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        raise ValueError("No rows to write")
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def mean_of(rows: List[dict], key: str) -> float:
    vals = []
    for row in rows:
        try:
            v = float(row[key])
        except Exception:
            continue
        if not math.isnan(v) and not math.isinf(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(np.mean(vals))



def write_markdown(rows: List[dict], out_md: Path, ref_name: str, cand_name: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    mean_wave_mae = mean_of(rows, "cand_vs_ref_wave_mae")
    mean_wave_rmse = mean_of(rows, "cand_vs_ref_wave_rmse")
    mean_cos = mean_of(rows, "cand_vs_ref_cosine")
    mean_pnrmse = mean_of(rows, "cand_vs_ref_peak_norm_rmse")
    mean_ref_clean_rmse = mean_of(rows, "ref_vs_clean_rmse")
    mean_cand_clean_rmse = mean_of(rows, "cand_vs_clean_rmse")
    mean_ref_snr_imp = mean_of(rows, "ref_snr_improvement_db")
    mean_cand_snr_imp = mean_of(rows, "cand_snr_improvement_db")

    worst5 = sorted(rows, key=lambda x: float(x["cand_vs_ref_wave_rmse"]), reverse=True)[:5]

    lines: List[str] = []
    lines.append(f"# {cand_name} vs {ref_name} waveform comparison")
    lines.append("")
    lines.append(f"- compared samples: `{len(rows)}`")
    lines.append(f"- mean cand_vs_ref_wave_mae: `{mean_wave_mae:.8f}`")
    lines.append(f"- mean cand_vs_ref_wave_rmse: `{mean_wave_rmse:.8f}`")
    lines.append(f"- mean cand_vs_ref_cosine: `{mean_cos:.8f}`")
    lines.append(f"- mean cand_vs_ref_peak_norm_rmse: `{mean_pnrmse:.8f}`")
    lines.append("")
    if not math.isnan(mean_ref_clean_rmse) or not math.isnan(mean_cand_clean_rmse):
        lines.append("## Against clean/reference (if available)")
        lines.append("")
        lines.append(f"- mean ref_vs_clean_rmse: `{mean_ref_clean_rmse:.8f}`")
        lines.append(f"- mean cand_vs_clean_rmse: `{mean_cand_clean_rmse:.8f}`")
        lines.append(f"- mean ref_snr_improvement_db: `{mean_ref_snr_imp:.8f}`")
        lines.append(f"- mean cand_snr_improvement_db: `{mean_cand_snr_imp:.8f}`")
        lines.append("")
    lines.append("## Worst 5 samples by cand_vs_ref_wave_rmse")
    lines.append("")
    lines.append("| sample_id | cand_vs_ref_wave_mae | cand_vs_ref_wave_rmse | cand_vs_ref_cosine | cand_vs_ref_peak_norm_rmse |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in worst5:
        lines.append(
            f"| {row['sample_id']} | {float(row['cand_vs_ref_wave_mae']):.8f} | {float(row['cand_vs_ref_wave_rmse']):.8f} | "
            f"{float(row['cand_vs_ref_cosine']):.8f} | {float(row['cand_vs_ref_peak_norm_rmse']):.8f} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare denoised waveform npz outputs from original model float vs A_d4_r6 float"
    )
    parser.add_argument("--ref_dir", required=True, help="Folder of reference model output npz files")
    parser.add_argument("--cand_dir", required=True, help="Folder of candidate model output npz files")
    parser.add_argument("--ref_name", default="original_float")
    parser.add_argument("--cand_name", default="A_d4_r6_float")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    cand_dir = Path(args.cand_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    ref_map = collect_npz_map(ref_dir)
    cand_map = collect_npz_map(cand_dir)
    common_ids = sorted(set(ref_map.keys()) & set(cand_map.keys()))
    if not common_ids:
        raise RuntimeError("No common sample ids found between ref_dir and cand_dir")

    rows: List[dict] = []
    for sid in common_ids:
        ref_npz = ref_map[sid]
        cand_npz = cand_map[sid]
        ref_wave = flatten_waveform(load_waveform_from_npz(ref_npz))
        cand_wave = flatten_waveform(load_waveform_from_npz(cand_npz))
        if ref_wave.shape != cand_wave.shape:
            raise ValueError(f"Shape mismatch for {sid}: {ref_wave.shape} vs {cand_wave.shape}")

        row = {
            "sample_id": sid,
            "ref_npz": str(ref_npz),
            "cand_npz": str(cand_npz),
            "cand_vs_ref_wave_mae": mae(cand_wave, ref_wave),
            "cand_vs_ref_wave_rmse": rmse(cand_wave, ref_wave),
            "cand_vs_ref_cosine": cosine_similarity(cand_wave, ref_wave),
            "cand_vs_ref_peak_norm_rmse": peak_normalized_rmse(cand_wave, ref_wave),
            "ref_vs_clean_rmse": float("nan"),
            "cand_vs_clean_rmse": float("nan"),
            "ref_snr_improvement_db": float("nan"),
            "cand_snr_improvement_db": float("nan"),
        }

        try:
            clean_ref = flatten_waveform(load_reference_from_npz(ref_npz))
            if clean_ref.shape == ref_wave.shape:
                row["ref_vs_clean_rmse"] = rmse(ref_wave, clean_ref)
                row["cand_vs_clean_rmse"] = rmse(cand_wave, clean_ref)
                try:
                    noisy = flatten_waveform(try_load_noisy_input(ref_npz))
                    if noisy.shape == clean_ref.shape:
                        row["ref_snr_improvement_db"] = snr_improvement(clean_ref, noisy, ref_wave)
                        row["cand_snr_improvement_db"] = snr_improvement(clean_ref, noisy, cand_wave)
                except KeyError:
                    pass
        except KeyError:
            pass

        rows.append(row)

    write_csv(rows, out_csv)
    write_markdown(rows, out_md, args.ref_name, args.cand_name)
    print(f"[OK] wrote csv: {out_csv}")
    print(f"[OK] wrote md : {out_md}")
    print(f"[INFO] compared {len(rows)} common samples")


if __name__ == "__main__":
    main()
