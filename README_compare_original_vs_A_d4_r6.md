# 原始模型 float vs A_d4_r6 float 对比说明

这个脚本用于比较两组 **已经导出的 denoised waveform npz 文件**：

- 原始模型 float 输出目录
- A_d4_r6 float 输出目录

它会自动按 **文件名 stem** 对齐同名样本，计算：

- cand_vs_ref_wave_mae
- cand_vs_ref_wave_rmse
- cand_vs_ref_cosine
- cand_vs_ref_peak_norm_rmse

如果 npz 里恰好带有 clean/reference 键（如 `clean`, `signal`, `target`），脚本还会额外计算：

- ref_vs_clean_rmse
- cand_vs_clean_rmse
- ref_snr_improvement_db
- cand_snr_improvement_db

## 1. 目录准备

建议准备两个目录：

- `experiments/small_model_ptq/eval_outputs/original_float/`
- `experiments/small_model_ptq/eval_outputs/A_d4_r6_float/`

并保证两边的 npz 文件名一一对应，例如：

- `BK_BKS_2008110908041793.npz`
- `NC_GDXB_2012032506012264.npz`

## 2. 运行命令

```powershell
.\deepdenoiser_tflite_env\Scripts\python.exe .\batch_compare_waveforms.py `
  --ref_dir .\experiments\small_model_ptq\eval_outputs\original_float `
  --cand_dir .\experiments\small_model_ptq\eval_outputs\A_d4_r6_float `
  --ref_name original_float `
  --cand_name A_d4_r6_float `
  --out_csv .\experiments\small_model_ptq\eval_outputs\orig_vs_A_d4_r6_float_summary.csv `
  --out_md .\experiments\small_model_ptq\eval_outputs\orig_vs_A_d4_r6_float_summary.md
```

## 3. 输出文件

会生成：

- `orig_vs_A_d4_r6_float_summary.csv`
- `orig_vs_A_d4_r6_float_summary.md`

## 4. 怎么解读

如果：

- `cand_vs_ref_wave_rmse` 平均值较小
- `cand_vs_ref_cosine` 接近 1

说明：

A_d4_r6 的最终去噪波形行为总体接近原始模型。

如果你还额外导出了 clean/reference，并且脚本成功识别：

- `cand_vs_clean_rmse` 接近 `ref_vs_clean_rmse`
- `cand_snr_improvement_db` 接近 `ref_snr_improvement_db`

说明：

A_d4_r6 不仅接近原始模型，而且它对真实 clean 的任务效果也接近原始模型。
