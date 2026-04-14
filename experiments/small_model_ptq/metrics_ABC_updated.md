# Small-Model PTQ Metrics Summary (A / B / C)

## 1. Project Goal

目标：为 STM32F407ZGT6 / X-CUBE-AI 寻找一个**更适合全整数量化**的小模型候选，并在
- 结构规模
- fixed-sample PTQ 稳定性
- 30 样本 waveform 与 original_float 的接近程度

这三方面之间做平衡。

当前已完整跑通三个候选：
- A_d4_r6
- B_d4_r4
- C_d5_r6_cap48

---

## 2. Candidate Structures

### A_d4_r6
- depth: `4`
- filters_root: `6`
- filters_cap: `None`
- kernel_size: `[3, 3]`
- pool_size: `[2, 2]`
- dilation_rate: `[1, 1]`
- drop_rate: `0`

### B_d4_r4
- depth: `4`
- filters_root: `4`
- filters_cap: `None`
- kernel_size: `[3, 3]`
- pool_size: `[2, 2]`
- dilation_rate: `[1, 1]`
- drop_rate: `0`

### C_d5_r6_cap48
- depth: `5`
- filters_root: `6`
- filters_cap: `48`
- kernel_size: `[3, 3]`
- pool_size: `[2, 2]`
- dilation_rate: `[1, 1]`
- drop_rate: `0`

---

## 3. Training Result

### A_d4_r6
- checkpoint_dir: `G:/dd_runs/A_d4_r6/260410-195448`
- training_status: `completed through epoch 39`
- final_mean_loss: `0.489309`

### B_d4_r4
- checkpoint_dir: `G:/dd_runs/B_d4_r4/260413-160200`
- training_status: `completed through epoch 39`
- final_mean_loss: `0.527945`

### C_d5_r6_cap48
- checkpoint_dir: `G:/dd_runs/C_d5_r6_cap48/260413-193618`
- training_status: `completed through epoch 39`
- final_checkpoint_confirmed: `yes`

---

## 4. Fixed Evaluation Sample

- fixed_npz: `Dataset/pred/BK_BKS_2008110908041793.npz`

---

## 5. Float Builtin-Only Result

### A_d4_r6
- float_tflite: `experiments/small_model_ptq/A_d4_r6/model_float_logits_builtin.tflite`
- float_size: `196.52 KB`
- tf_vs_float_logits_mae: `1.837740057908377e-07`
- tf_vs_float_logits_rmse: `2.91895105419826e-07`
- tf_vs_float_logits_max_abs: `4.291534423828125e-06`
- tf_vs_float_logits_argmax_acc: `1.0`

### B_d4_r4
- float_tflite: `experiments/small_model_ptq/B_d4_r4/model_float_logits_builtin.tflite`
- float_size: `92.01 KB`
- tf_vs_float_logits_mae: `1.7290102505285176e-07`
- tf_vs_float_logits_rmse: `2.403537848749693e-07`
- tf_vs_float_logits_max_abs: `2.6226043701171875e-06`
- tf_vs_float_logits_argmax_acc: `1.0`

### C_d5_r6_cap48
- float_tflite: `experiments/small_model_ptq/C_d5_r6_cap48/model_float_logits_builtin.tflite`
- float_size: `604.04 KB`
- tf_vs_float_logits_mae: `1.5401070641019032e-07`
- tf_vs_float_logits_rmse: `2.0986678350709553e-07`
- tf_vs_float_logits_max_abs: `2.1457672119140625e-06`
- tf_vs_float_logits_argmax_acc: `1.0`

### Float Judgement
- A / B / C 三者 float 导出链路都正确
- 当前差异主要不是 float 导出错误，而是模型结构本身与量化后的行为差异

---

## 6. INT8 Builtin-Only Result

### A_d4_r6
- int8_tflite: `experiments/small_model_ptq/A_d4_r6/model_int8_logits_builtin.tflite`
- int8_size: `62.34 KB`
- tf_vs_int8_logits_mae: `0.04474635049700737`
- tf_vs_int8_logits_rmse: `0.057864174246788025`
- tf_vs_int8_logits_max_abs: `0.3337172269821167`
- tf_vs_int8_logits_argmax_acc: `0.9959878029208795`
- tf_vs_int8_softmax_mae: `0.012948709540069103`
- tf_vs_int8_softmax_rmse: `0.016230395063757896`
- tf_vs_int8_softmax_argmax_acc: `0.9959878029208795`

### B_d4_r4
- int8_tflite: `experiments/small_model_ptq/B_d4_r4/model_int8_logits_builtin.tflite`
- int8_size: `34.27 KB`
- tf_vs_int8_logits_mae: `0.050608`
- tf_vs_int8_logits_rmse: `0.065788`
- tf_vs_int8_logits_max_abs: `0.410680`
- tf_vs_int8_logits_argmax_acc: `0.965977`
- tf_vs_int8_softmax_mae: `0.017515`
- tf_vs_int8_softmax_rmse: `0.022968`
- tf_vs_int8_softmax_argmax_acc: `0.965977`

### C_d5_r6_cap48
- int8_tflite: `experiments/small_model_ptq/C_d5_r6_cap48/model_int8_logits_builtin.tflite`
- int8_size: `170.95 KB`
- tf_vs_int8_logits_mae: `0.04280310869216919`
- tf_vs_int8_logits_rmse: `0.05455152690410614`
- tf_vs_int8_logits_max_abs: `0.35820522904396057`
- tf_vs_int8_logits_argmax_acc: `0.990210239126946`
- tf_vs_int8_softmax_mae: `0.010631509125232697`
- tf_vs_int8_softmax_rmse: `0.014731088653206825`
- tf_vs_int8_softmax_argmax_acc: `0.990210239126946`

### INT8 Judgement
- A: 当前最稳、最均衡
- B: 最小，但 PTQ 稳定性明显退一步
- C: PTQ 也很稳，但体积显著增大

---

## 7. 30-Sample Waveform Compare vs original_float

评估集：
- `experiments/small_model_ptq/eval_set_30.txt`

参考输出：
- `experiments/small_model_ptq/eval_outputs/original_float`

### A_d4_r6_float vs original_float
- compared_samples: `30`
- mean_wave_mae: `171.15541896`
- mean_wave_rmse: `221.74718778`
- mean_cosine: `0.68485672`
- mean_peak_norm_rmse: `0.14250056`

### B_d4_r4_float vs original_float
- compared_samples: `30`
- mean_wave_mae: `170.85904363`
- mean_wave_rmse: `233.94565414`
- mean_cosine: `0.64983624`
- mean_peak_norm_rmse: `0.13599613`

### C_d5_r6_cap48_float vs original_float
- compared_samples: `30`
- mean_wave_mae: `171.63170916`
- mean_wave_rmse: `231.83237039`
- mean_cosine: `0.64899144`
- mean_peak_norm_rmse: `0.13816712`

### Waveform Judgement
- A 在 waveform 接近程度上最好
- B 与 C 接近，但都比 A 差一档
- 三者都不是 original_float 的近似复制品
- A 更适合作为当前主线 small-model baseline

---

## 8. Overall Ranking

### By waveform similarity to original_float
1. `A_d4_r6`
2. `C_d5_r6_cap48` ≈ `B_d4_r4`

### By PTQ stability on fixed sample
1. `A_d4_r6` ≈ `C_d5_r6_cap48`
2. `B_d4_r4`

### By model size
1. `B_d4_r4`
2. `A_d4_r6`
3. `C_d5_r6_cap48`

### By deployment-oriented overall balance
1. `A_d4_r6`
2. `B_d4_r4`
3. `C_d5_r6_cap48`

---

## 9. Final Decision

- current_mainline_candidate: `A_d4_r6`
- extreme_compression_backup: `B_d4_r4`
- stop_deep_dive_for_now: `C_d5_r6_cap48`

### Why A wins
1. waveform 与 original_float 最接近
2. INT8 PTQ 很稳
3. 体积明显小于 C
4. 综合平衡最好

---

## 10. Next Action

下一阶段不再继续扩结构搜索，转为：

### Mainline
- `teacher = original large float model`
- `student = A_d4_r6`

### Target
- 对 `A_d4_r6` 做蒸馏训练
- 再重复：
  1. float logits 导出
  2. int8 PTQ 导出
  3. fixed-sample TF vs int8 logits compare
  4. 30-sample waveform compare

### Goal
- 尽量让 A_d4_r6 更接近 original_float
- 同时保持当前良好的 PTQ 友好性
