# A_d4_r6 Notes

## Experiment Purpose

A_d4_r6 是“小模型 + PTQ”主线中的第一个候选模型。

目标：
1. 缩小模型结构
2. 保持 float 侧正常
3. 检查 PTQ 是否明显优于当前大模型 PTQ
4. 为后续 B_d4_r4 / C_d5_r6_cap48 提供对照基线

---

## Current Status

A_d4_r6 已成功完成一轮正式训练，并已完成以下关键步骤：

1. TF raw debug 导出成功
2. float logits builtin-only TFLite 导出成功
3. TF vs float logits 对比成功
4. int8 logits builtin-only TFLite 导出成功
5. TF vs int8 logits 对比成功
6. original_float vs A_d4_r6_float 的 30 样本 waveform 对比完成

当前阶段：
- 训练主流程已跑通
- checkpoint 已确认存在
- 原始 TF 推理链已成功恢复
- float logits builtin-only 模型已验证对齐
- int8 logits builtin-only 模型已验证可用
- A_d4_r6 已可作为当前“小模型 PTQ 基线”
- 但 A_d4_r6 还不能视为原始模型的近似复制品

---

## What Happened

### Attempt 1
使用带 valid.csv 的命令启动训练，失败。

原因：
- 本地不存在 `./Dataset/valid.csv`

处理：
- 取消验证集参数
- 先按无验证集方式训练

---

### Attempt 2
无验证集版本开始构图，但失败。

原因：
- `tf.keras.regularizers.l2(l=...)` 与当前环境不兼容

处理：
- 将 `tf.keras.regularizers.l2(l=...)`
  改为 `tf.keras.regularizers.l2(...)`

---

### Attempt 3
再次启动训练，仍在构图阶段失败。

原因：
- `tf.keras.regularizers.l2(...)` 收到的是 Tensor
- 当前环境要求 Python float，而不是 Tensor

处理：
- 在 `deepdenoiser/model.py` 中增加：
  `self.weight_decay_value = float(config.weight_decay)`
- regularizer 改为：
  `tf.keras.regularizers.l2(0.5 * self.weight_decay_value)`

---

### Attempt 4
再次启动训练，进入模型构图，但失败。

原因：
- `tf.compat.v1.layers.conv2d` 在当前默认 Keras 3 环境下不可用

处理：
- 在 `deepdenoiser/train.py` 中将
  `TF_USE_LEGACY_KERAS=1`
  放到 `import tensorflow as tf` 之前
- 使用当前环境中的 `tf_keras`

---

### Attempt 5
再次启动训练，模型构图通过，但失败在 summary writer 初始化。

原因：
- `tf.compat.v1.summary.FileWriter(...)` 报：
  `is not a directory`

处理：
- 对 `train.py` 中 `log_dir` 做绝对路径 + 规范化路径处理

---

### Attempt 6
再次启动训练，仍然在 FileWriter 阶段失败。

原因：
- 即使路径规范化后，之前的日志路径仍报：
  `is not a directory`

判断：
- 更像是 TensorFlow 在 Windows 下对之前日志路径的兼容性问题
- 非 ASCII 路径也可能参与了触发

处理：
- 将成功运行所用的 `--log_dir` 改为纯英文绝对路径：
  `G:\dd_runs\A_d4_r6`

---

### Attempt 7
正式成功运行。

关键信息：
- 训练日志目录：`G:/dd_runs/A_d4_r6/260410-195448`
- 训练从 `epoch=0` 跑到 `epoch=39`
- final mean loss 约为 `0.489309`

结论：
- A_d4_r6 的训练主循环已经跑通
- checkpoint 可进入导出与比较阶段

---

### Attempt 8
第一次导出 TF raw debug 失败。

原因：
- 导出脚本构出来的图与训练 checkpoint 的变量名不一致
- restore 阶段出现 BN 变量名不匹配

处理：
- 修改 `export_tf_raw_debug_fixed.py`
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 保证只构建一个 UNet
- 增加 restore 失败时的变量名调试输出

重试结果：
- TF raw debug 导出成功
- 输出文件：`experiments/small_model_ptq/A_d4_r6/tf_raw_debug.npz`
- TF preds shape: `(3, 31, 201, 2)`

---

### Attempt 9
第一次导出 float logits builtin-only 失败。

原因：
- `convert_deepdenoiser_tflite.py` 构出来的图与训练 checkpoint 的变量名不一致

处理：
- 修改 `convert_deepdenoiser_tflite.py`
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 保证只构建一个 UNet
- 增加 restore 失败时的变量名调试输出

重试结果：
- float logits builtin-only 导出成功
- 输出文件：`experiments/small_model_ptq/A_d4_r6/model_float_logits_builtin.tflite`
- 文件大小：`196.52 KB`
- 输入 shape：`[1, 31, 201, 2]`
- 输出 shape：`[1, 31, 201, 2]`

---

### Attempt 10
第一次做 TF vs float logits 对比失败。

原因：
- `compare_tf_vs_float_logits.py` 构出来的图与训练 checkpoint 的变量名不一致

处理：
- 修改 `compare_tf_vs_float_logits.py`
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 保证只构建一个 UNet
- 增加 restore 失败时的变量名调试输出

重试结果：
- 对比成功
- logits mae = `1.837740057908377e-07`
- logits rmse = `2.91895105419826e-07`
- logits max_abs = `4.291534423828125e-06`
- logits argmax_acc = `1.0`
- softmax(logits) argmax_acc = `1.0`

结论：
- A_d4_r6 的 float builtin-only logits 与原始 TF logits 可视为严格对齐

---

### Attempt 11
第一次导出 int8 logits builtin-only 失败。

原因：
- `YOUR_CALIB_DIR` 只是占位符，不是真实代表性数据集目录

处理：
- 使用真实代表性特征目录：
  `G:\桌面/边缘部署/DeepDenoiser\calib_features_full`
- 固定 `max_calib_samples = 297`

重试结果：
- int8 logits builtin-only 导出成功
- 输出文件：`experiments/small_model_ptq/A_d4_r6/model_int8_logits_builtin.tflite`
- 文件大小：`62.34 KB`
- 输入 dtype：`int8`
- 输出 dtype：`int8`

---

### Attempt 12
第一次做 TF vs int8 logits 对比失败。

原因：
- `compare_tf_vs_int8_logits.py` 构出来的图与训练 checkpoint 的变量名不一致

处理：
- 修改 `compare_tf_vs_int8_logits.py`
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 保证只构建一个 UNet
- 增加 restore 失败时的变量名调试输出

重试结果：
- 对比成功
- logits mae = `0.04474635049700737`
- logits rmse = `0.057864174246788025`
- logits max_abs = `0.3337172269821167`
- logits argmax_acc = `0.9959878029208795`
- softmax(logits) mae = `0.012948709540069103`
- softmax(logits) rmse = `0.016230395063757896`
- softmax(logits) max_abs = `0.0805220901966095`
- softmax(logits) argmax_acc = `0.9959878029208795`

结论：
- A_d4_r6 的 int8 logits 在固定样本上保持了很高的一致性
- 这条线已经不是“量化崩掉”，而是“明显可用的 PTQ 小模型候选”

---

### Attempt 13
为了比较“原始模型 float vs A_d4_r6 float”的最终行为，新增原始模型导出脚本并进行批量导出。

处理：
- 新增 `export_tf_raw_debug_original.py`
- 保留 legacy keras、reset_default_graph、restore 调试逻辑
- 去掉 A_d4_r6 的显式硬编码配置，改为使用原始模型配置
- 输出格式与 A_d4_r6 导出脚本对齐，确保保存：
  - `input_waveform`
  - `denoised_waveform`
  - `model_input`
  - `preds`
  - `dt`

结果：
- 原始模型 checkpoint `./model/190614-104802` smoke test 已通过
- 可用于批量导出 `original_float` 的评估输出

---

### Attempt 14
进行 original_float vs A_d4_r6_float 的 30 样本批量 waveform 对比。

评估集：
- `experiments/small_model_ptq/eval_set_30.txt`

输出目录：
- `experiments/small_model_ptq/eval_outputs/original_float`
- `experiments/small_model_ptq/eval_outputs/A_d4_r6_float`

汇总结果：
- compared samples = `30`
- mean cand_vs_ref_wave_mae = `171.15541896`
- mean cand_vs_ref_wave_rmse = `221.74718778`
- mean cand_vs_ref_cosine = `0.68485672`
- mean cand_vs_ref_peak_norm_rmse = `0.14250056`

最差 5 个样本：
- `BK_PACP_2008010512113190`
- `BK_WENL_2015010711035475`
- `BK_ORV_2010121817463027`
- `BK_SCZ_2015110300244148`
- `BK_SCZ_2014050200233687`

结论：
- A_d4_r6 和 original_float 的最终输出存在明显差异
- 这不是“几乎复现原始模型”的水平
- A_d4_r6 更适合被视为“量化友好的小模型基线”，而不是原始模型的直接替代品

---

## Confirmed Facts So Far

- 数据集读取成功
- 训练集大小为 `100`
- 当前无验证集，因此 validation = `0`
- 成功训练日志目录为：
  `G:/dd_runs/A_d4_r6/260410-195448`
- 训练已完整跑完 40 个 epoch（0 到 39）
- `checkpoint` 文件与 `model_39.ckpt.*` 已确认存在
- `tf_raw_debug.npz` 已成功生成
- `model_float_logits_builtin.tflite` 已成功生成
- `compare_tf_vs_float_logits.npz` 已成功生成
- `model_int8_logits_builtin.tflite` 已成功生成
- `compare_tf_vs_int8_logits.npz` 已成功生成
- `orig_vs_A_d4_r6_float_summary.csv` 已成功生成
- `orig_vs_A_d4_r6_float_summary.md` 已成功生成

---

## Current Judgement

到目前为止，可以做出的判断是：

1. A_d4_r6 这个小模型结构是可训练的
2. 原始 TF 推理链已可恢复并导出 debug
3. float logits builtin-only 模型已成功生成
4. TF logits vs float logits 已验证为高精度对齐
5. int8 logits builtin-only 模型已成功生成
6. TF logits vs int8 logits 在固定样本上保持了很高一致性
7. 30 样本 waveform 对比表明，A_d4_r6 与 original_float 存在明显能力差异
8. 当前还没有完成 clean/reference 绝对任务评估
9. 当前还没有完成与 STM32 / X-CUBE-AI 的整链路部署验证

所以现在的阶段判断是：

- A_d4_r6 已通过 float 验证
- A_d4_r6 已通过固定样本上的 int8 logits 验证
- A_d4_r6 是当前最好的量化友好 small-model PTQ baseline
- 但 A_d4_r6 还不能直接当作原始模型的最终替代品
- 下一阶段重点是：
  与 B_d4_r4 / C_d5_r6_cap48 横向对比，以及后续 clean/reference 与部署侧验证

---

## Compatibility / Environment Fix Summary

本轮为了让旧工程在当前环境下训练和导出成功，已经完成这些兼容性修复：

### model.py
- `tf.keras.regularizers.l2(l=...)`
  -> `tf.keras.regularizers.l2(...)`
- 新增：
  `self.weight_decay_value = float(config.weight_decay)`
- regularizer 改为：
  `tf.keras.regularizers.l2(0.5 * self.weight_decay_value)`

### train.py
- 在 `import tensorflow as tf` 之前设置：
  `TF_USE_LEGACY_KERAS=1`
- 将 `log_dir` 改为绝对路径 + 规范化路径
- 最终成功运行使用 ASCII-only 绝对日志目录：
  `G:\dd_runs\A_d4_r6`

### export_tf_raw_debug_fixed.py
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 单次构图
- 增加 restore 失败时的变量名调试输出
- 批量评估路径下保持 `denoised_waveform` 导出一致

### export_tf_raw_debug_original.py
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 使用原始模型配置
- 输出格式与 fixed 版对齐
- 可用于 original_float 的批量 waveform 评估

### convert_deepdenoiser_tflite.py
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 单次构图
- 增加 restore 失败时的变量名调试输出

### compare_tf_vs_float_logits.py
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 单次构图
- 增加 restore 失败时的变量名调试输出

### compare_tf_vs_int8_logits.py
- 增加 legacy keras
- 增加 `tf.compat.v1.reset_default_graph()`
- 显式使用 A_d4_r6 配置
- 单次构图
- 增加 restore 失败时的变量名调试输出

---

## About The Shutdown Errors

训练完成后出现的：

- `Enqueue operation was cancelled`
- 多个 queue/thread traceback
- interpreter shutdown 相关错误

当前判断：
- 更像是 TF1 queue runner 在线程收尾阶段退出不干净
- 这些信息出现在 epoch 训练完成之后
- 当前先将其视为“训练完成后的退出噪声”
- 当前没有证据表明它破坏了这轮 checkpoint

---

## Next Step

1. 把 A_d4_r6 固定为当前 quant-friendly small-model baseline
2. 用同一套流程继续跑 B_d4_r4
3. 用同一套流程继续跑 C_d5_r6_cap48
4. 统一比较：
   - original_float vs candidate_float 的 waveform 相似度
   - float 模型大小
   - int8 模型大小
   - logits mae / rmse / max_abs
   - logits argmax_acc
   - softmax(logits) argmax_acc
5. 再决定谁最适合继续做 clean/reference 评估和部署验证