# A_d4_r6 Metrics

## 1. Candidate Info

- candidate: `A_d4_r6`
- depth: `4`
- filters_root: `6`
- kernel_size: `[3, 3]`
- pool_size: `[2, 2]`
- dilation_rate: `[1, 1]`
- drop_rate: `0`
- loss_type: `cross_entropy`

## 2. Training Info

- batch_size: `20`
- epochs: `40`
- optimizer: `adam`
- learning_rate: `3e-4`
- weight_decay: `1e-5`
- log_dir_arg_initial: `./runs/A_d4_r6`
- log_dir_arg_success: `G:\dd_runs\A_d4_r6`
- actual_log_dir_attempt_1: `not created successfully (valid.csv error before training start)`
- actual_log_dir_attempt_2: `./runs/A_d4_r6/260410-190318`
- actual_log_dir_attempt_3: `./runs/A_d4_r6/260410-191218`
- actual_log_dir_attempt_4: `./runs/A_d4_r6/260410-192843`
- actual_log_dir_attempt_5: `G:/桌面/边缘部署/DeepDenoiser/runs/A_d4_r6/260410-194932`
- actual_log_dir_success: `G:/dd_runs/A_d4_r6/260410-195448`
- dataset_size_training: `100`
- dataset_size_validation: `0`
- training_start_time: `2026-04-10 19:54:48`
- training_end_time: `2026-04-10 19:55:42`
- training_status: `training loop completed through epoch 39`
- final_epoch: `39`
- final_mean_loss: `0.489309`
- checkpoint_dir: `G:/dd_runs/A_d4_r6/260410-195448`
- best_checkpoint_dir: `G:/dd_runs/A_d4_r6/260410-195448`
- checkpoint_confirmed: `yes`
- latest_checkpoint: `model_39.ckpt`
- shutdown_noise: `yes`
- shutdown_noise_comment: `TF1 queue/thread shutdown produced CancelledError after training finished`

## 3. Fixed Evaluation Sample

- npz: `Dataset/pred/BK_BKS_2008110908041793.npz`

---

## 4. TF Raw Debug Export

- script: `export_tf_raw_debug_fixed.py`
- checkpoint_dir_used: `G:/dd_runs/A_d4_r6/260410-195448`
- output_npz: `./experiments/small_model_ptq/A_d4_r6/tf_raw_debug.npz`
- status: `success`

### TF Raw Debug Info
- loaded_npz: `Dataset/pred/BK_BKS_2008110908041793.npz`
- model_input_shape: `(3, 31, 201, 2)`
- tf_preds_shape: `(3, 31, 201, 2)`

---

## 5. Float Builtin-Only Export

- converter_script: `convert_deepdenoiser_tflite.py`
- checkpoint_dir_used: `G:/dd_runs/A_d4_r6/260410-195448`
- float_tflite_path: `./experiments/small_model_ptq/A_d4_r6/model_float_logits_builtin.tflite`
- file_size: `196.52 KB`
- export_status: `success`

### Float Model Info
- input_name: `X`
- input_shape: `[1, 31, 201, 2]`
- input_dtype: `float32`
- output_name: `Output/output_conv/BiasAdd`
- output_shape: `[1, 31, 201, 2]`
- output_dtype: `float32`

### TF vs Float Logits

- logits_mae: `1.837740057908377e-07`
- logits_rmse: `2.91895105419826e-07`
- logits_max_abs: `4.291534423828125e-06`
- logits_argmax_acc: `1.0`
- softmax_mae: `4.7700574157261144e-08`
- softmax_rmse: `7.343206220866705e-08`
- softmax_max_abs: `5.960464477539062e-07`
- softmax_argmax_acc: `1.0`
- compare_npz: `./experiments/small_model_ptq/A_d4_r6/compare_tf_vs_float_logits.npz`

### Float Judgement

- float_export_correct: `yes`
- comment: `A_d4_r6 float logits builtin-only is effectively identical to original TF logits on the fixed sample`

---

## 6. INT8 PTQ Export

- converter_script: `convert_deepdenoiser_tflite.py`
- checkpoint_dir_used: `G:/dd_runs/A_d4_r6/260410-195448`
- calibration_features_path: `G:/桌面/边缘部署/DeepDenoiser/calib_features_full`
- max_calib_samples: `297`
- int8_tflite_path: `./experiments/small_model_ptq/A_d4_r6/model_int8_logits_builtin.tflite`
- file_size: `62.34 KB`
- export_status: `success`

### INT8 Model Info
- input_name: `X`
- input_shape: `[1, 31, 201, 2]`
- input_dtype: `int8`
- output_name: `Output/output_conv/BiasAdd`
- output_shape: `[1, 31, 201, 2]`
- output_dtype: `int8`
- input_scale: `0.2968551814556122`
- input_zero_point: `15`
- output_scale: `0.08275671303272247`
- output_zero_point: `70`

### TF vs INT8 Logits

- logits_mae: `0.04474635049700737`
- logits_rmse: `0.057864174246788025`
- logits_max_abs: `0.3337172269821167`
- logits_argmax_acc: `0.9959878029208795`
- softmax_mae: `0.012948709540069103`
- softmax_rmse: `0.016230395063757896`
- softmax_max_abs: `0.0805220901966095`
- softmax_argmax_acc: `0.9959878029208795`
- compare_npz: `./experiments/small_model_ptq/A_d4_r6/compare_tf_vs_int8_logits.npz`

### INT8 Judgement

- int8_acceptable: `yes_on_fixed_sample`
- comment: `A_d4_r6 int8 logits remain close to TF logits on the fixed sample and preserve argmax almost perfectly`

---

## 7. Waveform / End-to-End Check

- waveform_compare_plot: `not run`
- waveform_comment: `not run`

---

## 8. Compare Against Current Large-Model PTQ Baseline

### Current baseline reference
- baseline_int8_logits_mae: `3.8618`
- baseline_int8_logits_rmse: `4.4226`
- baseline_int8_logits_max_abs: `8.6070`
- baseline_int8_logits_argmax_acc: `0.3449`
- baseline_int8_softmax_mae: `0.5557`
- baseline_int8_softmax_rmse: `0.6551`
- baseline_int8_softmax_argmax_acc: `0.3449`

### A_d4_r6 vs baseline
- smaller_model: `yes`
- better_int8_logits: `yes`
- better_softmax_argmax_acc: `yes`
- summary: `A_d4_r6 is far smaller and appears much more PTQ-friendly than the previous large-model baseline`

---

## 9. Current Failure / Fix Record

### Failure 1
- stage: `dataset init`
- reason: `./Dataset/valid.csv not found`
- fix: `remove validation CSV arguments and run training without validation set`

### Failure 2
- stage: `model build`
- reason: `tf.keras.regularizers.l2(l=...) incompatible with current tf.keras`
- fix: `changed to tf.keras.regularizers.l2(0.5 * weight_decay)`

### Failure 3
- stage: `model build`
- reason: `tf.keras.regularizers.l2(...) received Tensor, but current keras expects Python float`
- fix_status: `fixed`
- fix_detail: `added self.weight_decay_value = float(config.weight_decay), and regularizer now uses tf.keras.regularizers.l2(0.5 * self.weight_decay_value)`

### Failure 4
- stage: `model build`
- reason: ``tf.compat.v1.layers.conv2d` is not available with Keras 3`
- fix_status: `fixed`
- fix_detail: `set TF_USE_LEGACY_KERAS=1 before importing tensorflow and used installed tf_keras path`

### Failure 5
- stage: `summary writer init`
- reason: `FileWriter reported path is not a directory for previous log_dir paths`
- fix_status: `fixed`
- fix_detail: `normalized train.py log_dir path and switched successful run to ASCII-only absolute path G:\dd_runs\A_d4_r6`

### Failure 6
- stage: `process shutdown`
- reason: `TF1 queue runner threads raised CancelledError / interpreter shutdown messages after training loop finished`
- fix_status: `not fixed`
- impact: `non-blocking for this run unless checkpoint inspection shows corruption`

### Failure 7
- stage: `TF raw debug export`
- reason: `initial restore mismatch between export graph and checkpoint variable names`
- fix_status: `fixed`
- fix_detail: `updated export_tf_raw_debug_fixed.py with legacy keras, reset_default_graph, explicit A_d4_r6 config, single-graph build, and restore debug prints`

### Failure 8
- stage: `float builtin-only export`
- reason: `initial restore mismatch between convert graph and checkpoint variable names`
- fix_status: `fixed`
- fix_detail: `updated convert_deepdenoiser_tflite.py with legacy keras, reset_default_graph, explicit A_d4_r6 config, single-graph build, and restore debug prints`

### Failure 9
- stage: `TF vs float logits comparison`
- reason: `initial restore mismatch between compare graph and checkpoint variable names`
- fix_status: `fixed`
- fix_detail: `updated compare_tf_vs_float_logits.py with legacy keras, reset_default_graph, explicit A_d4_r6 config, single-graph build, and restore debug prints`

### Failure 10
- stage: `INT8 PTQ export`
- reason: `placeholder calibration path YOUR_CALIB_DIR caused file-not-found during representative dataset loading`
- fix_status: `fixed`
- fix_detail: `replaced placeholder with real feature directory G:/桌面/边缘部署/DeepDenoiser/calib_features_full`

### Failure 11
- stage: `TF vs INT8 logits comparison`
- reason: `initial restore mismatch between compare graph and checkpoint variable names`
- fix_status: `fixed`
- fix_detail: `updated compare_tf_vs_int8_logits.py with legacy keras, reset_default_graph, explicit A_d4_r6 config, single-graph build, and restore debug prints`

---

## 10. Final Decision

- decision: `current_best_small_model_ptq_candidate`
- reason_1: `A_d4_r6 training completed successfully through epoch 39`
- reason_2: `TF logits vs float logits are essentially identical`
- reason_3: `A_d4_r6 int8 logits remain close to TF logits on the fixed sample`
- reason_4: `A_d4_r6 int8 model size is only 62.34 KB`
- next_action: `use A_d4_r6 as the PTQ small-model baseline, then continue B_d4_r4 and C_d5_r6_cap48 for horizontal comparison`