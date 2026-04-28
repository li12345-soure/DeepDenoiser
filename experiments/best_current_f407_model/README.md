Best current F407 deployable model

model:
  B_d4_r4_distill_original_v2_T8_kd2_nomse_step2500

student:
  B_d4_r4
  depth=4
  filters_root=4
  drop_rate=0.0

teacher:
  original float DeepDenoiser
  depth=6
  filters_root=8

training:
  T=8.0
  kd_weight=2.0
  logit_mse_weight=0.0

evaluation:
  int8 TFLite vs original pred20 argmax_acc ~= 0.790748

decision:
  This is the current highest-accuracy deployable int8 model.
  d4_r5_decoder075_skip050 has better RAM margin but lower verified int8 accuracy.
