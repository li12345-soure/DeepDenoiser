Best current B_d4_r4 refine candidate

model:
  B_d4_r4_best_refine_v1_T6_kd1_nomse_step3000

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
  warm-start: B_d4_r4_distill_original_v2_T8_kd2_nomse_step2500
  temperature=6.0
  kd_weight=1.0
  logit_mse_weight=0.0

evaluation:
  pred100 int8 TFLite vs original argmax_acc = 0.757172
  previous step2500 pred100 int8 vs original argmax_acc = 0.755325
  improvement ~= +0.001847

legacy pred20:
  step3000 ~= 0.7916627
  step5000 ~= 0.7917429
  previous step2500 ~= 0.7907479

decision:
  step3000 is recommended as the more stable new PC-side best by pred100.
  The improvement is small; B_d4_r4 is likely near its current capacity/training ceiling.
  PTQ is not the main bottleneck because int8 vs own TF pred100 argmax_acc = 0.992870.
