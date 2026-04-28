Best current PC-side F407 candidate

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

logits / mask evaluation:
  pred100 int8 vs original mask_argmax_acc = 0.757172
  old step2500 pred100 mask_argmax_acc = 0.755325
  improvement = +0.001847

waveform evaluation, pred100:
  refine step3000:
    mean waveform_rmse = 128.446374
    mean waveform_corrcoef = 0.949308
    mean waveform_relative_l2 = 0.306751
    mean mask_argmax_acc = 0.757172
    mean softmax_mae = 0.235489
    mean softmax_rmse = 0.296143

  old step2500:
    mean waveform_rmse = 153.094750
    mean waveform_corrcoef = 0.944004
    mean waveform_relative_l2 = 0.335318
    mean mask_argmax_acc = 0.755325
    mean softmax_mae = 0.247609
    mean softmax_rmse = 0.302679

decision:
  refine step3000 is the current PC-side best.
  It improves both logits/mask metrics and waveform reconstruction metrics over old step2500.
  STM32 board-ready app is still based on old step2500 and should remain frozen until board validation is available.
