$PY = ".\deepdenoiser_tflite_env\Scripts\python.exe"

& $PY .\plot_three_way_compare.py `
  --teacher_npz .\result_teacher_float_eval_same_sample.npz `
  --base_npz .\result_A_d4_r6_base_float_tf_same_sample.npz `
  --distill_npz .\result_A_d4_r6_distill_v1_float_tf_same_sample.npz `
  --save_png .\three_way_logits_compare.png

& $PY .\plot_three_way_waveform_compare.py `
  --teacher_npz .\result_teacher_float_eval_same_sample.npz `
  --base_npz .\result_A_d4_r6_base_float_tf_same_sample.npz `
  --distill_npz .\result_A_d4_r6_distill_v1_float_tf_same_sample.npz `
  --save_png .\three_way_waveform_compare.png
