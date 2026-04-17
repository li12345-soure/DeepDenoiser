# A_d4_r6_distill_v1 patch instructions for Codex

This repo already has the right high-level training shape for TF1-style DeepDenoiser:
- `deepdenoiser/train.py` currently builds batches with `batch = data_reader.dequeue(args.batch_size)` and trains with `model.train_on_batch(sess, X_batch, Y_batch, ...)`. It restores checkpoints from `--model_dir` via `tf.train.latest_checkpoint(args.model_dir)`. citeturn221762view0
- `deepdenoiser/model.py` already exposes `self.logits` and `self.preds`, and the default hard-label cross-entropy is computed from `self.logits` and `self.Y`. citeturn927104view0turn927104view1
- The original data reader queue for training enqueues exactly two tensors, `X` and `Y`, using a `PaddingFIFOQueue` with `['float32', 'float32']`. citeturn143348view0turn143348view1

The goal of this patch is **minimal distillation**:
- teacher = original large float checkpoint
- student = A_d4_r6
- student warm-start from existing A checkpoint
- offline teacher logits cache
- total loss = `alpha * hard_ce + beta * distill_soft_ce(T) + reg_loss`

## Files already added
- `deepdenoiser/distill_utils.py`
- `make_teacher_logits_cache.py`

## Required code edits

### 1) `deepdenoiser/train.py`
Add CLI args:
- `--distill_enable` (int, default 0)
- `--distill_teacher_cache_dir` (str, default None)
- `--distill_val_teacher_cache_dir` (str, default None)
- `--distill_alpha` (float, default 0.5)
- `--distill_beta` (float, default 0.5)
- `--distill_temperature` (float, default 2.0)
- `--student_init_ckpt` (str, default None)

Restore logic:
- If `args.student_init_ckpt` is set, restore that exact checkpoint before training.
- Otherwise keep the existing `args.model_dir` restore behavior.

Batch logic:
- If `args.distill_enable == 0`, keep the current behavior exactly.
- If `args.distill_enable == 1`, instantiate a new reader class that returns `(X, Y, teacher_logits)`.
- Update train/valid loops to unpack either two tensors or three tensors.

Training call:
- Replace `model.train_on_batch(...)` and `model.valid_on_batch(...)` with overloads that optionally accept `teacher_logits_batch`.

### 2) `deepdenoiser/data_reader.py`
Create a new reader class by copying the existing train reader that currently enqueues `(X, Y)`.
Suggested name:
- `DataReaderDistill`

Changes:
- add init arg `teacher_cache_dir`
- add a third placeholder: `teacher_logits_placeholder`
- change queue dtypes/shapes to three tensors:
  - `['float32', 'float32', 'float32']`
  - shapes `[self.config.X_shape, self.config.Y_shape, self.config.Y_shape]`
- in the training thread, after computing `noisy_signal` and `mask`, locate the matching teacher cache file and load key `teacher_logits`
- enqueue `[noisy_signal, mask, teacher_logits]`

Path mapping rule:
- keep it mirror-based
- if signal sample path is `Dataset/train/foo/bar.npz`
- teacher cache path should be `teacher_cache_dir/foo/bar.npz`

Validation reader:
- do the same if you use a separate valid reader

### 3) `deepdenoiser/model.py`
Keep the existing hard loss path unchanged.
Add optional distillation support inside the model class.

Suggested changes:
- in `__init__`, add defaults:
  - `self.distill_enable = getattr(config, 'distill_enable', False)`
  - `self.distill_alpha = getattr(config, 'distill_alpha', 0.5)`
  - `self.distill_beta = getattr(config, 'distill_beta', 0.5)`
  - `self.distill_temperature = getattr(config, 'distill_temperature', 2.0)`
- in `add_placeholders`, when `input_batch` exists and has length >= 3 in train/valid/test mode, set:
  - `self.teacher_logits = input_batch[2]`
- in `add_loss_op`, preserve existing `loss` as the hard loss
- if distillation is enabled and `self.teacher_logits` exists:
  - import `distill_soft_ce` and `combine_losses` from `deepdenoiser.distill_utils`
  - compute `self.hard_loss = loss`
  - compute `self.distill_loss = distill_soft_ce(self.logits, self.teacher_logits, temperature=self.distill_temperature)`
  - keep current `weight_loss` logic if any
  - set `self.loss = combine_losses(self.hard_loss, self.distill_loss, alpha=self.distill_alpha, beta=self.distill_beta, reg_loss=weight_loss_or_none)`
- add scalar summaries for:
  - `hard_loss`
  - `distill_loss`
  - `train_loss` / `valid_loss`

Batch APIs:
- update `train_on_batch` and `valid_on_batch` to optionally accept `teacher_logits_batch=None`
- only feed `self.teacher_logits` when distillation is enabled and a batch is provided

## Minimal run sequence

### Step A: build teacher cache
If you already have feature npz files containing `X`:
```powershell
$PY=".\deepdenoiser_tflite_env\Scripts\python.exe"
& $PY .\make_teacher_logits_cache.py `
  --checkpoint_dir G:\dd_runs\original_teacher `
  --input_list .\train_feature_list.txt `
  --out_dir G:\dd_cache\teacher_logits_train `
  --feature_key X
```

If your list points to raw waveform `.npz`, provide an adapter module that returns `[31,201,2]` X.

### Step B: train distilled A
```powershell
$PY=".\deepdenoiser_tflite_env\Scripts\python.exe"
& $PY .\deepdenoiser\train.py `
  --mode=train `
  --depth 4 `
  --filters_root 6 `
  --student_init_ckpt G:\dd_runs\A_d4_r6\model_best.ckpt `
  --distill_enable 1 `
  --distill_teacher_cache_dir G:\dd_cache\teacher_logits_train `
  --distill_alpha 0.5 `
  --distill_beta 0.5 `
  --distill_temperature 2.0 `
  --learning_rate 1e-4 `
  --log_dir G:\dd_runs\A_d4_r6_distill_v1
```

## Validation order
1. compare **teacher float logits** vs **distilled student float logits**
2. compare **distilled student TF logits** vs **distilled student float TFLite logits**
3. compare **distilled student TF logits** vs **distilled student int8 logits**
4. finally compare end-to-end waveform
