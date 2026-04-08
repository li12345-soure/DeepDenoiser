# DeepDenoiser -> TFLite / INT8 / TFLM 指南

这份指南针对 AI4EPS/DeepDenoiser 仓库的当前结构：
- 预训练 checkpoint 默认在 `model/190614-104802`
- `model.py` 是 TF1 风格静态图，使用 `tf.compat.v1.disable_eager_execution()`、`tf.compat.v1.layers.*`、`tf.compat.v1.lite.TFLiteConverter` 更合适
- 网络输入张量默认是 `X_shape = [31, 201, 2]`

## 一、先改一处源码（强烈建议）

原始 `model.py` 在推理图里把 `is_training` 和 `drop_rate` 也做成了 placeholder。
这会导致导出的 TFLite 模型通常有 3 个输入：
- `X`
- `is_training`
- `drop_rate`

这在桌面 TFLite 还能勉强接受，但对 TFLM 很不友好。

建议把 `add_placeholders()` 改成下面这样：

```python
    def add_placeholders(self, input_batch=None, mode='train'):
        if input_batch is None:
            self.X = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, self.X_shape[-1]], name='X'
            )
            self.Y = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=[None, None, None, self.n_class], name='y'
            )
        else:
            self.X = input_batch[0]
            if mode in ["train", "valid", "test"]:
                self.Y = input_batch[1]
            self.input_batch = input_batch

        if mode == "pred":
            self.is_training = tf.constant(False, dtype=tf.bool, name="is_training")
            self.drop_rate = tf.constant(0.0, dtype=tf.float32, name="drop_rate")
        else:
            self.is_training = tf.compat.v1.placeholder(dtype=tf.bool, name="is_training")
            self.drop_rate = tf.compat.v1.placeholder(dtype=tf.float32, name="drop_rate")
```

这样导出后的推理图通常就只剩一个真正输入 `X`，更适合量化和 TFLM。

## 二、准备环境

建议不要在原始老环境里做量化，推荐使用较新的 TF2 环境，但通过 `tf.compat.v1` 兼容旧图。

例如：

```bash
python -m venv .venv
source .venv/bin/activate
pip install "tensorflow>=2.13,<2.17" numpy scipy
```

如果你还需要跑仓库原脚本，再补：

```bash
pip install matplotlib pandas tqdm obspy fastapi uvicorn kafka-python
```

## 三、代表性数据（INT8 校准）要准备什么

不要喂原始波形。

这个网络真正量化时要喂的是 **网络输入张量**，也就是形状接近 `[31, 201, 2]` 的时频特征，
而不是 `Dataset/train/*.npz` 里的原始 `9001 x 3` 波形。

推荐做法：
1. 先沿用 DeepDenoiser 原来的前处理，把原始波形变成网络输入 `X`
2. 把这些 `X` 保存成很多个 `.npy` 或 `.npz`
3. 把这些特征文件放进一个目录，例如 `calib_features/`

脚本默认支持：
- `*.npy`：直接保存单个 `float32` 特征，形状 `[31, 201, 2]` 或 `[1, 31, 201, 2]`
- `*.npz`：优先读取键名 `X`，否则取第一个数组

## 四、怎么跑

### 1. 先转浮点 TFLite

```bash
python convert_deepdenoiser_tflite.py \
  --repo_root /path/to/DeepDenoiser \
  --checkpoint_dir /path/to/DeepDenoiser/model/190614-104802 \
  --output_tflite deepdenoiser_float.tflite \
  --output_nodes preds
```

### 2. 做动态范围量化

```bash
python convert_deepdenoiser_tflite.py \
  --repo_root /path/to/DeepDenoiser \
  --checkpoint_dir /path/to/DeepDenoiser/model/190614-104802 \
  --output_tflite deepdenoiser_dynamic.tflite \
  --quant_mode dynamic \
  --output_nodes preds
```

### 3. 做全整数量化（推荐给 TFLM 前验证）

前提：
- 你已经按上面那样把 `mode='pred'` 的辅助 placeholder 固定成常量
- 你准备好了网络输入特征目录 `calib_features/`

```bash
python convert_deepdenoiser_tflite.py \
  --repo_root /path/to/DeepDenoiser \
  --checkpoint_dir /path/to/DeepDenoiser/model/190614-104802 \
  --output_tflite deepdenoiser_int8.tflite \
  --quant_mode int8 \
  --calib_dir /path/to/calib_features \
  --output_nodes logits \
  --int8_io
```

说明：
- `--output_nodes preds`：输出 softmax 概率
- `--output_nodes logits`：输出未 softmax 的 logits
- 做 TFLM 时，我更建议先试 `logits`，因为 softmax 虽然一般也支持，但验证主干网络时先把输出保持简单一点更好

## 五、怎么判断算子是否适合 TFLM

脚本会额外生成一个分析文本文件：
- `xxx.tflite.analysis.txt`

你重点看：
- 是否出现 `SELECT_TF_OPS`
- 是否出现 `CUSTOM`
- 最终 builtin ops 里是否主要是这些：
  - `CONV_2D`
  - `TRANSPOSE_CONV`
  - `SLICE`
  - `CONCATENATION`
  - `SOFTMAX`

如果出现 `SELECT_TF_OPS` 或 `CUSTOM`，就不适合直接上 TFLM。

## 六、TFLM 下一步

得到 `deepdenoiser_int8.tflite` 后：

1. 先用 TFLite Python/桌面侧验证数值
2. 再用 tflite-micro 的 resolver 生成工具检查实际用到的算子
3. 再把 `.tflite` 转成 C 数组：

```bash
xxd -i deepdenoiser_int8.tflite > model_data.cc
```

## 七、最现实的风险

DeepDenoiser 是 U-Net，包含：
- 多层卷积
- 转置卷积
- skip connection
- concat / slice

所以即使算子都支持，TFLM 真正落板时最可能先卡在 **tensor_arena 内存**，不是先卡在“能不能转模型”。
