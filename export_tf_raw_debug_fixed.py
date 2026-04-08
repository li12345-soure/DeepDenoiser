import argparse
import os
import sys
from pathlib import Path

# 必须在 import tensorflow / 导入旧版 tf.layers 图之前设置
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

# 让仓库根目录 / deepdenoiser 可导入
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from model import UNet
from run_tflite_on_npz_fixed import (
    load_npz_waveform,
    prepare_waveform,
    waveform_to_model_input,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export original TF checkpoint raw model_input/preds for one NPZ"
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="e.g. ./model/190614-104802",
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="e.g. ./Dataset/pred/BK_BKS_2008110908041793.npz",
    )
    parser.add_argument(
        "--save_npz",
        default="orig_raw_debug.npz",
        help="Output npz path",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override dt if needed",
    )
    parser.add_argument(
        "--target_nt",
        type=int,
        default=3000,
        help="Waveform length before STFT",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    npz_path = Path(args.npz)
    out_path = Path(args.save_npz)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir.resolve()}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Input npz not found: {npz_path.resolve()}")

    data, meta = load_npz_waveform(npz_path)
    dt = args.dt if args.dt is not None else float(meta.get("dt", 0.01))

    print(f"[INFO] Loaded NPZ: {npz_path}")
    print(f"[INFO] Available keys: {meta.get('keys', [])}")
    print(f"[INFO] Raw waveform shape: {data.shape} [nt, chn]")
    print(f"[INFO] Using dt={dt}")

    batch_waveform = prepare_waveform(data, dt=dt, target_nt=args.target_nt)
    print(f"[INFO] Prepared waveform shape: {batch_waveform.shape} [batch, nt, chn]")

    X_input, noisy_signal, nbt, nt, nch = waveform_to_model_input(batch_waveform)
    print(f"[INFO] Model input feature shape: {X_input.shape} [batch*chn, 31, 201, 2]")

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    model = UNet(mode="pred")

    latest_ckpt = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir.resolve()}")
    print(f"[INFO] Restoring checkpoint: {latest_ckpt}")

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    preds_list = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, latest_ckpt)

        # pred 模式图固定 batch=1：逐个 feature 跑再拼回去
        for i in range(X_input.shape[0]):
            x_i = X_input[i:i + 1].astype(np.float32)
            y_i = sess.run(model.preds, feed_dict={model.X: x_i})
            preds_list.append(y_i)

    preds = np.concatenate(preds_list, axis=0).astype(np.float32)
    print(f"[INFO] TF preds shape: {preds.shape}")

    np.savez(
        out_path,
        input_waveform=batch_waveform[0],
        model_input=X_input.astype(np.float32),
        preds=preds,
        dt=np.float32(dt),
    )
    print(f"[OK] Saved debug npz to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
