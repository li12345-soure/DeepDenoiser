#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build offline teacher-logits cache for DeepDenoiser distillation.

This script is intentionally conservative:
1. It first tries to read an already-prepared feature tensor from each .npz file.
2. If the input .npz does not contain the configured feature key, it can call a
   user-provided adapter function to build X from the raw sample.
3. It runs the teacher float TF checkpoint once and saves teacher_logits per sample.

Typical use:
  python make_teacher_logits_cache.py \
      --checkpoint_dir ./model/original_teacher \
      --input_list ./Dataset/train_feature_list.txt \
      --out_dir ./TeacherCache/train \
      --feature_key X

If your list points to raw waveform .npz files instead of feature .npz files, add:
  --adapter_module local_distill_adapter \
  --adapter_fn build_feature_from_npz

The adapter function must have signature:
    X = build_feature_from_npz(npz_path: str) -> np.ndarray
and must return float32 with shape [31, 201, 2] or [1, 31, 201, 2].
"""

from __future__ import print_function

import argparse
import importlib
import io
import os
import sys
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

# Keep behavior aligned with the repo's TF1-style training/inference code.
tf.compat.v1.disable_eager_execution()


def _import_teacher_builders():
    """Import UNet plus the canonical train-time config builder."""
    errors = []
    for model_module_name, train_module_name in (
        ('deepdenoiser.model', 'deepdenoiser.train'),
        ('model', 'train'),
    ):
        try:
            model_module = importlib.import_module(model_module_name)
            train_module = importlib.import_module(train_module_name)
            UNet = getattr(model_module, 'UNet')
            set_config = getattr(train_module, 'set_config')
            build_arg_parser = getattr(train_module, 'build_arg_parser')
            return UNet, set_config, build_arg_parser
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append('%s + %s: %s' % (model_module_name, train_module_name, exc))
    raise ImportError('Failed to import UNet/train helpers. Details: %s' % ' | '.join(errors))


class _StaticShapeReader(object):
    """Minimal reader stub so cache generation reuses train.py:set_config()."""

    def __init__(self, x_shape, y_shape, n_signal=1):
        self.X_shape = list(x_shape)
        self.Y_shape = list(y_shape)
        self.n_signal = int(n_signal)


def _read_list(path: str) -> List[str]:
    items = []
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            items.append(line)
    return items


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and (not os.path.exists(parent)):
        os.makedirs(parent)


def _normalize_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        x = x[None, ...]
    if x.ndim != 4:
        raise ValueError('Expected X ndim in {3,4}, got %s with shape %s' % (x.ndim, x.shape))
    if tuple(x.shape[-3:]) != (31, 201, 2):
        raise ValueError('Expected X shape [*,31,201,2], got %s' % (x.shape,))
    return x


def _load_adapter(module_name: Optional[str], fn_name: str) -> Optional[Callable[[str], np.ndarray]]:
    if not module_name:
        return None
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def _load_input_feature(npz_path: str, feature_key: str, adapter: Optional[Callable[[str], np.ndarray]]) -> np.ndarray:
    meta = np.load(npz_path, allow_pickle=True)
    if feature_key in meta.files:
        return _normalize_x(meta[feature_key])
    if adapter is None:
        raise KeyError(
            'Input file %s does not contain key %r. Provide --adapter_module/--adapter_fn '
            'to build X from raw sample.' % (npz_path, feature_key)
        )
    return _normalize_x(adapter(npz_path))


def _build_teacher_config(
    build_arg_parser,
    set_config,
    x_shape=(31, 201, 2),
    depth=6,
    filters_root=8,
    filters_cap=None,
    kernel_size=None,
    pool_size=None,
    dilation_rate=None,
    class_weights=None,
    weight_decay=0.0,
    drop_rate=0.0,
):
    teacher_args = build_arg_parser().parse_args([])
    teacher_args.mode = 'pred'
    teacher_args.batch_size = 1
    teacher_args.depth = int(depth)
    teacher_args.filters_root = int(filters_root)
    teacher_args.filters_cap = filters_cap
    if kernel_size is not None:
        teacher_args.kernel_size = list(kernel_size)
    if pool_size is not None:
        teacher_args.pool_size = list(pool_size)
    if dilation_rate is not None:
        teacher_args.dilation_rate = list(dilation_rate)
    if class_weights is not None:
        teacher_args.class_weights = list(class_weights)
    teacher_args.weight_decay = float(weight_decay)
    teacher_args.drop_rate = float(drop_rate)
    teacher_args.summary = False
    teacher_args.distill_enable = 0

    reader = _StaticShapeReader(
        x_shape=list(x_shape),
        y_shape=[x_shape[0], x_shape[1], 2],
        n_signal=1,
    )
    config = set_config(teacher_args, reader)

    # Extra aliases / compatibility flags for auditability and forward compatibility.
    config.filter_size = list(config.kernel_size)
    config.batch_norm = True

    required_fields = [
        'depths',
        'filters_root',
        'filters_cap',
        'kernel_size',
        'filter_size',
        'pool_size',
        'dilation_rate',
        'X_shape',
        'Y_shape',
        'n_channel',
        'n_class',
        'batch_size',
        'class_weights',
        'loss_type',
        'weight_decay',
        'optimizer',
        'decay_step',
        'decay_rate',
        'momentum',
        'learning_rate',
        'summary',
        'drop_rate',
        'batch_norm',
    ]
    missing = [name for name in required_fields if not hasattr(config, name)]
    if missing:
        raise AttributeError('Teacher config is missing required fields: %s' % ', '.join(missing))
    return config


def _build_teacher(
    checkpoint_dir: str,
    x_shape=(31, 201, 2),
    depth=6,
    filters_root=8,
    filters_cap=None,
    kernel_size=None,
    pool_size=None,
    dilation_rate=None,
    class_weights=None,
    weight_decay=0.0,
    drop_rate=0.0,
):
    UNet, set_config, build_arg_parser = _import_teacher_builders()
    config = _build_teacher_config(
        build_arg_parser=build_arg_parser,
        set_config=set_config,
        x_shape=x_shape,
        depth=depth,
        filters_root=filters_root,
        filters_cap=filters_cap,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dilation_rate=dilation_rate,
        class_weights=class_weights,
        weight_decay=weight_decay,
        drop_rate=drop_rate,
    )
    model = UNet(config=config, mode='pred')
    if not hasattr(model, 'logits'):
        raise AttributeError('Teacher model does not expose model.logits')

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)
    sess.run(tf.compat.v1.global_variables_initializer())
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise ValueError('No checkpoint found under %s' % checkpoint_dir)
    print('[INFO] restoring teacher checkpoint:', latest)
    saver.restore(sess, latest)
    return sess, model


def _infer_logits(sess, model, x: np.ndarray) -> np.ndarray:
    feed = {model.X: x}
    if hasattr(model, 'is_training') and model.is_training.op.type.startswith('Placeholder'):
        feed[model.is_training] = False
    if hasattr(model, 'drop_rate') and model.drop_rate.op.type.startswith('Placeholder'):
        feed[model.drop_rate] = 0.0
    logits = sess.run(model.logits, feed_dict=feed)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if tuple(logits.shape) != (31, 201, 2):
        raise ValueError('Teacher logits shape mismatch: %s' % (logits.shape,))
    return logits


def _resolve_input_path(item: str, input_root: Optional[str]) -> str:
    if os.path.isabs(item):
        return item
    if input_root:
        return os.path.normpath(os.path.join(input_root, item))
    return os.path.normpath(item)


def _resolve_output_path(item: str, input_path: str, out_dir: str, input_root: Optional[str]) -> str:
    # Preserve relative structure when possible.
    if input_root:
        rel = os.path.relpath(input_path, os.path.normpath(input_root))
    else:
        rel = item if not os.path.isabs(item) else os.path.basename(item)
    rel = os.path.splitext(rel)[0] + '.npz'
    return os.path.normpath(os.path.join(out_dir, rel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--input_list', required=True, help='Text file. One sample path per line.')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--input_root', default=None, help='Optional root for relative paths in input_list.')
    parser.add_argument('--feature_key', default='X', help='If present in input npz, use this as model input.')
    parser.add_argument('--adapter_module', default=None, help='Optional module that builds X from raw sample npz.')
    parser.add_argument('--adapter_fn', default='build_feature_from_npz')
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--filters_root', type=int, default=8)
    parser.add_argument('--filters_cap', type=int, default=None)
    parser.add_argument('--kernel_size', nargs='+', type=int, default=None)
    parser.add_argument('--pool_size', nargs='+', type=int, default=None)
    parser.add_argument('--dilation_rate', nargs='+', type=int, default=None)
    parser.add_argument('--class_weights', nargs='+', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    adapter = _load_adapter(args.adapter_module, args.adapter_fn)
    items = _read_list(args.input_list)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError('No items found in %s' % args.input_list)

    sess, model = _build_teacher(
        checkpoint_dir=args.checkpoint_dir,
        depth=args.depth,
        filters_root=args.filters_root,
        filters_cap=args.filters_cap,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        dilation_rate=args.dilation_rate,
        class_weights=args.class_weights,
        weight_decay=args.weight_decay,
        drop_rate=args.drop_rate,
    )

    ok = 0
    skip = 0
    fail = 0
    for idx, item in enumerate(items, 1):
        input_path = _resolve_input_path(item, args.input_root)
        output_path = _resolve_output_path(item, input_path, args.out_dir, args.input_root)
        if (not args.overwrite) and os.path.exists(output_path):
            print('[SKIP %d/%d] exists: %s' % (idx, len(items), output_path))
            skip += 1
            continue
        try:
            x = _load_input_feature(input_path, args.feature_key, adapter)
            teacher_logits = _infer_logits(sess, model, x)
            _ensure_parent(output_path)
            np.savez_compressed(output_path, teacher_logits=teacher_logits.astype(np.float32))
            print('[OK   %d/%d] %s -> %s' % (idx, len(items), input_path, output_path))
            ok += 1
        except Exception as exc:
            print('[FAIL %d/%d] %s :: %s' % (idx, len(items), input_path, exc), file=sys.stderr)
            fail += 1

    print('\n[SUMMARY] ok=%d skip=%d fail=%d out_dir=%s' % (ok, skip, fail, args.out_dir))
    sess.close()
    if fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
