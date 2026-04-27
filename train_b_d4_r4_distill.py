#!/usr/bin/env python3
"""Online checkpoint distillation training for a configurable student.

This script intentionally does not modify deepdenoiser/train.py. It reuses the
repo's DataReader, set_config, UNet, TensorBoard writer, and checkpoint style,
but builds a frozen teacher graph next to the student graph.
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deepdenoiser"))

from deepdenoiser.data_reader import DataReader
from deepdenoiser.model import UNet
from deepdenoiser.train import build_arg_parser, set_config


DEFAULT_STUDENT_INIT_DIR = r"G:\dd_runs\B_d4_r4\260413-160200"
DEFAULT_TEACHER_CKPT_DIR = r"G:\dd_tmp\orig_model_190614"
DEFAULT_LOG_DIR = r"G:\dd_runs\B_d4_r4_distill_v1"


class QuietDataReader(DataReader):
    """DataReader wrapper that suppresses expected shutdown cancellations."""

    def thread_main(self, sess, n_threads=1, start=0):
        try:
            return super().thread_main(sess, n_threads=n_threads, start=start)
        except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
            if self.coord is not None and self.coord.should_stop():
                return
            raise
        except Exception as exc:
            if self.coord is not None and self.coord.should_stop():
                logging.warning("Reader thread stopped during shutdown: %s", exc)
                return
            raise


def build_script_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Train a student DeepDenoiser with online frozen-teacher distillation"
    parser.set_defaults(
        mode="train",
        epochs=40,
        batch_size=20,
        learning_rate=3e-4,
        optimizer="adam",
        weight_decay=1e-5,
        loss_type="cross_entropy",
        depth=4,
        filters_root=4,
        filters_cap=None,
        decoder_width_mult=1.0,
        skip_bottleneck_mult=1.0,
        use_skip_bottleneck=0,
        kernel_size=[3, 3],
        pool_size=[2, 2],
        dilation_rate=[1, 1],
        drop_rate=0.0,
        log_dir=DEFAULT_LOG_DIR,
        train_signal_dir=r".\Dataset\train",
        train_signal_list=r".\Dataset\train.csv",
        train_noise_dir=r".\Dataset\train",
        train_noise_list=r".\Dataset\train.csv",
    )
    parser.add_argument("--student_depth", type=int, default=None, help="Student depth; defaults to --depth/4")
    parser.add_argument(
        "--student_filters_root",
        type=int,
        default=None,
        help="Student filters_root; defaults to --filters_root/4",
    )
    parser.add_argument(
        "--student_filters_cap",
        type=int,
        default=None,
        help="Student filters_cap; defaults to --filters_cap",
    )
    parser.add_argument(
        "--student_decoder_width_mult",
        type=float,
        default=None,
        help="Student decoder width multiplier; defaults to --decoder_width_mult",
    )
    parser.add_argument(
        "--student_skip_bottleneck_mult",
        type=float,
        default=None,
        help="Student skip bottleneck multiplier; defaults to --skip_bottleneck_mult",
    )
    parser.add_argument(
        "--student_use_skip_bottleneck",
        type=int,
        default=None,
        help="Student skip bottleneck enable flag; defaults to --use_skip_bottleneck",
    )
    parser.add_argument(
        "--student_drop_rate",
        type=float,
        default=None,
        help="Student drop_rate; defaults to --drop_rate/0.0",
    )
    parser.add_argument(
        "--student_init_dir",
        default=None,
        help=(
            "Directory containing a student warm-start checkpoint. If omitted, "
            "the B_d4_r4 default checkpoint is used only for the default B_d4_r4 student."
        ),
    )
    parser.add_argument(
        "--teacher_checkpoint_dir",
        default=DEFAULT_TEACHER_CKPT_DIR,
        help="Directory containing the frozen teacher checkpoint",
    )
    parser.add_argument("--teacher_depth", type=int, default=6, help="Teacher depth")
    parser.add_argument("--teacher_filters_root", type=int, default=8, help="Teacher filters_root")
    parser.add_argument("--teacher_filters_cap", type=int, default=None, help="Teacher filters_cap")
    parser.add_argument(
        "--teacher_decoder_width_mult",
        type=float,
        default=1.0,
        help="Teacher decoder width multiplier",
    )
    parser.add_argument(
        "--teacher_skip_bottleneck_mult",
        type=float,
        default=1.0,
        help="Teacher skip bottleneck multiplier",
    )
    parser.add_argument(
        "--teacher_use_skip_bottleneck",
        type=int,
        default=0,
        help="Teacher skip bottleneck enable flag",
    )
    parser.add_argument("--teacher_drop_rate", type=float, default=0.0, help="Teacher drop_rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--kd_weight", type=float, default=1.0, help="Weight for KL distillation loss")
    parser.add_argument("--logit_mse_weight", type=float, default=0.1, help="Weight for raw-logit MSE")
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum train steps for this run; 0 means unlimited")
    parser.add_argument("--save_every_steps", type=int, default=500, help="Save a checkpoint every N train steps")
    parser.add_argument("--log_every_steps", type=int, default=20, help="Print train losses every N train steps")
    parser.add_argument("--num_reader_threads", type=int, default=2, help="Number of DataReader worker threads")
    parser.add_argument("--queue_size", type=int, default=10, help="DataReader queue capacity in batches")
    parser.add_argument(
        "--reader_threads",
        type=int,
        default=0,
        help="Legacy alias for --num_reader_threads; 0 means unused",
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=-1,
        help="Legacy smoke-test cap; prefer --max_steps",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save a student checkpoint every N epochs",
    )
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.student_depth is None:
        args.student_depth = int(args.depth)
    if args.student_filters_root is None:
        args.student_filters_root = int(args.filters_root)
    if args.student_filters_cap is None:
        args.student_filters_cap = args.filters_cap
    if args.student_decoder_width_mult is None:
        args.student_decoder_width_mult = float(args.decoder_width_mult)
    if args.student_skip_bottleneck_mult is None:
        args.student_skip_bottleneck_mult = float(args.skip_bottleneck_mult)
    if args.student_use_skip_bottleneck is None:
        args.student_use_skip_bottleneck = int(args.use_skip_bottleneck)
    if args.student_drop_rate is None:
        args.student_drop_rate = float(args.drop_rate)

    default_b_student = (
        int(args.student_depth) == 4
        and int(args.student_filters_root) == 4
        and args.student_filters_cap is None
        and float(args.student_decoder_width_mult) == 1.0
        and int(args.student_use_skip_bottleneck) == 0
        and float(args.student_skip_bottleneck_mult) == 1.0
    )
    if args.student_init_dir is None and default_b_student:
        args.student_init_dir = DEFAULT_STUDENT_INIT_DIR

    args.depth = int(args.student_depth)
    args.filters_root = int(args.student_filters_root)
    args.filters_cap = args.student_filters_cap
    args.decoder_width_mult = float(args.student_decoder_width_mult)
    args.skip_bottleneck_mult = float(args.student_skip_bottleneck_mult)
    args.use_skip_bottleneck = int(args.student_use_skip_bottleneck)
    args.drop_rate = float(args.student_drop_rate)
    args.kernel_size = list(args.kernel_size)
    args.pool_size = list(args.pool_size)
    args.dilation_rate = list(args.dilation_rate)
    if args.reader_threads > 0:
        args.num_reader_threads = int(args.reader_threads)
    args.queue_capacity = max(int(args.queue_size), 1) * int(args.batch_size)
    return args


def namespace_copy(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def build_model_config(
    args: argparse.Namespace,
    data_reader,
    depth: int,
    filters_root: int,
    filters_cap,
    decoder_width_mult: float,
    skip_bottleneck_mult: float,
    use_skip_bottleneck: int,
    drop_rate: float,
    mode: str,
    weight_decay: float,
):
    model_args = namespace_copy(args)
    model_args.mode = mode
    model_args.depth = int(depth)
    model_args.filters_root = int(filters_root)
    model_args.filters_cap = None if filters_cap is None else int(filters_cap)
    model_args.decoder_width_mult = float(decoder_width_mult)
    model_args.skip_bottleneck_mult = float(skip_bottleneck_mult)
    model_args.use_skip_bottleneck = int(use_skip_bottleneck)
    model_args.drop_rate = float(drop_rate)
    model_args.weight_decay = float(weight_decay)
    model_args.distill_enable = 0
    return set_config(model_args, data_reader)


def resolve_checkpoint(path_or_dir, kind: str, required: bool = True):
    if path_or_dir is None:
        if required:
            raise ValueError(f"{kind} checkpoint path is required")
        return None
    path = os.path.normpath(path_or_dir)
    if os.path.isdir(path):
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt is None:
            raise FileNotFoundError(f"No {kind} checkpoint found under {path}")
        return ckpt
    for suffix in (".index", ".meta"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    data_suffix = ".data-00000-of-00001"
    if path.endswith(data_suffix):
        path = path[: -len(data_suffix)]
    if not (os.path.exists(path + ".index") or os.path.exists(path)):
        raise FileNotFoundError(f"{kind} checkpoint not found: {path}")
    return path


def strip_scope(var_name: str, scope: str) -> str:
    prefix = scope.rstrip("/") + "/"
    if not var_name.startswith(prefix):
        raise ValueError(f"Variable {var_name!r} is not under scope {scope!r}")
    return var_name[len(prefix) :]


def is_optimizer_variable(stripped_name: str) -> bool:
    leaf = stripped_name.split("/")[-1]
    if leaf in {"Adam", "Adam_1", "Momentum", "beta1_power", "beta2_power"}:
        return True
    if leaf.startswith("Adam_") or leaf.startswith("Momentum_"):
        return True
    return False


def checkpoint_var_map(scope: str, variables=None):
    if variables is None:
        variables = tf.compat.v1.global_variables(scope=scope)
    var_map = {}
    for var in variables:
        stripped = strip_scope(var.op.name, scope)
        if is_optimizer_variable(stripped):
            continue
        if stripped in var_map:
            raise ValueError(f"Duplicate checkpoint variable name after stripping {scope}: {stripped}")
        var_map[stripped] = var
    return var_map


def print_restore_debug(ckpt: str, var_map, kind: str) -> None:
    try:
        ckpt_names = {name for name, _ in tf.train.list_variables(ckpt)}
    except Exception as exc:
        print(f"[DEBUG] Failed to list {kind} checkpoint variables: {exc}")
        return
    graph_names = set(var_map.keys())
    missing = sorted(graph_names - ckpt_names)
    extra = sorted(ckpt_names - graph_names)
    print(f"[DEBUG] {kind} checkpoint path: {ckpt}")
    print(f"[DEBUG] {kind} variables expected by graph: {len(graph_names)}")
    print(f"[DEBUG] {kind} variables available in checkpoint: {len(ckpt_names)}")
    if missing:
        print(f"[DEBUG] First 30 graph variables missing from {kind} checkpoint:")
        for name in missing[:30]:
            print(f"  {name}")
    if extra:
        print(f"[DEBUG] First 30 extra variables in {kind} checkpoint:")
        for name in extra[:30]:
            print(f"  {name}")


def restore_with_debug(sess, saver, ckpt: str, var_map, kind: str) -> None:
    logging.info("Restoring %s checkpoint: %s", kind, ckpt)
    try:
        saver.restore(sess, ckpt)
    except Exception:
        print_restore_debug(ckpt, var_map, kind)
        raise


def build_kl_distill_loss(student_logits, teacher_logits, hard_loss, args):
    with tf.compat.v1.variable_scope("online_distill_loss"):
        temperature = tf.constant(float(args.temperature), dtype=tf.float32, name="temperature")
        kd_weight = tf.constant(float(args.kd_weight), dtype=tf.float32, name="kd_weight")
        mse_weight = tf.constant(float(args.logit_mse_weight), dtype=tf.float32, name="logit_mse_weight")

        student_logits = tf.convert_to_tensor(value=student_logits, dtype=tf.float32)
        teacher_logits = tf.stop_gradient(tf.convert_to_tensor(value=teacher_logits, dtype=tf.float32))

        flat_student = tf.reshape(student_logits / temperature, [-1, tf.shape(input=student_logits)[-1]])
        flat_teacher = tf.reshape(teacher_logits / temperature, [-1, tf.shape(input=teacher_logits)[-1]])

        teacher_prob = tf.nn.softmax(flat_teacher, axis=-1, name="teacher_prob_T")
        teacher_log_prob = tf.nn.log_softmax(flat_teacher, axis=-1, name="teacher_log_prob_T")
        student_log_prob = tf.nn.log_softmax(flat_student, axis=-1, name="student_log_prob_T")

        kl_map = tf.reduce_sum(input_tensor=teacher_prob * (teacher_log_prob - student_log_prob), axis=-1)
        kd_kl = tf.identity(tf.reduce_mean(input_tensor=kl_map) * temperature * temperature, name="kd_kl")
        logit_mse = tf.identity(tf.reduce_mean(input_tensor=tf.square(student_logits - teacher_logits)), name="logit_mse")
        total_loss = tf.identity(
            tf.cast(hard_loss, tf.float32) + kd_weight * kd_kl + mse_weight * logit_mse,
            name="total_loss",
        )
    return total_loss, kd_kl, logit_mse


def build_optimizer(total_loss, student, student_trainable_vars, args):
    with tf.compat.v1.variable_scope("online_distill_train"):
        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=float(student.learning_rate),
            global_step=student.global_step,
            decay_steps=int(student.decay_step),
            decay_rate=float(student.decay_rate),
            staircase=True,
            name="learning_rate",
        )
        if args.optimizer == "momentum":
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=student.momentum)
        elif args.optimizer == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope="student")
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                total_loss,
                global_step=student.global_step,
                var_list=student_trainable_vars,
                name="student_minimize",
            )
    return train_op, learning_rate


def build_summaries(total_loss, hard_loss, kd_kl, logit_mse, learning_rate):
    scalars = [
        tf.compat.v1.summary.scalar("loss/total", total_loss),
        tf.compat.v1.summary.scalar("loss/task", hard_loss),
        tf.compat.v1.summary.scalar("loss/kd_kl", kd_kl),
        tf.compat.v1.summary.scalar("loss/logit_mse", logit_mse),
        tf.compat.v1.summary.scalar("learning_rate", learning_rate),
    ]
    return tf.compat.v1.summary.merge(scalars)


def write_config(log_dir: str, args: argparse.Namespace, student_config, teacher_config) -> None:
    with open(os.path.join(log_dir, "config.log"), "w") as fp:
        fp.write("[args]\n")
        for key, value in sorted(vars(args).items()):
            fp.write(f"{key}: {value}\n")
        fp.write("\n[student_config]\n")
        fp.write("\n".join("%s: %s" % item for item in vars(student_config).items()))
        fp.write("\n\n[teacher_config]\n")
        fp.write("\n".join("%s: %s" % item for item in vars(teacher_config).items()))
        fp.write("\n")


def make_log_dir(base_dir: str) -> str:
    current_time = time.strftime("%y%m%d-%H%M%S")
    base_path = Path(base_dir).expanduser()
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    log_dir = os.path.normpath(str(base_path.resolve() / current_time)).replace("\\", "/")
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    return log_dir


def build_readers(args: argparse.Namespace):
    coord = tf.train.Coordinator()
    train_reader = QuietDataReader(
        signal_dir=args.train_signal_dir,
        signal_list=args.train_signal_list,
        noise_dir=args.train_noise_dir,
        noise_list=args.train_noise_list,
        queue_size=args.queue_capacity,
        coord=coord,
    )
    valid_reader = None
    if (args.valid_signal_list is not None) and (args.valid_noise_list is not None):
        valid_reader = QuietDataReader(
            signal_dir=args.valid_signal_dir,
            signal_list=args.valid_signal_list,
            noise_dir=args.valid_noise_dir,
            noise_list=args.valid_noise_list,
            queue_size=args.queue_capacity,
            coord=coord,
        )
    return coord, train_reader, valid_reader


def run_training(args: argparse.Namespace) -> int:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    coord, train_reader, valid_reader = build_readers(args)
    logging.info("Dataset size: training %d, validation %d", train_reader.n_signal, 0 if valid_reader is None else valid_reader.n_signal)

    log_dir = make_log_dir(args.log_dir)
    logging.info("Training log: %s", log_dir)

    with tf.compat.v1.name_scope("Input_Batch"):
        train_batch = train_reader.dequeue(args.batch_size)
        valid_batch = valid_reader.dequeue(args.batch_size) if valid_reader is not None else None

    student_config = build_model_config(
        args,
        train_reader,
        depth=args.student_depth,
        filters_root=args.student_filters_root,
        filters_cap=args.student_filters_cap,
        decoder_width_mult=args.student_decoder_width_mult,
        skip_bottleneck_mult=args.student_skip_bottleneck_mult,
        use_skip_bottleneck=args.student_use_skip_bottleneck,
        drop_rate=args.student_drop_rate,
        mode="train",
        weight_decay=args.weight_decay,
    )
    teacher_config = build_model_config(
        args,
        train_reader,
        depth=args.teacher_depth,
        filters_root=args.teacher_filters_root,
        filters_cap=args.teacher_filters_cap,
        decoder_width_mult=args.teacher_decoder_width_mult,
        skip_bottleneck_mult=args.teacher_skip_bottleneck_mult,
        use_skip_bottleneck=args.teacher_use_skip_bottleneck,
        drop_rate=args.teacher_drop_rate,
        mode="pred",
        weight_decay=0.0,
    )
    write_config(log_dir, args, student_config, teacher_config)

    x_input = tf.compat.v1.placeholder(
        dtype=tf.float32,
        shape=[None, train_reader.X_shape[0], train_reader.X_shape[1], train_reader.X_shape[-1]],
        name="distill_X",
    )
    y_target = tf.compat.v1.placeholder(
        dtype=tf.float32,
        shape=[None, train_reader.Y_shape[0], train_reader.Y_shape[1], train_reader.Y_shape[-1]],
        name="distill_Y",
    )

    with tf.compat.v1.variable_scope("student"):
        student = UNet(config=student_config, input_batch=[x_input, y_target], mode="train")
    student_model_var_map = checkpoint_var_map("student")
    student_trainable_vars = tf.compat.v1.trainable_variables(scope="student")

    with tf.compat.v1.variable_scope("teacher"):
        teacher = UNet(config=teacher_config, input_batch=[x_input], mode="pred")
    teacher_var_map = checkpoint_var_map("teacher")

    total_loss, kd_kl, logit_mse = build_kl_distill_loss(student.logits, teacher.logits, student.hard_loss, args)
    train_op, learning_rate = build_optimizer(total_loss, student, student_trainable_vars, args)
    with tf.control_dependencies([train_op]):
        global_step_after_train = tf.identity(student.global_step, name="global_step_after_train")
    summary_op = build_summaries(total_loss, student.hard_loss, kd_kl, logit_mse, learning_rate)

    student_saver = tf.compat.v1.train.Saver(student_model_var_map, max_to_keep=5)
    teacher_saver = tf.compat.v1.train.Saver(teacher_var_map)

    teacher_ckpt = resolve_checkpoint(args.teacher_checkpoint_dir, "teacher")
    resume_ckpt = resolve_checkpoint(args.model_dir, "student resume", required=False) if args.model_dir else None
    if resume_ckpt:
        student_ckpt = resume_ckpt
        reset_global_step = False
    elif args.student_init_ckpt:
        student_ckpt = resolve_checkpoint(args.student_init_ckpt, "student init")
        reset_global_step = True
    elif args.student_init_dir:
        student_ckpt = resolve_checkpoint(args.student_init_dir, "student init")
        reset_global_step = True
    else:
        student_ckpt = None
        reset_global_step = False

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False

    train_steps_per_epoch = int(np.ceil(train_reader.n_signal / float(args.batch_size)))
    valid_steps_per_epoch = 0
    if valid_reader is not None:
        valid_steps_per_epoch = int(np.ceil(valid_reader.n_signal / float(args.batch_size)))

    logging.info(
        "DataReader queue_size=%d batches, batch_size=%d, effective queue_capacity=%d samples",
        args.queue_size,
        args.batch_size,
        args.queue_capacity,
    )

    sess = tf.compat.v1.Session(config=sess_config)
    summary_writer = None
    loss_log = None
    train_threads = []
    valid_threads = []
    exit_code = 0
    try:
        try:
            summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        except Exception as exc:
            logging.warning("TensorBoard writer disabled for %s: %s", log_dir, exc)
            summary_writer = None
        sess.run(tf.compat.v1.global_variables_initializer())

        restore_with_debug(sess, teacher_saver, teacher_ckpt, teacher_var_map, "teacher")
        if student_ckpt is not None:
            restore_with_debug(sess, student_saver, student_ckpt, student_model_var_map, "student")
        if reset_global_step:
            sess.run(student.global_step.assign(0))
            logging.info("Reset student global_step to 0 after warm start")

        train_threads = train_reader.start_threads(sess, n_threads=args.num_reader_threads)
        valid_threads = (
            valid_reader.start_threads(sess, n_threads=args.num_reader_threads) if valid_reader is not None else []
        )
        loss_log = open(os.path.join(log_dir, "loss.log"), "w")

        mean_loss = 0.0
        steps_this_run = 0
        last_global_step = 0

        def save_student_checkpoint(global_step: int, final: bool = False) -> str:
            ckpt_path = student_saver.save(
                sess,
                os.path.join(log_dir, f"model_{global_step}.ckpt"),
                write_meta_graph=False,
            )
            final_suffix = " final=True" if final else ""
            print(f"[SAVE] step={global_step} checkpoint={ckpt_path}{final_suffix}", flush=True)
            return ckpt_path

        def run_train_step(epoch: int, epoch_step: int, progress) -> int:
            nonlocal mean_loss, steps_this_run, last_global_step

            x_batch, y_batch = sess.run(train_batch)
            feed = {
                x_input: x_batch,
                y_target: y_batch,
                student.drop_rate: args.student_drop_rate,
                student.is_training: True,
            }
            global_step, summary, total, task, kd, mse, lr = sess.run(
                [
                    global_step_after_train,
                    summary_op,
                    total_loss,
                    student.hard_loss,
                    kd_kl,
                    logit_mse,
                    learning_rate,
                ],
                feed_dict=feed,
            )
            if summary_writer is not None:
                summary_writer.add_summary(summary, global_step)

            steps_this_run += 1
            global_step = int(global_step)
            last_global_step = global_step
            mean_loss += (float(total) - mean_loss) / steps_this_run
            progress.set_description(
                f"{Path(log_dir).name}: step={global_step}, loss={total:.6f}, mean={mean_loss:.6f}"
            )
            loss_log.write(
                "step: {}, epoch: {}, epoch_step: {}, total_loss: {}, task_loss: {}, "
                "kd_loss: {}, logit_mse_loss: {}, lr: {}, mean_loss: {}\n".format(
                    global_step, epoch, epoch_step, total, task, kd, mse, lr, mean_loss
                )
            )
            loss_log.flush()

            should_log = (
                steps_this_run == 1
                or (steps_this_run % args.log_every_steps) == 0
                or (args.max_steps > 0 and steps_this_run == args.max_steps)
            )
            if should_log:
                print(
                    "[TRAIN] step={} total_loss={:.6f} task_loss={:.6f} "
                    "kd_loss={:.6f} logit_mse_loss={:.6f}".format(
                        global_step, total, task, kd, mse
                    ),
                    flush=True,
                )

            if global_step > 0 and (global_step % args.save_every_steps) == 0:
                save_student_checkpoint(global_step)
            return global_step

        if args.max_steps > 0:
            logging.info("Running step-driven training for %d train steps", args.max_steps)
            progress = tqdm(total=args.max_steps, desc=f"{Path(log_dir).name}: ")
            try:
                while steps_this_run < args.max_steps:
                    epoch = steps_this_run // max(train_steps_per_epoch, 1)
                    epoch_step = steps_this_run % max(train_steps_per_epoch, 1)
                    run_train_step(epoch, epoch_step, progress)
                    progress.update(1)
            finally:
                progress.close()
        else:
            logging.info("Running epoch-driven training for %d epochs", args.epochs)
            epoch_train_steps = train_steps_per_epoch
            if args.max_steps_per_epoch > 0:
                epoch_train_steps = min(epoch_train_steps, int(args.max_steps_per_epoch))
            for epoch in range(args.epochs):
                progress = tqdm(range(epoch_train_steps), desc=f"{Path(log_dir).name}: ")
                for epoch_step in progress:
                    run_train_step(epoch, epoch_step, progress)

                if valid_reader is not None and valid_steps_per_epoch > 0:
                    valid_mean = 0.0
                    for valid_step in tqdm(range(valid_steps_per_epoch), desc="Valid: "):
                        x_batch, y_batch = sess.run(valid_batch)
                        feed = {
                            x_input: x_batch,
                            y_target: y_batch,
                            student.drop_rate: 0.0,
                            student.is_training: False,
                        }
                        total, task, kd, mse, preds = sess.run(
                            [total_loss, student.hard_loss, kd_kl, logit_mse, student.preds],
                            feed_dict=feed,
                        )
                        valid_mean += (float(total) - valid_mean) / (valid_step + 1)
                        loss_log.write(
                            "Valid: {}, step: {}, total_loss: {}, task_loss: {}, kd_loss: {}, "
                            "logit_mse_loss: {}, mean_loss: {}\n".format(
                                epoch, valid_step, total, task, kd, mse, valid_mean
                            )
                        )
                    logging.info("Epoch %d validation mean loss: %.6f", epoch, valid_mean)

        if last_global_step > 0:
            save_student_checkpoint(last_global_step, final=True)
    except Exception:
        print("[FATAL] train_b_d4_r4_distill.py failed", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        if loss_log is not None:
            loss_log.close()
        if summary_writer is not None:
            summary_writer.close()
        coord.request_stop()
        all_reader_threads = train_threads + valid_threads
        try:
            sess.run(train_reader.queue.close(cancel_pending_enqueues=True))
            if valid_reader is not None:
                sess.run(valid_reader.queue.close(cancel_pending_enqueues=True))
        except Exception as exc:
            logging.warning("Queue close warning during shutdown: %s", exc)
        if all_reader_threads:
            try:
                coord.join(all_reader_threads, stop_grace_period_secs=5)
            except Exception as exc:
                logging.warning("Reader thread join warning during shutdown: %s", exc)
        sess.close()

    return exit_code


def main() -> int:
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    args = normalize_args(build_script_arg_parser().parse_args())
    if args.mode != "train":
        raise ValueError("train_b_d4_r4_distill.py only supports --mode=train")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.kd_weight < 0 or args.logit_mse_weight < 0:
        raise ValueError("--kd_weight and --logit_mse_weight must be >= 0")
    if args.student_decoder_width_mult <= 0 or args.teacher_decoder_width_mult <= 0:
        raise ValueError("student/teacher decoder width multipliers must be > 0")
    if args.student_skip_bottleneck_mult <= 0 or args.teacher_skip_bottleneck_mult <= 0:
        raise ValueError("student/teacher skip bottleneck multipliers must be > 0")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be >= 0")
    if args.save_every_steps <= 0:
        raise ValueError("--save_every_steps must be > 0")
    if args.log_every_steps <= 0:
        raise ValueError("--log_every_steps must be > 0")
    if args.num_reader_threads <= 0:
        raise ValueError("--num_reader_threads must be > 0")
    if args.queue_size <= 0:
        raise ValueError("--queue_size must be > 0")
    try:
        return run_training(args)
    except Exception:
        print("[FATAL] train_b_d4_r4_distill.py failed", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
