#!/usr/bin/env python3
"""Online checkpoint distillation training for the B_d4_r4 student.

This script intentionally does not modify deepdenoiser/train.py. It reuses the
repo's DataReader, set_config, UNet, TensorBoard writer, and checkpoint style,
but builds a frozen teacher graph next to the student graph.
"""

import argparse
import logging
import multiprocessing
import os
import sys
import time
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
DEFAULT_TEACHER_CKPT_DIR = r"G:\dd_distill_v2_logs\260417-133633"
DEFAULT_LOG_DIR = r"G:\dd_runs\B_d4_r4_distill_v1"


def build_script_arg_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Train B_d4_r4 with online frozen-teacher distillation"
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
        "--student_drop_rate",
        type=float,
        default=None,
        help="Student drop_rate; defaults to --drop_rate/0.0",
    )
    parser.add_argument(
        "--student_init_dir",
        default=DEFAULT_STUDENT_INIT_DIR,
        help="Directory containing the B_d4_r4 warm-start checkpoint",
    )
    parser.add_argument(
        "--teacher_checkpoint_dir",
        default=DEFAULT_TEACHER_CKPT_DIR,
        help="Directory containing the frozen teacher checkpoint",
    )
    parser.add_argument("--teacher_depth", type=int, default=4, help="Teacher depth")
    parser.add_argument("--teacher_filters_root", type=int, default=6, help="Teacher filters_root")
    parser.add_argument("--teacher_drop_rate", type=float, default=0.0, help="Teacher drop_rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--kd_weight", type=float, default=1.0, help="Weight for KL distillation loss")
    parser.add_argument("--logit_mse_weight", type=float, default=0.1, help="Weight for raw-logit MSE")
    parser.add_argument(
        "--reader_threads",
        type=int,
        default=0,
        help="Number of DataReader threads; 0 uses multiprocessing.cpu_count()",
    )
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=-1,
        help="Optional cap for quick smoke runs; -1 uses the full training set",
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
    if args.student_drop_rate is None:
        args.student_drop_rate = float(args.drop_rate)

    args.depth = int(args.student_depth)
    args.filters_root = int(args.student_filters_root)
    args.drop_rate = float(args.student_drop_rate)
    args.kernel_size = list(args.kernel_size)
    args.pool_size = list(args.pool_size)
    args.dilation_rate = list(args.dilation_rate)
    return args


def namespace_copy(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def build_model_config(
    args: argparse.Namespace,
    data_reader,
    depth: int,
    filters_root: int,
    drop_rate: float,
    mode: str,
    weight_decay: float,
):
    model_args = namespace_copy(args)
    model_args.mode = mode
    model_args.depth = int(depth)
    model_args.filters_root = int(filters_root)
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
    log_dir = os.path.normpath(os.path.join(base_dir, current_time)).replace("\\", "/")
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(os.path.join(log_dir, "figures"), exist_ok=True)
    return log_dir


def build_readers(args: argparse.Namespace):
    coord = tf.train.Coordinator()
    train_reader = DataReader(
        signal_dir=args.train_signal_dir,
        signal_list=args.train_signal_list,
        noise_dir=args.train_noise_dir,
        noise_list=args.train_noise_list,
        queue_size=args.batch_size * 2,
        coord=coord,
    )
    valid_reader = None
    if (args.valid_signal_list is not None) and (args.valid_noise_list is not None):
        valid_reader = DataReader(
            signal_dir=args.valid_signal_dir,
            signal_list=args.valid_signal_list,
            noise_dir=args.valid_noise_dir,
            noise_list=args.valid_noise_list,
            queue_size=args.batch_size * 2,
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
        drop_rate=args.student_drop_rate,
        mode="train",
        weight_decay=args.weight_decay,
    )
    teacher_config = build_model_config(
        args,
        train_reader,
        depth=args.teacher_depth,
        filters_root=args.teacher_filters_root,
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
    if args.max_steps_per_epoch > 0:
        train_steps_per_epoch = min(train_steps_per_epoch, args.max_steps_per_epoch)
    valid_steps_per_epoch = 0
    if valid_reader is not None:
        valid_steps_per_epoch = int(np.ceil(valid_reader.n_signal / float(args.batch_size)))
        if args.max_steps_per_epoch > 0:
            valid_steps_per_epoch = min(valid_steps_per_epoch, args.max_steps_per_epoch)

    reader_threads = args.reader_threads if args.reader_threads > 0 else multiprocessing.cpu_count()

    with tf.compat.v1.Session(config=sess_config) as sess:
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

        train_threads = train_reader.start_threads(sess, n_threads=reader_threads)
        valid_threads = valid_reader.start_threads(sess, n_threads=reader_threads) if valid_reader is not None else []
        loss_log = open(os.path.join(log_dir, "loss.log"), "w")
        try:
            mean_loss = 0.0
            total_seen = 0
            for epoch in range(args.epochs):
                progress = tqdm(range(train_steps_per_epoch), desc=f"{Path(log_dir).name}: ")
                for step in progress:
                    x_batch, y_batch = sess.run(train_batch)
                    feed = {
                        x_input: x_batch,
                        y_target: y_batch,
                        student.drop_rate: args.student_drop_rate,
                        student.is_training: True,
                    }
                    _, summary, global_step, total, task, kd, mse, lr = sess.run(
                        [
                            train_op,
                            summary_op,
                            student.global_step,
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
                    total_seen += 1
                    mean_loss += (float(total) - mean_loss) / total_seen
                    progress.set_description(
                        f"{Path(log_dir).name}: epoch={epoch}, loss={total:.6f}, mean={mean_loss:.6f}"
                    )
                    loss_log.write(
                        "Epoch: {}, step: {}, total: {}, task: {}, kd_kl: {}, logit_mse: {}, lr: {}, mean: {}\n".format(
                            epoch, step, total, task, kd, mse, lr, mean_loss
                        )
                    )
                    loss_log.flush()

                if ((epoch + 1) % args.save_every) == 0:
                    student_saver.save(sess, os.path.join(log_dir, f"model_{epoch}.ckpt"))

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
                            "Valid: {}, step: {}, total: {}, task: {}, kd_kl: {}, logit_mse: {}, mean: {}\n".format(
                                epoch, valid_step, total, task, kd, mse, valid_mean
                            )
                        )
                    logging.info("Epoch %d validation mean loss: %.6f", epoch, valid_mean)
        finally:
            loss_log.close()
            if summary_writer is not None:
                summary_writer.close()
            coord.request_stop()
            try:
                coord.join(train_threads + valid_threads, stop_grace_period_secs=10, ignore_live_threads=True)
            except Exception:
                pass
            try:
                sess.run(train_reader.queue.close(cancel_pending_enqueues=True))
                if valid_reader is not None:
                    sess.run(valid_reader.queue.close(cancel_pending_enqueues=True))
            except Exception:
                pass

    return 0


def main() -> int:
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    args = normalize_args(build_script_arg_parser().parse_args())
    if args.mode != "train":
        raise ValueError("train_b_d4_r4_distill.py only supports --mode=train")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.kd_weight < 0 or args.logit_mse_weight < 0:
        raise ValueError("--kd_weight and --logit_mse_weight must be >= 0")
    if args.save_every <= 0:
        raise ValueError("--save_every must be > 0")
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
