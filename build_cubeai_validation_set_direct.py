import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def run_cmd(cmd):
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def collect_from_one_file(python_exe, export_script, checkpoint_dir, src_npz, tmp_out_npz):
    cmd = [
        python_exe,
        export_script,
        '--checkpoint_dir', checkpoint_dir,
        '--npz', src_npz,
        '--save_npz', tmp_out_npz,
    ]
    run_cmd(cmd)

    d = np.load(tmp_out_npz)

    if 'model_input' not in d.files:
        raise KeyError(f"{tmp_out_npz} 里没有找到 'model_input'，现有 keys: {list(d.files)}")
    if 'preds' not in d.files:
        raise KeyError(f"{tmp_out_npz} 里没有找到 'preds'，现有 keys: {list(d.files)}")

    x = d['model_input'].astype(np.float32)
    y = d['preds'].astype(np.float32)

    if x.ndim != 4:
        raise ValueError(f'{src_npz} 导出的 model_input 维度异常: {x.shape}')
    if y.ndim != 4:
        raise ValueError(f'{src_npz} 导出的 preds 维度异常: {y.shape}')
    if x.shape != y.shape:
        raise ValueError(f'{src_npz} 的输入输出 shape 不一致: x={x.shape}, y={y.shape}')

    return x, y


def quantize_array(x, scale, zero_point, dtype):
    if scale == 0:
        raise ValueError('量化 scale 为 0，不合法。')
    q = np.round(x / scale + zero_point)
    info = np.iinfo(dtype)
    q = np.clip(q, info.min, info.max).astype(dtype)
    return q


def run_tflite_int8_inference(tflite_model, x_float):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    in_dtype = in_detail['dtype']
    out_dtype = out_detail['dtype']
    in_scale, in_zero = in_detail['quantization']
    out_scale, out_zero = out_detail['quantization']

    if in_dtype not in (np.int8, np.uint8):
        raise ValueError(f'这个 tflite 输入 dtype 不是 int8/uint8，而是: {in_dtype}')
    if out_dtype not in (np.int8, np.uint8):
        raise ValueError(f'这个 tflite 输出 dtype 不是 int8/uint8，而是: {out_dtype}')

    x_q = quantize_array(x_float, in_scale, in_zero, in_dtype)

    outputs_q = []
    input_index = in_detail['index']
    output_index = out_detail['index']

    for i in range(x_q.shape[0]):
        one = x_q[i:i + 1]

        current_shape = list(one.shape)
        if list(interpreter.get_input_details()[0]['shape']) != current_shape:
            interpreter.resize_tensor_input(input_index, current_shape)
            interpreter.allocate_tensors()
            input_index = interpreter.get_input_details()[0]['index']
            output_index = interpreter.get_output_details()[0]['index']

        interpreter.set_tensor(input_index, one)
        interpreter.invoke()
        y_q = interpreter.get_tensor(output_index)
        outputs_q.append(y_q.copy())

    y_q = np.concatenate(outputs_q, axis=0)

    meta = {
        'input_dtype': str(in_dtype),
        'output_dtype': str(out_dtype),
        'input_scale': float(in_scale),
        'input_zero_point': int(in_zero),
        'output_scale': float(out_scale),
        'output_zero_point': int(out_zero),
        'input_shape_runtime': list(x_q.shape),
        'output_shape_runtime': list(y_q.shape),
    }
    return x_q, y_q, meta


def main():
    parser = argparse.ArgumentParser(description='批量生成 X-CUBE-AI 验证集')
    parser.add_argument('--python_exe', default=r'.\\deepdenoiser_tflite_env\\Scripts\\python.exe', help='用于运行 export_tf_raw_debug.py 的 Python')
    parser.add_argument('--export_script', default=r'.\\export_tf_raw_debug.py', help='export_tf_raw_debug.py 路径')
    parser.add_argument('--checkpoint_dir', default=r'.\\model\\190614-104802', help='模型 checkpoint 目录')
    parser.add_argument('--input_glob', default=r'.\\Dataset\\pred\\*.npz', help='原始地震 npz 的 glob')
    parser.add_argument('--out_dir', default=r'.\\cubeai_validation', help='输出目录')
    parser.add_argument('--max_files', type=int, default=0, help='最多处理多少个文件，0 表示全部')
    parser.add_argument('--tflite_model', default='', help='可选：int8 tflite 模型路径')
    args = parser.parse_args()

    src_files = sorted(glob.glob(args.input_glob))
    if not src_files:
        raise FileNotFoundError(f'没有匹配到文件: {args.input_glob}')

    if args.max_files > 0:
        src_files = src_files[:args.max_files]

    os.makedirs(args.out_dir, exist_ok=True)

    all_x = []
    all_y = []
    manifest = []

    tmp_dir = tempfile.mkdtemp(prefix='cubeai_val_')
    try:
        for idx, src_npz in enumerate(src_files, start=1):
            tmp_out = os.path.join(tmp_dir, f'debug_{idx:04d}.npz')
            print(f'\n[{idx}/{len(src_files)}] processing: {src_npz}')

            x, y = collect_from_one_file(
                python_exe=args.python_exe,
                export_script=args.export_script,
                checkpoint_dir=args.checkpoint_dir,
                src_npz=src_npz,
                tmp_out_npz=tmp_out,
            )

            all_x.append(x)
            all_y.append(y)
            manifest.append({
                'source_file': src_npz,
                'num_samples': int(x.shape[0]),
                'sample_shape': list(x.shape[1:]),
            })

        x_f32 = np.concatenate(all_x, axis=0).astype(np.float32)
        y_f32 = np.concatenate(all_y, axis=0).astype(np.float32)

        in_f32_path = os.path.join(args.out_dir, 'val_input_f32.npy')
        out_f32_path = os.path.join(args.out_dir, 'val_output_f32.npy')
        np.save(in_f32_path, x_f32)
        np.save(out_f32_path, y_f32)

        print('\n=== FLOAT32 验证集已生成 ===')
        print('val_input_f32 :', in_f32_path, x_f32.shape, x_f32.dtype)
        print('val_output_f32:', out_f32_path, y_f32.shape, y_f32.dtype)

        summary = {
            'num_source_files': len(src_files),
            'total_samples': int(x_f32.shape[0]),
            'input_shape': list(x_f32.shape),
            'output_shape': list(y_f32.shape),
            'input_dtype': str(x_f32.dtype),
            'output_dtype': str(y_f32.dtype),
            'sources': manifest,
        }

        if args.tflite_model:
            x_q, y_q, qmeta = run_tflite_int8_inference(args.tflite_model, x_f32)

            in_q_path = os.path.join(args.out_dir, 'val_input_int8.npy')
            out_q_path = os.path.join(args.out_dir, 'val_output_int8.npy')
            np.save(in_q_path, x_q)
            np.save(out_q_path, y_q)

            print('\n=== INT8 验证集已生成 ===')
            print('val_input_int8 :', in_q_path, x_q.shape, x_q.dtype)
            print('val_output_int8:', out_q_path, y_q.shape, y_q.dtype)

            summary['int8'] = qmeta
            summary['int8']['input_file'] = in_q_path
            summary['int8']['output_file'] = out_q_path

        summary_path = os.path.join(args.out_dir, 'manifest.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print('\nmanifest:', summary_path)
        print('\n完成。把生成的 .npy 文件导入 X-CUBE-AI 的 Validation inputs / outputs 即可。')

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
