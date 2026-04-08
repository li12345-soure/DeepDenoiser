import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def describe_tensor(d):
    q = d.get("quantization_parameters", {})
    scales = q.get("scales", [])
    zero_points = q.get("zero_points", [])
    qdim = q.get("quantized_dimension", None)
    return {
        "name": d.get("name"),
        "shape": d.get("shape"),
        "shape_signature": d.get("shape_signature"),
        "dtype": str(d.get("dtype")),
        "quantization": d.get("quantization"),
        "quant_scales_len": len(scales) if hasattr(scales, "__len__") else 0,
        "quant_zero_points_len": len(zero_points) if hasattr(zero_points, "__len__") else 0,
        "quant_dim": qdim,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal TFLite runtime test for DeepDenoiser")
    parser.add_argument("--model", default="deepdenoiser_float.tflite", help="Path to .tflite model")
    parser.add_argument("--height", type=int, default=31, help="Input height")
    parser.add_argument("--width", type=int, default=201, help="Input width")
    parser.add_argument("--channels", type=int, default=2, help="Input channels")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--use_random", action="store_true", help="Use random input instead of zeros")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] Loading model: {model_path.resolve()}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n[INFO] Inputs:")
    for i, d in enumerate(input_details):
        print(f"  Input {i}: {describe_tensor(d)}")

    print("\n[INFO] Outputs:")
    for i, d in enumerate(output_details):
        print(f"  Output {i}: {describe_tensor(d)}")

    if len(input_details) != 1:
        print(f"\n[WARN] Expected 1 input, got {len(input_details)}. This script will only feed input 0.")

    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    target_shape = [args.batch, args.height, args.width, args.channels]
    print(f"\n[INFO] Resizing input 0 to: {target_shape}")
    interpreter.resize_tensor_input(input_index, target_shape, strict=False)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n[INFO] Inputs after allocate_tensors():")
    for i, d in enumerate(input_details):
        print(f"  Input {i}: {describe_tensor(d)}")

    print("\n[INFO] Outputs after allocate_tensors():")
    for i, d in enumerate(output_details):
        print(f"  Output {i}: {describe_tensor(d)}")

    if args.use_random:
        x = np.random.randn(*target_shape).astype(np.float32)
    else:
        x = np.zeros(target_shape, dtype=np.float32)

    x = x.astype(input_dtype)

    print(f"\n[INFO] Feeding input with shape={x.shape}, dtype={x.dtype}")
    interpreter.set_tensor(input_index, x)

    print("[INFO] Running interpreter.invoke() ...")
    interpreter.invoke()
    print("[OK] invoke() succeeded")

    for i, d in enumerate(output_details):
        y = interpreter.get_tensor(d["index"])
        print(f"\n[INFO] Output {i}")
        print(f"  name  : {d['name']}")
        print(f"  shape : {y.shape}")
        print(f"  dtype : {y.dtype}")
        print(f"  min   : {y.min()}")
        print(f"  max   : {y.max()}")
        print(f"  mean  : {y.mean()}")

    print("\n[OK] TFLite runtime smoke test completed.")


if __name__ == "__main__":
    main()
