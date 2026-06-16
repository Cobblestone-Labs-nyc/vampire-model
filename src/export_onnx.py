"""Export the Stage-2 student to ONNX, quantize (INT8/FP16), report size + a self-bench.

  python src/export_onnx.py --config config.yaml --quantize int8

Produces web/model/vampire_student.onnx for ONNX Runtime Web / WebGPU (in index.html) and
for the server EP. Prints final size and p50/p95 latency on available runtimes.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from train_student import UNetSmall


def bench(sess, res, inputs, n=50):
    import numpy as np
    xs = {k: v for k, v in inputs.items()}
    ts = []
    for _ in range(5):  # warmup
        sess.run(None, xs)
    for _ in range(n):
        t = time.perf_counter(); sess.run(None, xs); ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quantize", choices=["int8", "fp16", "none"], default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    st, ex = cfg["student"], cfg["export"]
    res = cfg["data"]["resolution"]
    quant = args.quantize or ex["quantize"]

    net = UNetSmall(st["base_channels"], st["width_mult"], st["film_strength"])
    net.load_state_dict(torch.load(Path(st["out_dir"]) / "student.pth", map_location="cpu"))
    net.eval()

    out_path = Path(ex["onnx_path"]); out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_x = torch.rand(1, 3, res, res)
    dummy_s = torch.tensor([0.65])
    inputs = (dummy_x, dummy_s) if st["film_strength"] else (dummy_x,)
    names = ["x", "strength"] if st["film_strength"] else ["x"]
    torch.onnx.export(
        net, inputs, str(out_path), input_names=names, output_names=["y"],
        dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, opset_version=17,
    )
    base_mb = out_path.stat().st_size / 1e6
    print(f"fp32 ONNX: {base_mb:.1f} MB -> {out_path}")

    final_path = out_path
    if quant == "int8":
        from onnxruntime.quantization import quantize_dynamic, QuantType
        final_path = out_path.with_name(out_path.stem + "_int8.onnx")
        quantize_dynamic(str(out_path), str(final_path), weight_type=QuantType.QInt8)
    elif quant == "fp16":
        import onnx
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(onnx.load(str(out_path)))
        final_path = out_path.with_name(out_path.stem + "_fp16.onnx")
        onnx.save(m, str(final_path))

    size_mb = final_path.stat().st_size / 1e6
    cap = cfg["acceptance"]["model_size_mb_max"]
    print(f"{quant} ONNX: {size_mb:.1f} MB -> {final_path}  "
          f"[{'OK' if size_mb <= cap else 'OVER CAP'} vs {cap} MB]")

    # Self-bench on available runtimes
    import onnxruntime as ort
    feed = {"x": dummy_x.numpy()}
    if st["film_strength"]:
        feed["strength"] = dummy_s.numpy()
    for ep in [("CUDAExecutionProvider", "cuda"), ("CPUExecutionProvider", "cpu")]:
        if ep[0] in ort.get_available_providers():
            sess = ort.InferenceSession(str(final_path), providers=[ep[0]])
            p50, p95 = bench(sess, res, feed)
            ok = p95 <= cfg["acceptance"]["latency_p95_s_max"]
            print(f"  {ep[1]:>5}: p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms "
                  f"[{'OK' if ok else 'SLOW'} vs {cfg['acceptance']['latency_p95_s_max']}s]")
    print("  webgpu: bench in-browser via web/infer.js (ORT-Web reports its own timing)")


if __name__ == "__main__":
    main()
