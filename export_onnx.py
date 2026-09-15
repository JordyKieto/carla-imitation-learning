"""Exports a Keras policy checkpoint to ONNX and checks the outputs match.

    python export_onnx.py                                   # artifacts/policy.keras -> artifacts/policy.onnx
    python export_onnx.py checkpoints/policy_epoch_010.keras --output my_policy.onnx

Requires: pip install tf2onnx onnx onnxruntime
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from model.policy import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default="artifacts/policy.keras")
    parser.add_argument("--output", help="defaults to the checkpoint path with .onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output or Path(args.checkpoint).with_suffix(".onnx"))

    model = load_policy(args.checkpoint)
    signature = [tf.TensorSpec((None,) + tuple(i.shape[1:]), tf.float32, name=i.name) for i in model.inputs]

    @tf.function(input_signature=signature)
    def policy(vision, lidar, telemetry, waypoints):
        return {"controls": model([vision, lidar, telemetry, waypoints], training=False)}

    # Freeze weights and captured constants (the baked-in positional embedding) into the graph.
    frozen = convert_variables_to_constants_v2(policy.get_concrete_function())
    onnx_model = tf2onnx.convert.from_graph_def(
        frozen.graph.as_graph_def(),
        input_names=[t.name for t in frozen.inputs],
        output_names=[t.name for t in frozen.outputs],
        opset=args.opset,
    )[0]
    # Clean tensor names: "vision:0" -> "vision", output -> "controls".
    renames = {t.name: t.name.split(":")[0] for t in frozen.inputs}
    renames.update({t.name: "controls" for t in frozen.outputs})
    for node in onnx_model.graph.node:
        node.input[:] = [renames.get(n, n) for n in node.input]
        node.output[:] = [renames.get(n, n) for n in node.output]
    for value in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        value.name = renames.get(value.name, value.name)
    onnx.save(onnx_model, str(output))

    rng = np.random.default_rng(0)
    feeds = {spec.name: rng.random((2,) + tuple(spec.shape[1:]), dtype=np.float32) for spec in signature}
    expected = model([feeds[spec.name] for spec in signature], training=False).numpy()
    session = ort.InferenceSession(str(output))
    actual = session.run(["controls"], feeds)[0]
    max_diff = float(np.max(np.abs(expected - actual)))
    print(f"[ONNX] wrote {output} | inputs={[(s.name, tuple(s.shape)) for s in signature]} max_abs_diff={max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError("ONNX outputs differ from Keras outputs")


if __name__ == "__main__":
    main()
