# Patches Kokoro-82M's half precision ONNX export so it never gives a waveform of NaN.
#
# Input: onnx/model_fp16.onnx from onnx-community/Kokoro-82M-v1.0-ONNX-timestamped at revision
# dd4401a9add81ac692d20e240d22ec9dda82cc29 (163,232,776 bytes, sha256 220724d5c5e0cc01be30f38faa6cf0c895a7cde6e7773e91db2973c8c7e5123c).
#
# The defect: the generator takes the phase of its harmonic source's STFT with atan2(imag, real), which the export
# writes as Atan(Div(imag, real)) and two Where nodes that pick the quadrant. In half precision a bin of that STFT can be
# exactly 0+0i, so the Div computes 0/0, which is NaN; Atan passes it on, and the layers after it spread it until the
# whole waveform is NaN, heard as silence. It happens to about one piece of text in eight.
#
# The fix puts Where(IsNaN(q), 0, q) between the Div and the Atan. The quotient is NaN only for 0/0, which now gives a
# phase of 0, what atan2(0, 0) gives in PyTorch; every other quotient, the infinities of x/0 included, passes through
# bit for bit, so the model is unchanged wherever it was not NaN. Opset, IR version and all other nodes stay as they are.
#
# Usage: python patch_atan.py model_fp16.onnx patched/model_fp16.onnx
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

source, target = sys.argv[1], sys.argv[2]
model = onnx.load(source)
graph = model.graph
atan = [node for node in graph.node if node.op_type == "Atan"]
assert len(atan) == 1, "expected the model's one Atan, the phase's"
atan = atan[0]
div = next((node for node in graph.node if atan.input[0] in node.output), None)
assert div is not None and div.op_type == "Div", "the Atan is not fed by a Div: already patched?"

quotient = div.output[0]
zero, nan, safe = quotient + "_zero", quotient + "_isnan", quotient + "_safe"
graph.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.float16), zero))
at = list(graph.node).index(atan)
graph.node.insert(at, helper.make_node("IsNaN", [quotient], [nan], name=div.name + "_isnan"))
graph.node.insert(at + 1, helper.make_node("Where", [nan, zero, quotient], [safe], name=div.name + "_safe"))
atan.input[0] = safe

onnx.checker.check_model(model)
onnx.save(model, target)
print(f"patched {div.name} -> {atan.name}; ir {model.ir_version}, opsets {[(o.domain, o.version) for o in model.opset_import]}")
