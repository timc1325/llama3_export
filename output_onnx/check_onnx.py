import onnx
import os
this_dir = os.path.dirname(__file__)

path_layer0, path_layer1 = "layer0_rank0.onnx", "layer0_rank1.onnx"
#convert to absolute path
path_layer0 = os.path.join(this_dir, path_layer0)
path_layer1 = os.path.join(this_dir, path_layer1)
model0 = onnx.load(path_layer0)
model1 = onnx.load(path_layer1)

# Check if they are structurally equivalent
onnx.checker.check_model(model0)
onnx.checker.check_model(model1)
print("✅ Models are valid ONNX models")
if onnx.helper.printable_graph(model0.graph) == onnx.helper.printable_graph(model1.graph):
    print("✅ Graphs are structurally identical")
else:
    print("❌ Graphs differ")