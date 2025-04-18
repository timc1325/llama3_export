import argparse
from transformers import AutoModel, AutoTokenizer
from torch.distributed.device_mesh import DeviceMesh

from .distributed_utils import setup_distributed, destroy_distributed
from .model_wrappers import LlamaDecoderLayerExportable
from .tp_apply import apply_tp_llama3
from .weight_loader import load_layer0_weights_into_custom_block
from .onnx_export import export_to_onnx
import torch.nn as nn
import os


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.layers = nn.ModuleDict({str(i): block for i, block in enumerate(model.layers)})
    return tokenizer, model

def main(args):
    local_rank, world_size, rank = setup_distributed() 
    tokenizer, newmodel = load_model(args.model_name) #load model 
    mesh = DeviceMesh("cuda", list(range(world_size))) 
    apply_tp_llama3(
        newmodel,
        mesh,
        loss_parallel=False,
        enable_float8_tensorwise_tp=False,
        enable_async_tp=False
    )
    layer0 = newmodel.layers["0"]
    model = LlamaDecoderLayerExportable(world_size=world_size)
    load_layer0_weights_into_custom_block(layer0, model)

    output_dir = os.path.join(os.path.dirname(__file__), "output_onnx")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    filename = args.output.replace(".onnx", f"_rank{rank}.onnx")
    output_path = os.path.join(output_dir, filename)
    export_to_onnx(model, output_path)
    destroy_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sharded layer0 of llama3 to ONNX with Tensor Parallelism.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path.")
    parser.add_argument("--output", type=str, default="layer0.onnx", help="Output ONNX file path.")
    args = parser.parse_args()
    main(args)
