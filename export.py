import argparse
from transformers import AutoModel, AutoTokenizer
from torch.distributed.device_mesh import DeviceMesh

from .distributed_utils import setup_distributed, destroy_distributed
from .model_wrappers import LlamaDecoderLayerExportable
from .tp_apply import apply_tp_llama3
from .weight_loader import load_layer0_weights_into_custom_block
from .onnx_export import export_to_onnx, save_json, run_ezkl_cmd
import torch.nn as nn
import os
import torch

import logging
logger = logging.getLogger("llama3_export")

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.layers = nn.ModuleDict({str(i): block for i, block in enumerate(model.layers)})
    return tokenizer, model

def get_output_path(filename):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output_onnx"))
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)

def main(args):
    # Initialize distributed environment (sets rank, world size, etc.)
    local_rank, world_size, rank = setup_distributed() 
    tokenizer, newmodel = load_model(args.model_name) #load model 
    mesh = DeviceMesh("cuda", list(range(world_size))) 
    model = newmodel.layers["0"]
    apply_tp_llama3(
        newmodel,
        mesh,
        loss_parallel=False,
        enable_float8_tensorwise_tp=False,
        enable_async_tp=False
    )
    #Extract sharded Layer 0 and wrap it in an exportable block
    layer0 = newmodel.layers["0"]
    model = LlamaDecoderLayerExportable(world_size=world_size)
    load_layer0_weights_into_custom_block(layer0, model)
    model.eval()
    
    if rank == 0:

        # File paths
        paths = {
            "onnx": "network.onnx",
            "input": "input.json",
            "settings": "settings.json",
            "compiled": "network.compiled",
            "witness": "witness.json",
            "pk": "test.pk",
            "vk": "test.vk",
            "proof": "test.pf"
        }
        B, T, hidden_dim  = 2, 8, 2048
        x = torch.randn(B, T, hidden_dim)
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        export_to_onnx(x, position_ids, model, paths["onnx"])  #export model to onnx
        input_data = {
            "input_data": {
                "x": x.detach().numpy().reshape([-1]).tolist(),                     # input name must match ONNX input
                "position_ids": position_ids.detach().numpy().reshape([-1]).tolist()
            }
        }
        data_json = dict(input_data = [x.detach().numpy().reshape([-1]).tolist(),position_ids.detach().numpy().reshape([-1]).tolist()])
        save_json(data_json, paths["input"])
        
    logger.info("✅ Done.")
    destroy_distributed()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sharded layer0 of llama3 to ONNX with Tensor Parallelism.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path.")
    parser.add_argument("--output", type=str, default="layer0.onnx", help="Output ONNX file path.")
    args = parser.parse_args()
    main(args)
