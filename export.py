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
import ezkl

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
        x, position_ids = export_to_onnx(model, paths["onnx"])  #export model to onnx
        input_data = {
            "x": x.tolist(),
            "position_ids": position_ids.tolist()
        }
        input_data = {"x": x.tolist()}
        save_json(input_data, paths["input"])
        # Step 3: generate settings.json
        
        res = ezkl.gen_settings(model = paths["onnx"], output = paths["settings"])
        res = ezkl.calibrate_settings(paths["onnx"], paths["settings"], target="resources")
        # assert res == True
        
        # Step 4: compile the circuit
        res = ezkl.compile_circuit(model=paths["onnx"],
            compiled_circuit=paths["compiled"],
            settings_path=paths["settings"])
        assert res == True

        ezkl.gen_witness(
            compiled_model_path=paths["compiled"],
            input_path=paths["input"],
            output_path=paths["witness"]
        )

        ezkl.setup_circuit(
            compiled_model_path=paths["compiled"],
            pk_path=paths["pk"],
            vk_path=paths["vk"]
        )

        ezkl.prove_circuit(
            witness_path=paths["witness"],
            compiled_model_path=paths["compiled"],
            pk_path=paths["pk"],
            proof_path=paths["proof"],
            strategy="single"
        )

    logger.info("✅ Done.")
        
    destroy_distributed()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export sharded layer0 of llama3 to ONNX with Tensor Parallelism.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path.")
    parser.add_argument("--output", type=str, default="layer0.onnx", help="Output ONNX file path.")
    args = parser.parse_args()
    main(args)
