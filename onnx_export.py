import torch
import json
import subprocess
import logging
import ezkl

logger = logging.getLogger("llama3_export")

def export_to_onnx(model, output_path):
    model.eval()
    B, T = 2, 8
    hidden_dim = 2048
    x = torch.randn(B, T, hidden_dim)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

    torch.onnx.export(
        model,
        (x, position_ids),
        output_path,
        input_names=["x", "position_ids"],
        output_names=["output"],
        dynamic_axes={
            "x": {0: "batch", 1: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
            "output": {0: "batch", 1: "seq"},
        },
        opset_version=13
    )
    return x, position_ids

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


import ezkl

def run_ezkl_cmd(args, description):
    logger.info(f"▶ {description}")
    
    if description == "Compile":
        ezkl.compile_circuit(
            model=args[args.index("--model")+1],
            compiled_circuit=args[args.index("--compiled-model")+1],
            settings_path=args[args.index("--settings")+1],
        )

    elif description == "Gen Witness":
        ezkl.gen_witness(
            model=args[args.index("--compiled-model")+1],
            data=args[args.index("--input")+1],
            output=args[args.index("--witness")+1],
        )

    elif description == "Setup":
        ezkl.setup(
            model=args[args.index("--compiled-model")+1],
            pk_path=args[args.index("--pk-path")+1],
            vk_path=args[args.index("--vk-path")+1],
        )

    elif description == "Prove":
        ezkl.prove_circuit(
            witness=args[args.index("--witness")+1],
            model=args[args.index("--compiled-model")+1],
            pk_path=args[args.index("--pk")+1],
            proof_path=args[args.index("--proof")+1],
            proof_type="single",
        )
