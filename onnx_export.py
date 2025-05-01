import torch
import json
import subprocess
import logging
import ezkl

logger = logging.getLogger("llama3_export")

def export_to_onnx(x, position_ids, model, output_path):
    model.eval()
    torch.onnx.export(
        model,
        (x, position_ids),
        output_path,
        export_params=True,
        input_names=["x", "position_ids"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=18
    )
    return x, position_ids

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


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
