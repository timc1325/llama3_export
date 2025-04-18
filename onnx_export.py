import torch

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
