# 🦙 LLaMA 3 Export (Tensor Parallel + ONNX)

This repository contains tools and scripts to export sharded LLaMA 3.2-1B model blocks to ONNX format using tensor parallelism via PyTorch Distributed.  
Supports model slicing, rank-wise export, and ZK-proof-ready computation blocks.

## 🚀 Features

- ✅ Shards LLaMA decoder blocks using tensor parallelism
- ✅ Supports rank-wise ONNX export (`layer0_rank{N}.onnx`)
- ✅ Output folder auto-managed and ignored from Git
- ✅ Modular architecture (TP, loaders, ONNX, CLI)
- ✅ Production-grade logging, CLI, and packaging

## 📦 Quick Start

### Export Layer 0 Across 4 GPUs:
    ```bash
    torchrun --nproc_per_node=4 -m llama3_export.export
    ```
Inputs: Pretrained LLaMA-3.2-1B checkpoint
Outputs: Sharded ONNX graphs for each rank under the designated export folder
Applications: ZK-proof systems, distributed inference, efficient model compilation