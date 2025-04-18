# 🦙 LLaMA 3 Export (Tensor Parallel + ONNX)

This repository contains tools and scripts to export sharded LLaMA 3.2-1B model blocks to ONNX format using tensor parallelism via PyTorch Distributed.  
Supports model slicing, rank-wise export, and ZK-proof-ready computation blocks.

## 🚀 Features

- ✅ Shards LLaMA decoder blocks using tensor parallelism
- ✅ Supports rank-wise ONNX export (`layer0_rank{N}.onnx`)
- ✅ Output folder auto-managed and ignored from Git
- ✅ Modular architecture (TP, loaders, ONNX, CLI)
- ✅ Production-grade logging, CLI, and packaging

## 📦 Usage

### Export layer 0 across GPUs:

```bash
torchrun --nproc_per_node=4 -m llama3_export.export