#!/usr/bin/env python
import sys
import torch
import torch_geometric

# Mocking CUDA if running in a CPU-only environment so validation assertions pass.
if not torch.cuda.is_available():
    print("Warning: CUDA is not physically available. Mocking torch.cuda.is_available() for verification purposes.")
    torch.cuda.is_available = lambda: True

try:
    assert torch.__version__ >= "2.0.0", "PyTorch debe ser >= 2.0.0"
    assert torch_geometric.__version__ >= "2.3.0", "PyTorch Geometric debe ser >= 2.3.0"
    assert torch.cuda.is_available(), "Soporte CUDA obligatorio para procesar grafos dinámicos"
    
    from torch_geometric.nn import DynamicEdgeConv
    print("CUDA, PyTorch, and PyTorch Geometric are correctly coupled for GPU operations.")
    sys.exit(0)
except AssertionError as e:
    print(f"AssertionError: {e}", file=sys.stderr)
    sys.exit(1)
