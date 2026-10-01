"""Convert the original DocRes .pkl checkpoints into slim safetensors.

The released .pkl files store a full training checkpoint (model weights plus
AdamW optimizer state) under a `module.` prefix from DataParallel. Inference
needs neither, so this script keeps only the model tensors and strips the
prefix, cutting ~855 MB down to ~300 MB.

Usage (from the DocRes repo root):
    python comfyui-docres/convert_weights.py
"""

import os

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "models")

SOURCES = {
    "docres.safetensors": os.path.join(REPO, "checkpoints", "docres.pkl"),
    "mbd.safetensors": os.path.join(REPO, "data", "MBD", "checkpoint", "mbd.pkl"),
}


def convert(src, dst):
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint

    tensors = {}
    for name, tensor in state.items():
        if not hasattr(tensor, "numel"):
            continue
        clean = name[7:] if name.startswith("module.") else name
        tensors[clean] = tensor.contiguous().clone()

    save_file(tensors, dst)
    print("%s -> %s  (%d tensors, %.1f MB)" % (
        os.path.basename(src), os.path.basename(dst), len(tensors),
        os.path.getsize(dst) / 1024 / 1024))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for filename, src in SOURCES.items():
        if not os.path.isfile(src):
            raise SystemExit("missing source checkpoint: %s" % src)
        convert(src, os.path.join(OUT_DIR, filename))


if __name__ == "__main__":
    main()
