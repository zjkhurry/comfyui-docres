"""ComfyUI nodes for DocRes document image restoration.

Split into a loader node and task nodes, following the usual ComfyUI pattern.
The loader reads the weights once and emits a handle; every task node consumes
that handle, so a chain of tasks reuses a single copy of the model in memory.
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pipeline  # noqa: E402
from ddc_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as DDC_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DDC_DISPLAY_NAME_MAPPINGS,
)

TASKS = ["dewarping", "deshadowing", "appearance", "deblurring", "binarization"]

# Tasks that need the MBD segmentation net to build their prompt.
MBD_TASKS = {"dewarping"}


def _to_rgb(image):
    """Normalize a task result to HxWx3 uint8 BGR.

    `binarization` returns a single-channel mask, so replicate it across the
    three channels; BGR->RGB is then a no-op on identical channels.
    """
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    return image


def _numpy_to_comfy(image):
    """HxWx3 uint8 RGB -> 1xHxWx3 float32 tensor in [0, 1]."""
    array = image.astype(np.float32) / 255.0
    return torch.from_numpy(array[None, ...])


class DocResLoader:
    """Load the DocRes model and hand it to the task nodes.

    Run this once, then connect its `model` output to as many DocRes task nodes
    as you need. All of them share these weights, so chaining tasks does not
    load the model repeatedly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("DOCRES_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "image/DocRes"
    DESCRIPTION = "Load the DocRes model. Connect its output to DocRes task nodes."

    def load(self, device="auto"):
        device = pipeline._resolve_device(None if device == "auto" else device)
        dtype = pipeline._resolve_dtype(device)
        model = pipeline.get_docres(device)
        bundle = pipeline.DocResBundle(model, device, dtype)
        print("[DocRes] model ready on %s (%s)" % (device, dtype))
        return (bundle,)


class DocResRestore:
    """Apply one DocRes restoration task using a loaded model.

    Task is communicated to the network through a 3-channel prompt map derived
    from the image, so the same weights cover dewarping, deshadowing, appearance
    enhancement, deblurring and binarization.

    Chain stages by feeding one node's IMAGE output into the next node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DOCRES_MODEL",),
                "image": ("IMAGE",),
                "task": (TASKS, {"default": "dewarping"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "image/DocRes"
    DESCRIPTION = "Restore a document image with a loaded DocRes model."

    def restore(self, model, image, task):
        if not isinstance(model, pipeline.DocResBundle):
            raise TypeError("model input must come from the DocRes Loader node")

        # ComfyUI tensors are RGB; the model and cv2 prompts expect BGR.
        rgb = (np.clip(image[0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])

        mbd = model.dewarper() if task in MBD_TASKS else None
        restored = pipeline.restore(
            model.model, bgr, task, model.device, model.dtype, mbd
        )

        # Pipeline works in BGR; ComfyUI expects RGB, 3 channels, float [0, 1].
        rgb = _to_rgb(restored)[:, :, ::-1]
        return (_numpy_to_comfy(np.ascontiguousarray(rgb)),)


NODE_CLASS_MAPPINGS = {
    "DocResLoader": DocResLoader,
    "DocResRestore": DocResRestore,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DocResLoader": "DocRes Loader",
    "DocResRestore": "DocRes Restore",
}

NODE_CLASS_MAPPINGS.update(DDC_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DDC_DISPLAY_NAME_MAPPINGS)
