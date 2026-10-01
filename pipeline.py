"""DocRes inference pipeline (forward only).

Mirrors the original `inference.py` task handlers, minus the training-only
pieces. Models load once and are cached at module level so a ComfyUI graph can
run many images without re-reading weights.
"""

import copy
import gc
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import prompts
import tiles
import weights_fetch
from mbd.deeplab import DeepLab
from restormer_arch import Restormer

# Inference only: no autograd graph, no grad bookkeeping on any path.
torch.set_grad_enabled(False)

HERE = os.path.dirname(os.path.abspath(__file__))
DEWARP_INPUT_SIZE = 256
MAX_SIZE = 1600

_MODEL_CACHE = {}


def _find_weight(name):
    """Locate `name`, preferring a local copy and downloading it if absent.

    Search order: the node's own weights/ cache, then ComfyUI's model folders
    (so a user who placed the file there is not made to fetch it again), then
    Hugging Face.
    """
    candidates = [os.path.join(weights_fetch.WEIGHT_DIR, name)]
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.models_dir, name))
        for folder in folder_paths.get_folder_paths("checkpoints"):
            candidates.append(os.path.join(folder, name))
    except ImportError:
        pass

    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path

    return weights_fetch.ensure(name)


def _resolve_device(preferred=None):
    """Resolve the requested backend, or auto-detect the best available one."""
    if preferred in ("cuda", "mps", "cpu"):
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if preferred == "mps" and not (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available")
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(device):
    """Half precision everywhere, matching the original repo.

    The released inference code runs the whole pipeline in fp16 on every
    backend (`in_im.half()`, `model.half()`). That is not just a speed choice:
    Restormer's level-1 decoder holds 96-channel activations at full resolution,
    and its FFN projects to 255 channels, so a 2634x2096 scan needs several GB.
    Staying in fp32 doubles every one of those buffers and is what makes large
    scans run out of memory.
    """
    return torch.float16


class DocResBundle:
    """Loaded model handle passed from the loader node to the task nodes.

    ComfyUI passes this object along the wire between nodes, so a chain of task
    nodes reuses one set of weights instead of each node loading its own copy.
    `mbd` stays None until a task actually needs the dewarper.
    """

    def __init__(self, model, device, dtype):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.mbd = None

    def dewarper(self):
        """Lazily attach the MBD segmenter; only dewarping needs it."""
        if self.mbd is None:
            self.mbd = get_mbd(self.device)
        return self.mbd


def get_docres(device=None):
    """Load (and cache) the DocRes Restormer backbone."""
    device = _resolve_device(device)
    key = ("docres", str(device))
    if key not in _MODEL_CACHE:
        model = Restormer(
            inp_channels=6,
            out_channels=3,
            dim=48,
            num_blocks=[2, 3, 3, 4],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=True,
        )
        model.load_state_dict(load_file(_find_weight("docres.safetensors")))
        model.eval().to(device=device, dtype=_resolve_dtype(device))
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def get_mbd(device=None):
    """Load (and cache) the MBD segmentation net used for the dewarp mask."""
    device = _resolve_device(device)
    key = ("mbd", str(device))
    if key not in _MODEL_CACHE:
        model = DeepLab(num_classes=1, backbone="resnet", output_stride=16,
                        sync_bn=None, freeze_bn=False)
        model.load_state_dict(load_file(_find_weight("mbd.safetensors")))
        model.eval().to(device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _empty_backend_cache():
    """Return cached allocator blocks to the OS.

    Needed after every task, not just at teardown: the caching allocator keeps
    freed blocks reserved, so a large intermediate (e.g. a 2634x2093 tensor)
    stays charged to the process after the tensors die. On Apple silicon the GPU
    shares system memory, so those reserved blocks count against RAM and stack up
    across chained nodes until the machine stalls.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _to_tensor(img, device, dtype):
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=dtype)


def _postprocess(pred):
    return (pred[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)


def dewarping(model, img, mbd, device, dtype):
    """Flatten a warped page into a rectangle via a predicted dense flow."""
    im_org = img.copy()
    h, w = im_org.shape[:2]

    mask = _mbd_mask(mbd, im_org, device)
    im_masked = im_org.copy()
    im_masked[mask == 0] = 0
    base_coord = prompts.get_base_coord(DEWARP_INPUT_SIZE, DEWARP_INPUT_SIZE) / DEWARP_INPUT_SIZE
    mask_small = cv2.resize(mask, (DEWARP_INPUT_SIZE, DEWARP_INPUT_SIZE)) / 255.0
    prompt_org = np.concatenate((base_coord, np.expand_dims(mask_small, -1)), -1)

    img_in = cv2.resize(im_masked, (DEWARP_INPUT_SIZE, DEWARP_INPUT_SIZE))
    in_im = torch.cat((_to_tensor(img_in, device, dtype),
                       _to_tensor(prompt_org, device, dtype)), dim=1)

    # Dewarp runs in fp32, like the original repo. The shared model is cached in
    # fp16 to keep every other task's memory down, and model.float() would mutate
    # that shared instance in place, so cast a copy and leave the cache alone.
    flow_model = model.float() if dtype == torch.float32 else copy.deepcopy(model).float()
    pred = flow_model(in_im.float())
    flow = pred[0][:2].permute(1, 2, 0).cpu().numpy() + base_coord
    for _ in range(15):
        flow = cv2.blur(flow, (3, 3), borderType=cv2.BORDER_REPLICATE)

    flow = (cv2.resize(flow, (w, h)) * (w, h)).astype(np.float32)
    return cv2.remap(im_org, flow[:, :, 0], flow[:, :, 1], cv2.INTER_LINEAR)


def _mbd_mask(mbd, img, device):
    h_org, w_org = img.shape[:2]
    resized = cv2.resize(img, (448, 448))
    resized = cv2.GaussianBlur(resized, (15, 15), 0, 0)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = _to_tensor(resized, device, torch.float32)

    pred = mbd(tensor)
    mask = pred[:, 0:1, :, :]
    mask = F.interpolate(mask, (h_org, w_org))
    mask = (mask.squeeze(0).squeeze(0).float().cpu().numpy() * 255).astype(np.uint8)
    kernel = np.ones((3, 3))
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    return np.where(mask > 100, 255, 0).astype(np.uint8)


def _generic_restore(model, img, prompt, device, dtype):
    """Shared body for tasks whose output is a restored RGB (or tiled RGB) image."""
    h, w = img.shape[:2]
    in_im = np.concatenate((img, prompt), -1)

    if max(w, h) < MAX_SIZE:
        in_im, orig_h, orig_w = tiles.stride_integral(in_im, 8)
        out = _postprocess(torch.clamp(model(_to_tensor(in_im, device, dtype)), 0, 1))
        return out[:orig_h, :orig_w]

    resized = cv2.resize(in_im, (MAX_SIZE, MAX_SIZE))
    pred = _postprocess(torch.clamp(model(_to_tensor(resized, device, dtype)), 0, 1))
    # Model regresses a shading/illumination field; divide it out to recover the page.
    pred[pred == 0] = 1
    shadow_map = cv2.resize(img, (MAX_SIZE, MAX_SIZE)).astype(float) / pred.astype(float)
    shadow_map = cv2.resize(shadow_map, (w, h))
    shadow_map[shadow_map == 0] = 0.00001
    return np.clip(img.astype(float) / shadow_map, 0, 255).astype(np.uint8)


def deblurring(model, img, device, dtype):
    padded, orig_h, orig_w = tiles.stride_integral(img, 8)
    prompt = prompts.deblur_prompt(padded)
    out = _postprocess(torch.clamp(model(_to_tensor(np.concatenate((padded, prompt), -1), device, dtype)), 0, 1))
    return out[:orig_h, :orig_w]


def deshadowing(model, img, device, dtype):
    return _generic_restore(model, img, prompts.deshadow_prompt(img), device, dtype)


def appearance(model, img, device, dtype):
    return _generic_restore(model, img, prompts.appearance_prompt(img), device, dtype)


def binarization(model, img, device, dtype):
    padded, orig_h, orig_w = tiles.stride_integral(img, 8)
    prompt = prompts.binarization_prompt(padded)
    h, w = padded.shape[:2]

    logits = model(_to_tensor(np.concatenate((padded, prompt), -1), device, dtype))
    pred = torch.max(torch.softmax(logits[:, :2], 1), 1)[1]
    pred = (pred[0].cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(pred, (w, h))[:orig_h, :orig_w]


def _run(model, img, task, device, dtype, mbd):
    if task == "dewarping":
        return dewarping(model, img, mbd, device, dtype)
    if task == "deshadowing":
        return deshadowing(model, img, device, dtype)
    if task == "appearance":
        return appearance(model, img, device, dtype)
    if task == "deblurring":
        return deblurring(model, img, device, dtype)
    if task == "binarization":
        return binarization(model, img, device, dtype)
    raise ValueError("unsupported task: %s" % task)


def restore(model, img, task, device, dtype, mbd=None):
    """Dispatch a single restoration task, then release the intermediates.

    The caching allocator keeps freed blocks reserved, so the explicit
    empty_cache() matters when several of these run back to back in one graph.
    """
    result = _run(model, img, task, device, dtype, mbd)
    _empty_backend_cache()
    return result
