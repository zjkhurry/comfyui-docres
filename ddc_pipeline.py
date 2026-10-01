"""Control-point document dewarping (Document-Dewarping-with-Control-Points).

Forward-only port of gwxie/Document-Dewarping-with-Control-Points. The network
predicts a 31x31 grid of control points plus a reference grid; a thin-plate
spline fitted to those points maps the curved page onto a flat rectangle.

The control points are the interesting part: they are plain numbers, so they
can be dragged by hand before the warp. `predict` returns them in pixel units
of the prediction canvas, the frontend edits that array, and `rectify` warps
with whatever it is given.

Conventions, verified against `Source/test.py` on ITIN.jpg (max diff 0.0000):

  * the network is fed the image resized once to 992x992 with raw 0-255
    floats. `dataloader.transform_im` only does ``transpose(2,0,1)`` +
    ``float()``; there is no /255 anywhere in the original pipeline.
  * the network emits ``(B, 2, 31, 31)``; transposing to ``(31, 31, 2)`` gives
    points indexed ``[row, col]`` with component 0 = x and component 1 = y.
  * those values are absolute pixels of the 992x992 prediction canvas, which
    `utilsV4` rescales by ``perturbed_img.shape[::-1] / 992`` when it draws the
    overlay. A point is drawn at ``p / 992 * [960, 1024]``.
  * the TPS reference grid is flattened col-major, i.e.
    ``points.transpose(1, 0, 2).reshape(-1, 2)``, and mapped from [0, 1] to
    [-1, 1] after dividing by 992.
"""
import json
import os

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

import weights_fetch
from ddc.network import DilatedResnetForFlatByFiducialPointsS2, FiducialPoints
from ddc.tpsV2 import createThinPlateSplineShapeTransformer

# Inference only: no autograd graph on any path.
torch.set_grad_enabled(False)

INPUT_SIZE = 992                 # the network was trained at this resolution
GRID = 31                        # 31 x 31 = 961 control points

# The canvas the points are predicted on and edited against: utilsV4 loads the
# test image and resizes it to (960, 1024), i.e. 960 wide by 1024 tall. Its
# overlay is drawn on exactly that array, so this is the frame the points
# belong to and the frame the editor has to show.
CANVAS_SIZE = (960, 1024)        # (w, h)

# utilsV4 builds the TPS at this fixed size and then resamples the sampling
# grid up to the requested output shape, instead of solving at the final size.
TPS_BASE_SIZE = (320, 320)       # (h, w) of the TPS reference grid

# Building the network and the 961-point TPS are both expensive, so both are
# cached per process. The TPS base grid does not depend on the control points.
_MODEL_CACHE = {}
_TPS_CACHE = {}


def canvas_for(image_bgr):
    """The prediction canvas for an image, i.e. utilsV4's `perturbed_img`.

    Two different resizes are involved and mixing them up is the whole source
    of the "points are rotated / scaled wrong" confusion:

      * the *network* sees 992x992 (`dataloader.resize_im`);
      * the *overlay* and the *TPS source* are the image at 960x1024
        (`utilsV4.flatByfiducial_TPS`, only for scheme 'test'/'eval').

    A predicted point is therefore a pixel of the 992x992 canvas, and reaches
    the overlay through `to_canvas`.
    """
    return cv2.resize(image_bgr, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR)


def to_canvas(points_model, canvas_wh=CANVAS_SIZE):
    """Map 992-canvas pixels onto overlay pixels, as utilsV4 draws them.

    utilsV4 divides by 992 and multiplies by ``perturbed_img.shape[:2][::-1]``,
    i.e. (width, height) of the 960x1024 overlay.
    """
    width, height = canvas_wh
    return (np.asarray(points_model, dtype=np.float64) / INPUT_SIZE
            * np.array([width, height], dtype=np.float64))


def to_model(points_canvas, canvas_wh=CANVAS_SIZE):
    """Inverse of `to_canvas`: overlay pixels back to 992-canvas units.

    The editor stores points in overlay pixels, and the network and the TPS
    both expect to be handed back the 992-canvas frame they emitted.
    """
    width, height = canvas_wh
    return (np.asarray(points_canvas, dtype=np.float64)
            * (INPUT_SIZE / np.array([width, height], dtype=np.float64)))


def resolve_device(preferred="auto"):
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


# Control-point weights, fetched from Hugging Face on first use and cached in
# weights/. The upstream release is a training checkpoint whose optimizer state
# is twice the size of the model; the copy on the Hub is the stripped
# state_dict, so nothing here needs torch.load or the prefix fixup.
CHECKPOINT_NAME = "ddc_fiducial1024_v1.safetensors"


def _require_checkpoint():
    """Path to the control-point weights, downloading them if necessary."""
    local = os.path.join(weights_fetch.WEIGHT_DIR, CHECKPOINT_NAME)
    if os.path.isfile(local) and os.path.getsize(local) > 0:
        return local
    return weights_fetch.ensure(CHECKPOINT_NAME)


def load_model(device):
    """Build the network and load the released weights, caching per device."""
    path = _require_checkpoint()
    key = (path, str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model = FiducialPoints(
        n_classes=2,
        num_filter=32,
        architecture=DilatedResnetForFlatByFiducialPointsS2,
        BatchNorm="BN",
        in_channels=3,
    )
    state = load_file(path)
    # Tolerate a file converted from the original .pkl, which carries the
    # DataParallel prefix.
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.eval().to(device)

    _MODEL_CACHE[key] = model
    return model


def predict(model, image_bgr, device):
    """Run the network on one image, exactly as `test.py` does.

    The dataloader path is: `cv2.resize(im, (992, 992))`, then
    `transpose(2, 0, 1)` and `float()`. There is no /255 and no second resize;
    adding either one changes every predicted coordinate.

    Returns the (31, 31, 2) control grid in pixels of the 992x992 prediction
    canvas, and the raw (2,) `segment` head.
    """
    resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(
        np.ascontiguousarray(resized.transpose(2, 0, 1))[None].astype(np.float32)
    ).to(device)

    points, segment = model(tensor)
    points = points.cpu().numpy()[0].transpose(1, 2, 0)   # (31, 31, 2)
    # Left raw, exactly like utilsV4: `segment` scales into the output size.
    segment = segment.cpu().numpy()[0]                     # (2,)

    return points.astype(np.float32), segment.astype(np.float32)


def segment_output_hw(segment):
    """Output size the network asks for, per utilsV4's `flat_shap`.

    utilsV4 rounds and casts the segment head to int *before* using it, then
    scales it by the point-grid size:

        flat_shap = segment * [col_gap, row_gap] * [point_num_col, point_num_row]

    With the default gaps that is `segment * 31`, and `list(flat_shap)` is
    passed to the TPS as (height, width). Truncating first matters: rounding
    the product instead gives a page that is 482x904 rather than 930x992.
    """
    gaps = np.array([1, 1], dtype=np.float64)                    # col_gap, row_gap
    counts = np.array([GRID, GRID], dtype=np.float64)            # fiducial_point_num
    flat_shap = np.asarray(segment, dtype=np.float64).round().astype(np.int64)
    flat_shap = flat_shap * gaps * counts
    # `list(flat_shap)` is handed straight to F.interpolate, which reads it as
    # (height, width), so the segment head's order is already (H, W).
    return int(flat_shap[0]), int(flat_shap[1])                   # (h, w)


def rectify(points_norm, image_bgr, output_hw=None, segment=None):
    """Warp the page flat using thin-plate-spline interpolation.

    `points_norm` is the (31, 31, 2) control grid in pixels of the 992x992
    prediction canvas; it is what the frontend hands back after the user drags
    points around. `output_hw` is (height, width); when omitted it is taken
    from `segment` exactly the way the original does, so the page comes out at
    the size the network predicts.

    The TPS solve is a large float64 linear system. MPS has no float64, so this
    stage always runs on the CPU.
    """
    if output_hw is None:
        if segment is None:
            raise ValueError("rectify needs either output_hw or segment")
        output_hw = segment_output_hw(segment)
    points_norm = np.asarray(points_norm, dtype=np.float64)
    if points_norm.shape != (GRID, GRID, 2):
        raise ValueError(
            "control points must be %dx%dx2, got %s" % (GRID, GRID, (points_norm.shape,))
        )

    out_h, out_w = output_hw

    # Mirror utilsV4: the TPS is constructed once at the fixed base size, and
    # its sampling grid is interpolated to the requested output shape. The
    # base grid does not depend on the control points, so it is cached.
    tps = _TPS_CACHE.get("base")
    if tps is None:
        base_h, base_w = TPS_BASE_SIZE
        tps = createThinPlateSplineShapeTransformer(
            (base_h, base_w), fiducial_num=[GRID, GRID], device=torch.device("cpu")
        ).to("cpu")
        _TPS_CACHE["base"] = tps

    # utilsV4 warps the image it already resized to (960, 1024); do the same so
    # the control points and the pixels they index share one coordinate space.
    source = canvas_for(image_bgr)
    image = torch.from_numpy(
        np.ascontiguousarray(source.transpose(2, 0, 1))[None].astype(np.float32)
    ).double()

    # Normalize to [0, 1] and shift to [-1, 1], as utilsV4 does:
    #   fiducial_points / 992, then (x - 0.5) * 2
    normalized = points_norm / INPUT_SIZE
    # Col-major flatten matches the module's own reference-grid ordering.
    flat = normalized.transpose(1, 0, 2).reshape(-1, 2)[None]
    pts = ((flat - 0.5) * 2).astype(np.float64)

    with torch.no_grad():
        # The third argument is the output shape; passing None keeps the TPS
        # base size, so resample to the network's predicted page size instead.
        rectified = tps(image, torch.from_numpy(pts), [out_h, out_w])

    out = rectified[0].float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


# The coarse grid the user actually drags. 9x9 = 81 handles is enough to steer
# a page outline by hand and stays precise to hit, where 961 points are not.
EDIT_GRID = 9


def subsample_grid(points_grid, n=EDIT_GRID):
    """Evenly spaced samples of a (G, G, 2) grid, as (n, n, 2).

    Picks the sample positions nearest the even split of [0, G-1] so the coarse
    grid stays centred on the page rather than drifting to one edge.
    """
    grid = np.asarray(points_grid, dtype=np.float32)
    g = grid.shape[0]
    if g == n:
        return grid.copy()
    idx = np.linspace(0, g - 1, n).round().astype(int)
    return grid[np.ix_(idx, idx)].copy()


# How strongly the hand-placed handles steer the full 31x31 mesh. The handles are
# what the user actually asked for; the model's own 961 points are kept as a weak
# prior, so the regions nobody touched stay close to the prediction.
HANDLE_WEIGHT = 1.0
MESH_WEIGHT = 0.02
KERNEL_RIDGE = 1e-4         # damps the kernel block; it is what can oscillate


def _tps_basis(points, control):
    """TPS design rows [x, y, 1, u(|p-c_1|), ...] with u(r) = r^2 log r^2."""
    diff = points[:, None, :] - control[None, :, :]
    r2 = np.einsum("mkd,mkd->mk", diff, diff)
    kernel = np.where(r2 > 1e-12, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)
    return np.concatenate(
        [points, np.ones((points.shape[0], 1)), kernel], axis=1
    )


def lift_edits(coarse, coarse_grid, fine):
    """Make the full-resolution grid follow the distribution of the handles.

    The user moves 81 handles; the warp is fitted to 961 points. Both sets are fed
    into one least-squares fit of a thin-plate-spline *displacement* field,
    expressed on the model's own coordinates:

      * at each handle's model position the field must equal the drag the user
        made there (`HANDLE_WEIGHT`);
      * at every other of the 961 model positions the field must be zero
        (`MESH_WEIGHT`).

    Two consequences follow, and both are deliberate:

      * the handles steer rather than dictate. A handle asking for a deformation
        the surrounding mesh cannot absorb is pulled back, and the mesh carries
        the deformation smoothly -- so the 961 points end up distributed like the
        81 instead of kinking at them.
      * the handles do not have to be hit exactly. They are samples of the same
        field as everything else, so landing on a user coordinate to within a few
        pixels is not a failure; what matters is the shape.

    `coarse` and `coarse_grid` are both (n, n, 2); `fine` is (G, G, 2).
    """
    fine = np.asarray(fine, dtype=np.float32)
    coarse = np.asarray(coarse, dtype=np.float32)
    coarse_grid = np.asarray(coarse_grid, dtype=np.float32)
    g = fine.shape[0]
    n = coarse.shape[0]

    if np.abs(coarse - coarse_grid).max() <= 1e-4:
        return fine.copy()                       # nothing dragged: model wins

    src = np.linspace(0, g - 1, n).round().astype(int)
    flat = fine.reshape(-1, 2)

    # The kernel is |r|^2 log|r|^2, which is hopelessly ill-conditioned in raw
    # 960x1024 pixel units, so the solve runs on a unit box and is undone after.
    lo = flat.min(axis=0)
    span = np.maximum(flat.max(axis=0) - lo, 1e-6)
    mesh = (flat - lo) / span

    # Handle rows sit at the *model* position of each handle and target the drag.
    # Placing them at the new position would ask for zero displacement there and
    # cancel the entire edit.
    is_handle = np.zeros((g, g), dtype=bool)
    is_handle[np.ix_(src, src)] = True

    control = mesh.reshape(g, g, 2)[np.ix_(src, src)].reshape(-1, 2)
    handle_disp = (coarse - coarse_grid).reshape(-1, 2) / span
    prior_pos = mesh.reshape(g, g, 2)[~is_handle]     # points nobody moved

    rows = np.concatenate([control, prior_pos], axis=0)
    values = np.concatenate([handle_disp, np.zeros_like(prior_pos)], axis=0)
    weights = np.concatenate([
        np.full(control.shape[0], HANDLE_WEIGHT),
        np.full(prior_pos.shape[0], MESH_WEIGHT),
    ])

    design = _tps_basis(rows, control)
    ridge = np.concatenate([
        np.zeros(4), np.full(design.shape[1] - 4, KERNEL_RIDGE)
    ])
    weighted = design * weights[:, None]
    lhs = weighted.T @ design + np.diag(ridge)
    rhs = weighted.T @ values

    # Evaluate the fitted field at all 961 points. Both axes share one design
    # matrix; only the right-hand side differs.
    evaluate = _tps_basis(mesh, control)
    disp = np.empty_like(mesh)
    for axis in (0, 1):
        disp[:, axis] = evaluate @ np.linalg.solve(lhs, rhs[:, axis])

    out = (mesh + disp) * span + lo
    return out.reshape(fine.shape).astype(np.float32)


def encode_points(points, width, height):
    """Serialize a control-point grid into the JSON payload the node stores.

    The grid is stored in canvas pixels together with the size it belongs to,
    so a saved workflow carries both the numbers and the frame they mean. The
    grid dimensions are recorded rather than assumed: the editor works on a
    coarse sample, not the full 31x31.
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError("control points must be (n, n, 2), got %s" % (arr.shape,))
    n = arr.shape[0]
    return json.dumps({
        "v": 1,
        "w": int(width),
        "h": int(height),
        "grid": [n, n],
        "points": arr.round(4).tolist(),
    }, separators=(",", ":"))


def decode_points(payload):
    """Inverse of `encode_points`; accepts a JSON string or an already-parsed dict."""
    if payload is None or payload == "":
        return None
    data = json.loads(payload) if isinstance(payload, str) else payload

    points = np.asarray(data["points"], dtype=np.float32)
    if points.ndim != 3 or points.shape[2] != 2 or points.shape[0] != points.shape[1]:
        raise ValueError("control points must be (n, n, 2), got %s" % (points.shape,))

    grid = data.get("grid")
    if grid is not None and list(grid) != [points.shape[0], points.shape[1]]:
        raise ValueError(
            "grid %s does not match the point array %s" % (grid, points.shape)
        )
    return points


def draw_points(image_rgb, points_canvas, edited=False):
    """Draw the control-point grid over an RGB canvas, as utilsV4 does.

    `utilsV4.SaveFlatImage.location_mark` draws one filled radius-3 circle per
    point and nothing else, so at 31x31 the result reads as a dotted mesh of the
    page outline. Reproduced here so the node preview is the same picture as
    the original's `mark_*.jpg`.

    `points_canvas` is in the pixel frame of `image_rgb`, i.e. what `to_canvas`
    produced, so the preview lands exactly where the editor will let you drag.

    The canvas widget in the frontend is the interactive surface; this is the
    node's own image output, so the grid is visible in the graph preview and in
    any downstream preview consumer.
    """
    import cv2

    canvas = image_rgb.copy()
    color = (0, 255, 255) if edited else (255, 0, 0)   # RGB: cyan once edited
    pts = np.asarray(points_canvas, dtype=np.float64).reshape(-1, 2)
    for x, y in pts.astype(np.int64):
        cv2.circle(canvas, (int(x), int(y)), 3, color, -1)
    return canvas
