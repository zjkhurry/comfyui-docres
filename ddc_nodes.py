"""ComfyUI nodes for control-point document dewarping.

The flow is deliberately split into three stages, so that a human can sit
between the network and the warp:

    DDC Predict Points  --raw points-->  DDC Edit Points  --edited points-->  DDC Rectify

Stage 1 runs the network and nothing else. Stage 2 never runs the network: it
only displays the points it is handed and lets you drag them. Stage 3 consumes
the grid, automatic or hand-tuned, and warps the page flat.

Keeping prediction and editing apart is what makes the loop work. Points are
the expensive part to obtain and the cheap part to fix, so a correction costs
one cheap node instead of a full forward pass.
"""

import json
import os

import cv2
import numpy as np
import torch

import ddc_pipeline


def _save_preview(image, prefix):
    """Write an RGB array to a temp PNG and return its {filename, ...} ref.

    The frontend cannot reach an IMAGE socket, so the editor panel gets the
    source image through the `ui` channel, exactly like SaveImage does.
    """
    import cv2

    try:
        import folder_paths
        directory = folder_paths.get_temp_directory()
    except ImportError:
        # Outside ComfyUI (tests, CLI) there is no temp folder to write to.
        import tempfile
        directory = tempfile.gettempdir()

    os.makedirs(directory, exist_ok=True)
    name = "%s_%s.png" % (prefix, os.urandom(4).hex())
    cv2.imwrite(os.path.join(directory, name), image[:, :, ::-1])
    return {"filename": name, "subfolder": "", "type": "temp"}

def _preview_ui(image_rgb, prefix="ddc_points"):
    """`ui` payload for a node whose only purpose is to show a picture.

    ComfyUI renders an `images` entry as the node's own preview, which is how a
    node without a PreviewImage downstream still shows something.
    """
    ui = {}
    ref = _save_preview(image_rgb, prefix)
    if ref:
        ui["images"] = [ref]
    return ui


DDCPointsType = "DDC_RAW_POINTS"
DDCEditedType = "DDC_EDITED_POINTS"


def _numpy_to_comfy(image):
    """HxWx3 uint8 RGB -> 1xHxWx3 float32 tensor in [0, 1]."""
    return torch.from_numpy(image.astype(np.float32)[None, ...] / 255.0)


def _to_rgb_bgr(image):
    """ComfyUI 1xHxWx3 float tensor -> (HxWx3 uint8 RGB, HxWx3 uint8 BGR)."""
    rgb = (np.clip(image[0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    return rgb, np.ascontiguousarray(rgb[:, :, ::-1])


class DDCPredictPoints:
    """Run the control-point network and emit the raw grid. No editing here."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = (DDCPointsType, "IMAGE", "IMAGE")
    RETURN_NAMES = ("points", "preview", "image")
    FUNCTION = "predict"
    CATEGORY = "image/DocRes"
    DESCRIPTION = (
        f"Predict the control-point grid for a distorted page and reduce it to "
        f"a {ddc_pipeline.EDIT_GRID}x{ddc_pipeline.EDIT_GRID} set of draggable "
        f"handles. Feed the result to DDC Edit Points."
    )

    def predict(self, image, device):
        rgb, bgr = _to_rgb_bgr(image)

        resolved = ddc_pipeline.resolve_device(device)
        model = ddc_pipeline.load_model(resolved)
        points, segment = ddc_pipeline.predict(model, bgr, resolved)

        # The points are pixels of the 992x992 prediction canvas, but the
        # editor draws on the 960x1024 canvas the original marks up, so hand
        # the editor that frame and keep the model frame for the warp.
        canvas = ddc_pipeline.canvas_for(bgr)
        height, width = canvas.shape[:2]
        canvas_points = ddc_pipeline.to_canvas(points, (width, height))
        canvas_rgb = np.ascontiguousarray(canvas[:, :, ::-1])

        # 961 handles cannot be grabbed one at a time, so the editor works on a
        # coarse sample. The full grid rides along untouched and is what the
        # warp is fitted to once the coarse edits are diffused back onto it.
        coarse = ddc_pipeline.subsample_grid(canvas_points, ddc_pipeline.EDIT_GRID)

        preview = ddc_pipeline.draw_points(canvas_rgb, coarse)
        return {"ui": _preview_ui(preview), "result": ({
            "points": canvas_points,
            "coarse": coarse,
            "segment": segment,
            "width": width,
            "height": height,
            "json": ddc_pipeline.encode_points(coarse, width, height),
            "preview": canvas_rgb,
        }, _numpy_to_comfy(preview), _numpy_to_comfy(np.ascontiguousarray(rgb)))}


class DDCEditPoints:
    """Show incoming control points over the image and let them be dragged.

    This node never runs the network and produces no dewarped pixels. It draws
    the grid on the 960x1024 canvas the original marks up, lets you drag points
    on that canvas, and carries the grid forward with your edits applied.

    Two image outputs, because they serve different purposes: `preview` is the
    canvas with the points drawn on it, which is what you look at, and `image`
    is the same canvas unmarked, which is what `DDC Rectify` should warp. Wiring
    the marked one into the warp would bake the dots into the rectified page.

    The edited grid lives in the `points_json` widget, which means a saved
    workflow keeps your adjustments and a later run reproduces the warp without
    you touching the mouse again.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": (DDCPointsType,),
                "points_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (DDCEditedType, "IMAGE", "IMAGE")
    RETURN_NAMES = ("points", "preview", "image")
    FUNCTION = "edit"
    OUTPUT_NODE = True
    CATEGORY = "image/DocRes"
    DESCRIPTION = (
        "Interactive control-point editor. Drag points to correct them; the "
        "grid is carried forward with your edits applied."
    )

    def edit(self, points, points_json):
        width, height = points["width"], points["height"]
        # The editable grid is the coarse sample; `points` also carries the full
        # 31x31 prediction, which the warp still needs.
        fine = points["points"]
        base_coarse = points.get("coarse")
        if base_coarse is None:
            base_coarse = ddc_pipeline.subsample_grid(fine, ddc_pipeline.EDIT_GRID)
        grid, edited = base_coarse, False

        # The canvas writes drags into points_json, in canvas pixels. Prefer
        # those over whatever arrived on the wire, but only for the same size.
        if points_json.strip():
            try:
                data = json.loads(points_json)
                if data.get("w") == int(width) and data.get("h") == int(height):
                    grid, edited = ddc_pipeline.decode_points(points_json), True
            except (ValueError, KeyError, TypeError):
                pass

        payload = ddc_pipeline.encode_points(grid, width, height)

        # `preview` carries the dots so the points are visible in the graph
        # preview; `image` stays clean for the warp.
        canvas_rgb = points["preview"]
        preview = ddc_pipeline.draw_points(canvas_rgb, grid, edited=edited)

        # The Python node object cannot reach frontend widgets, so the edited
        # grid is handed over through the `ui` channel (the route SaveImage
        # uses), which is also what puts the marked image in the node preview.
        ui = {"points_json": [payload]}
        ref = _save_preview(preview, "ddc_points")
        if ref:
            ui["images"] = [ref]

        return {"ui": ui, "result": ({
            "points": grid,
            "fine": fine,
            "coarse_base": base_coarse,
            "segment": points.get("segment"),
            "width": width,
            "height": height,
            "edited": edited,
            "json": payload,
            # Carried so DDC Rectify can draw the grid it actually warped with.
            "preview": canvas_rgb,
        }, _numpy_to_comfy(preview), _numpy_to_comfy(canvas_rgb))}


class DDCRectify:
    """Warp the page flat using a control-point grid, edited or automatic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "points": (DDCEditedType,),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "preview")
    FUNCTION = "rectify"
    CATEGORY = "image/DocRes"
    DESCRIPTION = "Rectify a curved page using the control points from DDC Edit Points."

    def rectify(self, image, points):
        _, bgr = _to_rgb_bgr(image)
        width, height = points["width"], points["height"]

        # The editor only moved a handful of coarse handles. Fit the full 31x31
        # grid so it follows their distribution: a TPS displacement field that
        # matches the drags at the handles while the untouched model points act
        # as a weak prior. The result is a smooth mesh, not a kinked one.
        fine = points.get("fine")
        coarse = points["points"]
        if fine is not None and points.get("coarse_base") is not None:
            grid = ddc_pipeline.lift_edits(coarse, points["coarse_base"], fine)
        else:
            grid = coarse

        # The grid arrives in 960x1024 canvas pixels; the TPS wants the 992
        # prediction-canvas units the network emitted.
        model_points = ddc_pipeline.to_model(grid, (width, height))

        # Output size comes from the network's `segment` head, exactly as
        # utilsV4 computes `flat_shap`.
        flat = ddc_pipeline.rectify(model_points, bgr, segment=points.get("segment"))
        flat_rgb = np.ascontiguousarray(flat[:, :, ::-1])

        # `preview` is the input canvas with the grid that was actually used for
        # this warp drawn back on it, so the result can be checked against the
        # points that produced it. `image` is the rectified page.
        canvas_rgb = points.get("preview")
        if canvas_rgb is None:
            return {"ui": _preview_ui(flat_rgb), "result": (
                _numpy_to_comfy(flat_rgb), _numpy_to_comfy(flat_rgb),
            )}
        marked = ddc_pipeline.draw_points(
            canvas_rgb, grid, edited=points.get("edited", False)
        )
        return {"ui": _preview_ui(marked), "result": (
            _numpy_to_comfy(flat_rgb), _numpy_to_comfy(marked),
        )}


NODE_CLASS_MAPPINGS = {
    "DDCPredictPoints": DDCPredictPoints,
    "DDCEditPoints": DDCEditPoints,
    "DDCRectify": DDCRectify,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DDCPredictPoints": "DDC Predict Points",
    "DDCEditPoints": "DDC Edit Points",
    "DDCRectify": "DDC Rectify",
}
