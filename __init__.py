"""ComfyUI nodes for DocRes restoration and control-point dewarping.

Forward-only port: no training code, no optimizer state, no dataset loaders.

`nodes.py` holds the DocRes tasks; `ddc_nodes.py` holds the control-point
dewarping trio (predict -> edit -> rectify).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serves web/ddc_control_points.js, which draws the draggable control-point
# canvas on the DDC Edit Points node.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
