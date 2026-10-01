/**
 * Control-point editor for the DDC Edit Points node.
 *
 * Prediction happens in DDC Predict Points; this extension attaches to the
 * editing stage and opens a floating panel with a real DOM canvas, following
 * the pattern used by ComfyUI-OpenPose-Studio. An inline canvas *widget* was
 * tried first and never rendered, so the editor is a plain <canvas> in a
 * floating panel: it owns its own hit-testing, and cannot be starved by
 * LiteGraph's widget layout.
 *
 * Drags are written into the node's `points_json` widget, which is what the
 * backend reads on the next run. That keeps corrections in the saved workflow
 * and lets `DDC Rectify` reproduce an edited warp unattended.
 *
 * The last run's payload and preview ref are also cached on the node
 * (`_ddcPayload`, `_ddcImageRef`), so opening the panel after the workflow has
 * already finished still shows the grid instead of an empty canvas.
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "DDCEditPoints";
const PANEL_W = 900;
const PANEL_H = 640;
const HIT_RADIUS = 14;   // screen px, so grabbing stays easy when zoomed out
const HANDLE_R = 5;     // screen px, constant regardless of zoom
const MIN_ZOOM = 1;
const MAX_ZOOM = 12;

function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
}

function getPointsWidget(node) {
    return node.widgets?.find((w) => w.name === "points_json");
}

function parsePoints(text) {
    if (!text) return null;
    try {
        const data = JSON.parse(text);
        if (!Array.isArray(data?.points)) return null;
        return data;
    } catch {
        return null;
    }
}

function loadImage(ref) {
    if (!ref?.filename) return Promise.resolve(null);
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = api.apiURL(
            `/view?filename=${encodeURIComponent(ref.filename)}` +
            `&subfolder=${encodeURIComponent(ref.subfolder || "")}` +
            `&type=${encodeURIComponent(ref.type || "temp")}`
        );
    });
}

/**
 * Pixel -> canvas transform for a payload whose points are stored in the
 * `w` x `h` space recorded alongside them (the 960x1024 canvas the original
 * marks up). Keeping the payload's own size means the editor is independent of
 * how the preview image happens to be scaled into the panel.
 */
function pointToCanvas(data, rect, p) {
    const w = data.w || rect.w;
    const h = data.h || rect.h;
    return [rect.x + (p[0] / w) * rect.w, rect.y + (p[1] / h) * rect.h];
}

/** Nearest grid point to a canvas position, or null when nothing is in range. */
function nearestPoint(data, rect, mx, my) {
    const rows = data.grid?.[0] ?? data.points.length;
    const cols = data.grid?.[1] ?? data.points[0].length;
    let best = null;
    let bestDist = HIT_RADIUS;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const p = data.points[r][c];
            if (!p) continue;
            const [px, py] = pointToCanvas(data, rect, p);
            const dx = px - mx;
            const dy = py - my;
            const dist = Math.hypot(dx, dy);
            if (dist < bestDist) {
                bestDist = dist;
                best = { r, c };
            }
        }
    }
    return best;
}

class ControlPointEditor {
    constructor(node) {
        this.node = node;
        this.data = parsePoints(getPointsWidget(node)?.value)
            ?? parsePoints(node._ddcPayload);
        this.image = null;
        this.rect = null;
        this.drag = null;
        this.hover = null;
        this.pan = null;
        this.spaceDown = false;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.build();
        this.refresh();
    }

    build() {
        this.root = document.createElement("div");
        this.root.style.cssText = `
            position: fixed; inset: 0; z-index: 1000;
            display: flex; align-items: center; justify-content: center;
            background: rgba(0,0,0,0.6);`;
        this.root.addEventListener("mousedown", (e) => {
            if (e.target === this.root) this.close();
        });

        this.panel = document.createElement("div");
        this.panel.style.cssText = `
            width: ${PANEL_W}px; height: ${PANEL_H}px; max-width: 95vw;
            display: flex; flex-direction: column; gap: 8px; padding: 12px;
            background: var(--comfy-menu-bg, #202020); color: var(--input-text, #ddd);
            border: 1px solid var(--border-color, #444); border-radius: 6px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);`;

        const bar = document.createElement("div");
        bar.style.cssText = "display:flex; align-items:center; gap:10px;";

        this.status = document.createElement("span");
        this.status.style.cssText = "flex:1; font:12px monospace; opacity:0.85;";
        this.status.textContent = "no control points yet - run the node first";

        this.resetBtn = document.createElement("button");
        this.resetBtn.textContent = "Reset view";
        this.resetBtn.style.cssText = "cursor:pointer; padding:4px 10px;";
        this.resetBtn.onclick = () => this.resetView();

        const reset = document.createElement("button");
        reset.textContent = "Close";
        reset.style.cssText = "cursor:pointer; padding:4px 10px;";
        reset.onclick = () => this.close();

        bar.append(this.status, this.resetBtn, reset);

        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `
            flex:1; width:100%; background:#111; border-radius:4px; cursor:crosshair;`;

        this.hint = document.createElement("div");
        this.hint.style.cssText = "font:11px monospace; opacity:0.7;";
        this.hint.textContent =
            "drag a handle   |   zoom: ctrl+scroll   |   pan: scroll, space+drag, middle or right drag   |   Esc to exit";

        this.panel.append(bar, this.canvas, this.hint);
        this.root.append(this.panel);
        document.body.append(this.root);

        this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
        this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
        this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
        this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
        // Bound to the instance so close() can actually detach them; anonymous
        // listeners leaked and stacked up on every open.
        this.onUp = (e) => this._onUp(e);
        this.onKey = (e) => {
            if (e.key === "Escape") { this.close(); return; }
            if (e.code === "Space") {
                // Space is the keyboard equivalent of the middle button, for
                // trackpads and mice where the middle button is awkward.
                this.spaceDown = e.type === "keydown";
                if (this.spaceDown) e.preventDefault();
                this.root.style.cursor = this.spaceDown ? "grab" : "";
            }
        };
        this.onWindowResize = () => this.resize();
        window.addEventListener("pointerup", this.onUp);
        window.addEventListener("keydown", this.onKey);
        window.addEventListener("resize", this.onWindowResize);
    }

    setImageRef(ref) {
        this.imageRef = ref ?? null;
        if (ref) this.image = null;   // force a reload on the next refresh
    }

    async refresh() {
        // The widget is the source of truth, because a drag writes to it and
        // the backend re-reads it on the next run. The node-level cache is the
        // fallback for a panel opened after the run, or after the widget was
        // cleared, so reopening never lands on an empty canvas.
        const data =
            parsePoints(getPointsWidget(this.node)?.value) ??
            parsePoints(this.node._ddcPayload);
        if (data) {
            // The backend re-seeds the widget on every run, so a fresh payload
            // means a new prediction and a fresh baseline. Without this the
            // "edited N" count would compare against a stale grid.
            if (!this._lastPayload || this._lastPayload !== JSON.stringify(data)) {
                this.original = data.points.map((row) => row.map((p) => [p[0], p[1]]));
                this._lastPayload = JSON.stringify(data);
            }
            this.data = data;
        }

        // Prefer the backend preview; fall back to a directly attached image
        // widget when the node is fed from LoadImage without a link.
        this.imageRef = this.imageRef ?? this.node._ddcImageRef;
        if (!this.imageRef) {
            const ref = this.node.widgets?.find((w) => w.name === "image")?.value;
            if (ref && typeof ref === "object" && ref.filename) this.imageRef = ref;
        }
        if (this.imageRef) {
            this.image = await loadImage(this.imageRef);
        }
        this.resize();
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        this.ctx = this.canvas.getContext("2d");
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.cssW = rect.width;
        this.cssH = rect.height;
        this.draw();
    }

    localPos(event) {
        const rect = this.canvas.getBoundingClientRect();
        return [event.clientX - rect.left, event.clientY - rect.top];
    }

    onDown(event) {
        if (!this.data || !this.rect) return;
        const [mx, my] = this.localPos(event);

        // Panning: middle button, right button, or space held. Without this the
        // zoomed-in view is a dead end -- there is no way to reach the parts of
        // the page that are off-panel.
        if (event.button === 1 || event.button === 2 || this.spaceDown) {
            this.pan = { x: mx, y: my, panX: this.panX, panY: this.panY };
            this.canvas.style.cursor = "grabbing";
            this.canvas.setPointerCapture?.(event.pointerId);
            event.preventDefault();
            return;
        }
        if (event.button !== 0) return;

        const hit = nearestPoint(this.data, this.rect, mx, my);
        if (!hit) return;
        this.drag = hit;
        this.canvas.setPointerCapture?.(event.pointerId);
        this.draw();
    }

    onMove(event) {
        if (!this.data || !this.rect) return;
        const [mx, my] = this.localPos(event);

        if (this.pan) {
            this.panX = this.pan.panX + (mx - this.pan.x);
            this.panY = this.pan.panY + (my - this.pan.y);
            this.draw();
            return;
        }

        if (this.drag) {
            // Store in the payload's own pixel space, not in normalized units,
            // so the value handed to the backend is the coordinate it expects.
            const w = this.data.w || this.rect.w;
            const h = this.data.h || this.rect.h;
            const x = clamp((mx - this.rect.x) / this.rect.w, 0, 1) * w;
            const y = clamp((my - this.rect.y) / this.rect.h, 0, 1) * h;
            this.data.points[this.drag.r][this.drag.c] = [
                Number(x.toFixed(4)),
                Number(y.toFixed(4)),
            ];
            this.data.dirty = this.data.dirty || [];
            (this.data.dirty[this.drag.r] = this.data.dirty[this.drag.r] || [])[this.drag.c] = true;
            this.commit();
            return;
        }

        const hover = nearestPoint(this.data, this.rect, mx, my);
        const changed = (!!hover) !== (!!this.hover);
        this.hover = hover;
        if (changed) this.draw();
    }

    _onUp() {
        if (this.pan) {
            this.pan = null;
            this.canvas.style.cursor = this.spaceDown ? "grab" : "crosshair";
            return;
        }
        if (!this.drag) return;
        this.drag = null;
        this.draw();
    }

    /** Ctrl+wheel zooms about the cursor; plain wheel scrolls the panel. */
    onWheel(event) {
        // The panel does not scroll, so plain wheel pans and ctrl+wheel zooms.
        if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            this.panX -= event.deltaX;
            this.panY -= event.deltaY;
            this.draw();
            return;
        }
        event.preventDefault();

        const [mx, my] = this.localPos(event);
        const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
        const next = clamp(this.zoom * factor, MIN_ZOOM, MAX_ZOOM);
        if (next === this.zoom) return;

        // Keep the point under the cursor fixed while scaling.
        const cx = (mx - this.cssW / 2 - this.panX) / this.zoom;
        const cy = (my - this.cssH / 2 - this.panY) / this.zoom;
        this.panX = mx - this.cssW / 2 - cx * next;
        this.panY = my - this.cssH / 2 - cy * next;
        this.zoom = next;
        this.draw();
    }

    resetView() {
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.draw();
    }

    /** How many handles the user has actually moved from the predicted grid. */
    countMoved() {
        if (!this.original) return 0;
        let n = 0;
        for (let r = 0; r < this.data.points.length; r++) {
            for (let c = 0; c < this.data.points[r].length; c++) {
                const a = this.data.points[r][c], b = this.original[r]?.[c];
                if (b && Math.hypot(a[0] - b[0], a[1] - b[1]) > 0.5) n++;
            }
        }
        return n;
    }

    /** Persist the edited grid into the widget the backend reads. */
    commit() {
        const text = JSON.stringify(this.data);
        const widget = getPointsWidget(this.node);
        if (widget) widget.value = text;
        // Keep the node-level cache in step, so closing the panel and opening
        // it again shows the edit rather than reverting to the predicted grid.
        this.node._ddcPayload = text;
        this.draw();
    }

    draw() {
        const ctx = this.ctx;
        if (!ctx) return;
        ctx.clearRect(0, 0, this.cssW, this.cssH);

        if (!this.data) {
            ctx.fillStyle = "rgba(255,255,255,0.6)";
            ctx.font = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(
                "no control points yet - run the workflow, then reopen",
                this.cssW / 2,
                this.cssH / 2
            );
            this.status.textContent = "no control points yet";
            return;
        }

        // Fit the image, then apply the viewer's zoom and pan.
        let rect = { x: 0, y: 0, w: this.cssW, h: this.cssH };
        if (this.image && this.image.width) {
            const scale = Math.min(this.cssW / this.image.width, this.cssH / this.image.height);
            const iw = this.image.width * scale;
            const ih = this.image.height * scale;
            rect = { x: (this.cssW - iw) / 2, y: (this.cssH - ih) / 2, w: iw, h: ih };
        }

        const cx = this.cssW / 2 + this.panX;
        const cy = this.cssH / 2 + this.panY;
        const w = rect.w * this.zoom;
        const h = rect.h * this.zoom;
        rect = { x: cx - w / 2, y: cy - h / 2, w, h };
        this.rect = rect;

        if (this.image && this.image.width) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, this.cssW, this.cssH);
            ctx.clip();
            ctx.drawImage(this.image, rect.x, rect.y, rect.w, rect.h);
            ctx.restore();
        }

        const rows = this.data.grid?.[0] ?? this.data.points.length;
        const cols = this.data.grid?.[1] ?? this.data.points[0].length;
        const P = (r, c) => pointToCanvas(this.data, rect, this.data.points[r][c]);

        // Trace the mesh through the points themselves. Straight guide lines
        // would be wrong here: the whole point of the grid is that it bows with
        // the page, and the outline you see is what the warp is fitted to.
        ctx.strokeStyle = "rgba(94,203,255,0.40)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let c = 0; c < cols; c++) {
            for (let r = 0; r < rows; r++) {
                const [x, y] = P(r, c);
                r === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
        }
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const [x, y] = P(r, c);
                c === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // The outer ring is the page outline; draw it heavier so the boundary
        // the network actually fitted is readable at a glance.
        ctx.strokeStyle = "rgba(255,215,94,0.95)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        let [x, y] = P(0, 0);
        ctx.moveTo(x, y);
        for (let c = 1; c < cols; c++) [x, y] = P(0, c), ctx.lineTo(x, y);
        for (let r = 1; r < rows; r++) [x, y] = P(r, cols - 1), ctx.lineTo(x, y);
        for (let c = cols - 2; c >= 0; c--) [x, y] = P(rows - 1, c), ctx.lineTo(x, y);
        for (let r = rows - 2; r >= 0; r--) [x, y] = P(r, 0), ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();

        // Handles keep a constant on-screen size. Scaling them with the mesh
        // made them sub-pixel when zoomed out and impossible to hit, and huge
        // blobs when zoomed in that swallowed their neighbours.
        const hr = HANDLE_R;
        ctx.lineWidth = 1.5;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const [px, py] = P(r, c);
                // Dark rim first so a handle stays readable over a white page.
                ctx.beginPath();
                ctx.arc(px, py, hr + 1.5, 0, Math.PI * 2);
                ctx.strokeStyle = "rgba(0,0,0,0.55)";
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(px, py, hr, 0, Math.PI * 2);
                ctx.fillStyle = this.data.dirty?.[r]?.[c] ? "#ffd75e" : "#5ecbff";
                ctx.fill();
            }
        }

        const active = this.drag || this.hover;
        if (active) {
            const p = this.data.points[active.r][active.c];
            const [ax, ay] = pointToCanvas(this.data, rect, p);
            ctx.fillStyle = "#ffd75e";
            ctx.beginPath();
            ctx.arc(ax, ay, 6, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "rgba(0,0,0,0.75)";
            ctx.fillRect(rect.x, rect.y, 230, 18);
            ctx.fillStyle = "#fff";
            ctx.font = "11px monospace";
            ctx.textAlign = "left";
            ctx.fillText(
                `[${active.r},${active.c}]  ${p[0].toFixed(4)}, ${p[1].toFixed(4)}`,
                rect.x + 5,
                rect.y + 13
            );
            this.status.textContent =
                `point [${active.r},${active.c}] = ${p[0].toFixed(4)}, ${p[1].toFixed(4)}`;
        } else {
            const moved = this.countMoved();
            this.status.textContent =
                `${rows}x${cols} handles   (${this.data.w}x${this.data.h})   ` +
                `zoom ${this.zoom.toFixed(1)}x   edited ${moved}` +
                (moved ? "   -> run the node to apply" : "");
        }
    }

    close() {
        window.removeEventListener("pointerup", this.onUp);
        window.removeEventListener("keydown", this.onKey);
        window.removeEventListener("resize", this.onWindowResize);
        this.root?.remove();
        // Clear the handle so the button reopens instead of calling close()
        // on an already-dismissed editor.
        if (this.node) this.node._ddcEditor = null;
        this.closed = true;
    }
}

app.registerExtension({
    name: "DDC.ControlPoints",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const node = this;
            const result = onNodeCreated?.apply(this, arguments);

            // The widget carries the grid, so it must stay visible: hiding it
            // made the payload unreadable and left the editor with nothing to
            // draw. It is collapsed rather than removed, because 961 points is
            // a lot of JSON to look at.
            const pointsWidget = getPointsWidget(node);
            if (pointsWidget) {
                pointsWidget.hidden = false;
                delete pointsWidget.options?.hidden;
            }

            // A button is the reliable way in: an inline canvas widget is drawn
            // by LiteGraph and did not show up in this ComfyUI version.
            const button = node.addWidget("button", "edit control points", "", () => {
                // Toggle, but only against a live editor: close() nulls the
                // handle, so a dismissed editor reopens instead of no-oping.
                if (node._ddcEditor) {
                    node._ddcEditor.close();
                    return;
                }
                node._ddcEditor = new ControlPointEditor(node);
                node._ddcEditor.refresh();
            });
            button.serialize = false;

            return result;
        };

        // Wrapped once per node *type*, not per node instance -- doing it inside
        // onNodeCreated would stack a new wrapper for every node on the canvas.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);

            // The backend ships the grid on the `ui` channel. Which key it lands
            // under depends on the ComfyUI version (node type name vs. the
            // "executed" envelope), so look for `points_json` at either level
            // rather than betting on one shape.
            const payload =
                message?.points_json?.[0] ??
                message?.output?.points_json?.[0] ??
                message?.ddc_edit_points?.points_json?.[0];
            if (payload) {
                const widget = getPointsWidget(this);
                if (widget && widget.value !== payload) widget.value = payload;
                this._ddcPayload = payload;
            }

            // The image travels on a socket, so the editor can only see it via
            // the preview the backend writes to the temp folder. Cache it on the
            // node: a run that happened while the panel was closed still has to
            // be there when the panel is opened again.
            const ref =
                message?.images?.[0] ??
                message?.output?.images?.[0] ??
                message?.ddc_edit_points?.images?.[0];
            if (ref) this._ddcImageRef = ref;
            this._ddcEditor?.setImageRef(this._ddcImageRef);
            this._ddcEditor?.refresh();

            return result;
        };
    },
});
