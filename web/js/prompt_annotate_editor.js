/**
 * File:   prompt_annotate_editor.js
 * Brief:  Rich highlighted input for Prompt Annotate and
 *         mode-aware widget states for Annotation Segment Edit.
 * Author: Mistress-Lukutar
 * Date:   2026-09-01
 * Version: v0.11.1
 *
 * The node's own prompt field is replaced with a "rich input": a
 * transparent-text textarea stacked over a colored backdrop div with
 * identical metrics, so the markup is highlighted live while typing
 * (native caret, selection, undo and IME keep working). The built-in
 * text widget is only hidden (widget.hidden = true) — it stays in
 * node.widgets, so its value keeps serializing into the workflow.
 *
 * Sizing: the in-node input is a FILL widget — the node layout owns its
 * height (it stretches with the node; the minimum is declared to the
 * layout engine via --comfy-widget-min-height) and typing past the
 * available height grows the node — but only up to MAX_FIELD_HEIGHT.
 * Past the cap the field stops growing and scrolls instead, and the
 * wheel scrolls the field while it can, zooming the canvas once the
 * field hits the top/bottom edge (or has no overflow at all).
 *
 * Alignment rules that keep the caret glued to the highlight:
 *  - the backdrop carries NO horizontal padding/border widths that the
 *    textarea does not have (span padding used to drift the caret 4px
 *    per tag);
 *  - the backdrop is sized to the textarea's clientWidth/clientHeight
 *    (excluding the scrollbar) and re-synced on resize;
 *  - both share font, line-height, padding, wrap and box-sizing.
 *
 * Input follows the canvas conventions: keydowns keep bubbling so
 * ComfyUI's global textarea-aware shortcuts work in the field
 * (Ctrl+Enter queues, Ctrl+Up/Down adjusts attention weight), and a
 * wheel the field cannot consume by scrolling its own overflow is
 * re-dispatched on the graph canvas (DOM widgets otherwise swallow
 * wheels — canvas zoom is bound to the canvas element itself).
 *
 * Served at /extensions/<pack>/js/prompt_annotate_editor.js, hence the
 * THREE parent hops to reach /scripts/app.js (two hops would resolve
 * inside /extensions/ and 404, killing the whole module).
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes] prompt annotate editor extension loaded");

const NODE_NAME = "PromptAnnotate";
const TEXT_WIDGET = "text";

/**
 * Auto-growth cap for the in-node field, in px (~25 lines at the
 * 11px/1.45 field font). Typing grows the node only until the field
 * reaches this height; past it the field scrolls instead of the node
 * stretching indefinitely. A manually enlarged node still fills (the
 * fill behavior is untouched — only auto-growth is capped).
 */
const MAX_FIELD_HEIGHT = 400;

/** Matches |labels: text| tags the same way core/prompt_annotator.py does. */
const TAG_RE = /\|([^:|\n]+):([^|\n]*)\|/g;

/* ------------------------------------------------------------------ */
/* Pastel palette (dark theme): desaturated hues, light text, subtle   */
/* tinted backgrounds.                                                 */
/* ------------------------------------------------------------------ */

/** Stable hue (0-359) per label for chips and preview highlighting. */
function labelHue(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  }
  return hash % 360;
}

const pastel = {
  bg: (hue) => `hsla(${hue}, 38%, 60%, 0.16)`,
  border: (hue) => `hsla(${hue}, 45%, 62%, 0.6)`,
  text: (hue) => `hsl(${hue}, 42%, 80%)`,
};

/* ------------------------------------------------------------------ */
/* Shared markup helpers                                               */
/* ------------------------------------------------------------------ */

/**
 * Render the RAW markup into a container: pipes and "label:" dimmed,
 * span text highlighted per label, unmarked text plain. The special
 * characters stay visible, just quiet.
 */
function renderRawMarkup(container, text) {
  container.replaceChildren();
  const addDim = (chunk) => {
    if (chunk) {
      const dim = document.createElement("span");
      dim.className = "lk-dim";
      dim.textContent = chunk;
      container.appendChild(dim);
    }
  };
  let cursor = 0;
  TAG_RE.lastIndex = 0;
  let match;
  while ((match = TAG_RE.exec(text)) !== null) {
    if (match.index > cursor) {
      container.appendChild(
        document.createTextNode(text.slice(cursor, match.index)),
      );
    }
    addDim(`|${match[1]}:`);
    const hue = labelHue(match[1].split(",")[0].trim());
    const span = document.createElement("span");
    span.className = "lk-span";
    span.style.background = pastel.bg(hue);
    span.style.borderBottom = `1.5px solid ${pastel.border(hue)}`;
    span.style.color = pastel.text(hue);
    span.textContent = match[2];
    container.appendChild(span);
    addDim("|");
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    container.appendChild(document.createTextNode(text.slice(cursor)));
  }
}

/* ------------------------------------------------------------------ */
/* Rich input: transparent-text textarea over a colored backdrop       */
/* ------------------------------------------------------------------ */

/**
 * Build a rich input element pair. The textarea stays a plain STRING
 * editor (native caret/undo/IME); the backdrop renders the highlighted
 * markup behind the invisible glyphs.
 *
 * @param {object} options
 * @param {string} options.value   Initial markup text.
 * @param {string} options.className Extra class for the wrapper
 *   (e.g. "lk-node-rich" for the compact in-node variant).
 * @param {Function} options.onInput Called with the new text on input.
 * @param {Function} options.onOverflow Called with the pixel shortfall
 *   when the content no longer fits the field, so the host can grow the
 *   node (the host decides how much — it caps the growth so the field
 *   scrolls past MAX_FIELD_HEIGHT).
 */
function createRichInput({
  value = "",
  className = "",
  onInput = null,
  onOverflow = null,
} = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `lk-rich${className ? " " + className : ""}`;

  const backdrop = document.createElement("div");
  backdrop.className = "lk-backdrop";
  backdrop.setAttribute("aria-hidden", "true");

  const textarea = document.createElement("textarea");
  textarea.spellcheck = false;
  textarea.value = value;
  wrapper.append(backdrop, textarea);

  // Size the backdrop to the textarea CONTENT box: clientWidth excludes
  // the scrollbar, so wrapped lines stay aligned when one appears.
  const syncSize = () => {
    backdrop.style.width = `${textarea.clientWidth}px`;
    backdrop.style.height = `${textarea.clientHeight}px`;
  };
  const syncScroll = () => {
    backdrop.scrollTop = textarea.scrollTop;
    backdrop.scrollLeft = textarea.scrollLeft;
  };
  const render = () => {
    renderRawMarkup(backdrop, textarea.value);
    syncScroll();
  };
  // Fill mode: the node layout owns the height — the textarea stretches
  // with the node and a shortfall is reported via onOverflow so the host
  // can grow the node (no internal cap, no phantom scrollbar while the
  // node has room).
  const refit = () => {
    syncSize();
    render();
  };
  const sync = (text) => {
    if (textarea.value !== text) textarea.value = text;
    // No onOverflow here: a loaded workflow's saved node size rules.
    syncSize();
    render();
  };

  textarea.addEventListener("input", () => {
    refit();
    const deficit = textarea.scrollHeight - textarea.clientHeight;
    if (deficit > 0) onOverflow?.(Math.ceil(deficit) + 1);
    onInput?.(textarea.value);
  });
  textarea.addEventListener("scroll", syncScroll);
  // Consume a wheel only while the field can scroll its own overflow in
  // that direction; anything else is re-dispatched on the graph canvas,
  // so hovering the panel still zooms/pans the canvas. Without this the
  // DOM widget would swallow the wheel — ComfyUI core does not forward
  // wheels out of DOM widgets either (markdown previews behave the
  // same).
  textarea.addEventListener(
    "wheel",
    (event) => {
      if (event.deltaY !== 0) {
        const overflow = textarea.scrollHeight - textarea.clientHeight;
        if (overflow > 0.5) {
          const atTop = textarea.scrollTop <= 0;
          const atBottom = textarea.scrollTop >= overflow - 0.5;
          const canScroll = event.deltaY < 0 ? !atTop : !atBottom;
          if (canScroll) return;
        }
      }
      event.preventDefault();
      const canvas = app.canvas?.canvas;
      if (canvas) canvas.dispatchEvent(new WheelEvent("wheel", event));
    },
    { passive: false },
  );
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      syncSize();
      render();
    }).observe(textarea);
  }
  // No initial auto-grow here: the element is still detached, so
  // scrollHeight is 0 and the height would clamp to the CSS minimum.
  // The host calls refit() once the input is mounted.
  syncSize();
  render();

  return { wrapper, textarea, render, sync, refit };
}

/* ------------------------------------------------------------------ */
/* Styles                                                              */
/* ------------------------------------------------------------------ */

const STYLE = `
/* Markup runs (no horizontal padding anywhere — padding shifts the
   highlight relative to the invisible textarea glyphs). */
.lk-dim { color: rgba(200, 200, 210, 0.38); }
.lk-span { border-radius: 3px; }

/* Rich input: transparent-text textarea over a colored backdrop with
   identical metrics — the highlight appears live while typing. The
   backdrop carries NO border and sits at the textarea's content-box
   origin (+1px border offset), so only one frame is ever visible and
   the glyphs align exactly. */
.lk-rich { position: relative; }
.lk-rich > .lk-backdrop { position: absolute; top: 1px; left: 1px; z-index: 1;
  background: #1e1e1e; border: none; border-radius: 3px;
  font: 13px/1.5 monospace; padding: 8px; margin: 0;
  white-space: pre-wrap; word-break: break-word; box-sizing: border-box;
  overflow: hidden; pointer-events: none; user-select: none;
  color: rgba(222, 222, 228, 0.85); }
.lk-rich > textarea { position: relative; z-index: 2; display: block;
  background: transparent; color: transparent; caret-color: #e8e8e8;
  font: 13px/1.5 monospace; padding: 8px; margin: 0;
  white-space: pre-wrap; word-break: break-word; box-sizing: border-box;
  width: 100%; overflow-y: auto; resize: none;
  border: 1px solid #555; border-radius: 4px; outline: none; }
.lk-rich > textarea:focus { border-color: #777; }
.lk-rich > textarea::selection { background: rgba(120, 160, 255, 0.3); }

/* Compact variant used inside the node body: a FILL widget. ComfyUI
   stretches the wrapper to the DOM-widget area (computedHeight minus
   2x10px margin), the textarea flexes to fill the wrapper — no inline
   height. --comfy-widget-min-height declares the widget's layout
   minimum to the node layout engine (104px = ~84px field + the 2x10px
   margin); no CSS max — auto-growth is capped in JS (onOverflow stops
   at MAX_FIELD_HEIGHT, the field then scrolls), while a manually
   enlarged node still fills all the way. Content past the available
   height grows the node (onOverflow) — the scrollbar only shows at the
   cap or when the node is deliberately smaller than the text. */
.lk-rich.lk-node-rich { display: flex; flex-direction: column;
  --comfy-widget-min-height: 104px; }
.lk-rich.lk-node-rich > textarea,
.lk-rich.lk-node-rich > .lk-backdrop { font: 11px/1.45 monospace; }
.lk-rich.lk-node-rich > textarea { flex: 1 1 auto; min-height: 0;
  max-height: none; }

`;

function ensureStyle() {
  if (document.getElementById("lk-annotator-style")) return;
  const style = document.createElement("style");
  style.id = "lk-annotator-style";
  style.textContent = STYLE;
  document.head.appendChild(style);
}

// Injected at load: the in-node rich input must be styled from its
// very first frame.
ensureStyle();

/* ------------------------------------------------------------------ */
/* In-node rich input widget                                           */
/* ------------------------------------------------------------------ */

/**
 * Replace the node's built-in text widget with a rich input DOM widget.
 *
 * The built-in widget is hidden, not removed: it stays in node.widgets
 * (so its value keeps serializing into the workflow) while the layout
 * engine skips it (getLayoutWidgets filters on widget.hidden). The rich
 * input is placed at the hidden widget's position and kept in sync
 * both ways.
 */
function attachRichInputWidget(node) {
  if (typeof node.addDOMWidget !== "function") return;
  const textWidget = node.widgets?.find((w) => w.name === TEXT_WIDGET);
  if (!textWidget) return;

  const rich = createRichInput({
    value: String(textWidget.value ?? ""),
    className: "lk-node-rich",
    onInput: (text) => {
      textWidget.value = text;
    },
    onOverflow: (deficit) => {
      // Content no longer fits the widget area: grow the node (native
      // multiline-widget behavior) — the rich input keeps filling it.
      // Only up to MAX_FIELD_HEIGHT though: past the cap the field
      // stops growing and scrolls, so the wheel handler's "scroll the
      // field / zoom the canvas at the edges" policy can engage.
      const room = Math.max(0, MAX_FIELD_HEIGHT - rich.textarea.clientHeight);
      const grow = Math.min(deficit, room);
      if (grow <= 0) return;
      node.setSize([node.size[0], node.size[1] + grow]);
      node.setDirtyCanvas(true, true);
    },
  });

  // Interacting with the input must not drag the node. Keydowns must
  // NOT be stopped: ComfyUI's global handlers listen on window and are
  // textarea-aware (Ctrl+Enter queues, Ctrl+Up/Down adjusts attention
  // weight); stopping them here is what used to disable those in this
  // field. Canvas shortcuts still cannot misfire — the global keybinding
  // service ignores text-input-reserved combos targeted at a textarea,
  // and litegraph's key handler only receives events that target the
  // canvas element itself.
  rich.wrapper.addEventListener("mousedown", (event) =>
    event.stopPropagation(),
  );

  const widget = node.addDOMWidget("lk_rich", "lk-rich-input", rich.wrapper, {
    serialize: false,
    getValue: () => "",
    setValue: () => {},
  });
  widget.serialize = false;

  // Hide the built-in widget and seat the rich input in its place.
  textWidget.hidden = true;
  node.widgets.splice(node.widgets.indexOf(widget), 1);
  node.widgets.splice(node.widgets.indexOf(textWidget) + 1, 0, widget);

  // Called on workflow configure to sync the rich input from saved widget value.
  node.lukutarSyncRichInput = () => rich.sync(String(textWidget.value ?? ""));
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.PromptAnnotateEditor",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      attachRichInputWidget(this);
      return result;
    };

    // Serialized widget values are applied after node creation; pull
    // them into the rich input once configure has run.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigure?.apply(this, arguments);
      this.lukutarSyncRichInput?.();
      return result;
    };
  },
});

/* ------------------------------------------------------------------ */
/* Annotation Segment Edit: gray the text widget in delete mode        */
/* ------------------------------------------------------------------ */

const EDIT_NODE_NAME = "AnnotationSegmentEdit";

/**
 * Toggle the text widget's disabled state from the current mode value.
 * The backend ignores the text in delete mode anyway; this only makes
 * that visible (the pack stays fully functional without web assets).
 */
function syncEditWidgets(node) {
  const modeWidget = node.widgets?.find((w) => w.name === "mode");
  const textWidget = node.widgets?.find((w) => w.name === "text");
  if (!modeWidget || !textWidget) return;
  const disabled = modeWidget.value === "delete";
  textWidget.disabled = disabled;
  const element = textWidget.inputEl ?? textWidget.element;
  if (element) element.disabled = disabled;
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "LukutarNodes.AnnotationSegmentEditWidgets",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== EDIT_NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const node = this;
      const modeWidget = node.widgets?.find((w) => w.name === "mode");
      const callback = modeWidget?.callback;
      if (modeWidget) {
        modeWidget.callback = function (...args) {
          const cbResult = callback?.apply(this, args);
          syncEditWidgets(node);
          return cbResult;
        };
      }
      syncEditWidgets(node);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigure?.apply(this, arguments);
      syncEditWidgets(this);
      return result;
    };
  },
});
