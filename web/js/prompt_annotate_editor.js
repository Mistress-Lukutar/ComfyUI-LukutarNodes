/**
 * File:   prompt_annotate_editor.js
 * Brief:  Rich highlighted input + popup editor for Prompt Annotate.
 * Author: Mistress-Lukutar
 * Date:   2026-08-28
 * Version: v0.5.1
 *
 * The node's own prompt field is replaced with a "rich input": a
 * transparent-text textarea stacked over a colored backdrop div with
 * identical metrics, so the markup is highlighted live while typing
 * (native caret, selection, undo and IME keep working). The built-in
 * text widget is only hidden (widget.hidden = true) — it stays in
 * node.widgets, so its value keeps serializing into the workflow.
 *
 * The same rich input (larger) powers the "Annotate..." popup editor
 * with a label palette that wraps the current selection into tags.
 *
 * Alignment rules that keep the caret glued to the highlight:
 *  - the backdrop carries NO horizontal padding/border widths that the
 *    textarea does not have (span padding used to drift the caret 4px
 *    per tag);
 *  - the backdrop is sized to the textarea's clientWidth/clientHeight
 *    (excluding the scrollbar) and re-synced on resize;
 *  - both share font, line-height, padding, wrap and box-sizing.
 *
 * Served at /extensions/<pack>/js/prompt_annotate_editor.js, hence the
 * THREE parent hops to reach /scripts/app.js (two hops would resolve
 * inside /extensions/ and 404, killing the whole module).
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes] prompt annotate editor extension loaded");

const NODE_NAME = "PromptAnnotate";
const TEXT_WIDGET = "text";

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

/** Labels used inside the marked-up text, in order of first use. */
function labelsInText(text) {
  const labels = [];
  TAG_RE.lastIndex = 0;
  let match;
  while ((match = TAG_RE.exec(text)) !== null) {
    for (let label of match[1].split(",")) {
      label = label.trim();
      if (label && !labels.includes(label)) labels.push(label);
    }
  }
  return labels;
}

/** True when the markup looks unbalanced (odd '|' count, ':'-less tags). */
function markupError(text) {
  const pipes = (text.match(/\|/g) || []).length;
  if (pipes % 2 !== 0) return "Odd number of '|' — a tag is not closed.";
  let stripped = text.replace(TAG_RE, "");
  if (stripped.includes("|")) {
    return "A '|' pair is missing the ':' between labels and text.";
  }
  return null;
}

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
 */
function createRichInput({ value = "", className = "", onInput = null } = {}) {
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
  // Auto-growing field: no manual resize grip, the height follows the
  // content between CSS min-height and max-height (scrollbar appears
  // past the cap).
  const autoGrow = () => {
    textarea.style.height = "auto";
    const styles = window.getComputedStyle(textarea);
    const min = parseFloat(styles.minHeight) || 0;
    const max = parseFloat(styles.maxHeight) || Number.POSITIVE_INFINITY;
    const needed = textarea.scrollHeight + 2; // + textarea borders
    textarea.style.height = `${Math.min(Math.max(needed, min), max)}px`;
    syncSize();
    render();
  };
  const sync = (text) => {
    if (textarea.value !== text) textarea.value = text;
    autoGrow();
  };

  textarea.addEventListener("input", () => {
    autoGrow();
    onInput?.(textarea.value);
  });
  textarea.addEventListener("scroll", syncScroll);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      syncSize();
      render();
    }).observe(textarea);
  }
  autoGrow();

  return { wrapper, textarea, render, sync, autoGrow };
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
  font: 12px/1.5 monospace; padding: 8px; margin: 0;
  white-space: pre-wrap; word-break: break-word; box-sizing: border-box;
  overflow: hidden; pointer-events: none; user-select: none;
  color: rgba(222, 222, 228, 0.85); }
.lk-rich > textarea { position: relative; z-index: 2; display: block;
  background: transparent; color: transparent; caret-color: #e8e8e8;
  font: 12px/1.5 monospace; padding: 8px; margin: 0;
  white-space: pre-wrap; word-break: break-word; box-sizing: border-box;
  width: 100%; min-height: 140px; max-height: 60vh;
  overflow-y: auto; resize: none;
  border: 1px solid #555; border-radius: 4px; outline: none; }
.lk-rich > textarea:focus { border-color: #777; }
.lk-rich > textarea::selection { background: rgba(120, 160, 255, 0.3); }

/* Compact variant used inside the node body. */
.lk-rich.lk-node-rich > textarea,
.lk-rich.lk-node-rich > .lk-backdrop { font: 11px/1.45 monospace; }
.lk-rich.lk-node-rich > textarea { min-height: 48px; max-height: 320px; }

.lk-overlay { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.6);
  display: flex; align-items: center; justify-content: center; }
.lk-panel { background: #2b2b2b; color: #dcdcdc; border: 1px solid #555;
  border-radius: 8px; padding: 14px; width: min(820px, 92vw);
  display: flex; flex-direction: column; gap: 10px;
  font: 12px/1.4 sans-serif; box-shadow: 0 8px 32px rgba(0,0,0,.5); }
.lk-title { font-size: 14px; font-weight: bold; }
.lk-palette { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.lk-chip { padding: 3px 10px; border-radius: 10px; cursor: pointer;
  border: 1px solid #666; background: transparent; }
.lk-chip:hover { filter: brightness(1.3); }
.lk-hint { color: #8a8a92; padding: 3px 0; }
.lk-error { color: #ff8f88; min-height: 14px; }
.lk-buttons { display: flex; gap: 8px; justify-content: flex-end; }
.lk-buttons button { padding: 5px 14px; border-radius: 4px; cursor: pointer;
  border: 1px solid #666; background: #3a3a3a; color: #ddd; }
.lk-buttons .lk-save { background: #4a7a4a; }
.lk-tools { display: flex; gap: 8px; }
.lk-tools button { padding: 3px 10px; border-radius: 4px; cursor: pointer;
  border: 1px solid #555; background: #333; color: #bbb; font-size: 11px; }
`;

function ensureStyle() {
  if (document.getElementById("lk-annotator-style")) return;
  const style = document.createElement("style");
  style.id = "lk-annotator-style";
  style.textContent = STYLE;
  document.head.appendChild(style);
}

// Injected at load: the in-node rich input must be styled from its
// first frame, before any popup has ever been opened.
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
  });

  // Interacting with the input must not drag the node or fire canvas
  // keyboard shortcuts.
  rich.wrapper.addEventListener("mousedown", (event) =>
    event.stopPropagation(),
  );
  rich.textarea.addEventListener("keydown", (event) =>
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

  // Called after popup-editor saves and on workflow configure.
  node.lukutarSyncRichInput = () => rich.sync(String(textWidget.value ?? ""));
}

/* ------------------------------------------------------------------ */
/* Popup editor                                                        */
/* ------------------------------------------------------------------ */

function openEditor(node) {
  const textWidget = node.widgets?.find((w) => w.name === TEXT_WIDGET);
  if (!textWidget) return;

  const overlay = document.createElement("div");
  overlay.className = "lk-overlay";

  const panel = document.createElement("div");
  panel.className = "lk-panel";

  const title = document.createElement("div");
  title.className = "lk-title";
  title.textContent = `Annotate prompt — ${node.title || NODE_NAME}`;

  const palette = document.createElement("div");
  palette.className = "lk-palette";

  const tools = document.createElement("div");
  tools.className = "lk-tools";
  const unwrapBtn = document.createElement("button");
  unwrapBtn.textContent = "Unwrap selection";
  const stripBtn = document.createElement("button");
  stripBtn.textContent = "Strip all markup";
  tools.append(unwrapBtn, stripBtn);

  const rich = createRichInput({
    value: String(textWidget.value ?? ""),
    onInput: () => refresh(),
  });

  const errorLine = document.createElement("div");
  errorLine.className = "lk-error";

  const buttons = document.createElement("div");
  buttons.className = "lk-buttons";
  const cancelBtn = document.createElement("button");
  cancelBtn.textContent = "Cancel";
  const saveBtn = document.createElement("button");
  saveBtn.className = "lk-save";
  saveBtn.textContent = "Save";
  buttons.append(cancelBtn, saveBtn);

  panel.append(title, palette, tools, rich.wrapper, errorLine, buttons);
  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  rich.textarea.focus();

  function close() {
    overlay.remove();
  }

  function save() {
    textWidget.value = rich.textarea.value;
    node.lukutarSyncRichInput?.();
    node.setDirtyCanvas(true, true);
    close();
  }

  cancelBtn.addEventListener("click", close);
  saveBtn.addEventListener("click", save);
  overlay.addEventListener("mousedown", (event) => {
    if (event.target === overlay) close();
  });
  panel.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      close();
    }
  });
  rich.textarea.addEventListener("keydown", (event) => {
    // Keep canvas shortcuts out; handle Esc/Ctrl+Enter locally since
    // stopPropagation blocks bubbling to the panel.
    event.stopPropagation();
    if (event.key === "Escape") {
      close();
    } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      save();
    }
  });

  function refresh() {
    // Palette: labels already used in the text — chips wrap the
    // current selection for quick re-use. New labels are simply typed
    // into the markup.
    const text = rich.textarea.value;
    const labels = labelsInText(text);
    palette.replaceChildren();
    if (!labels.length) {
      const hint = document.createElement("span");
      hint.className = "lk-hint";
      hint.textContent =
        "labels appear here once used in the text, e.g. |face: ...|";
      palette.appendChild(hint);
    }
    for (const label of labels) {
      const chip = document.createElement("button");
      chip.className = "lk-chip";
      const hue = labelHue(label);
      chip.style.background = pastel.bg(hue);
      chip.style.borderColor = pastel.border(hue);
      chip.style.color = pastel.text(hue);
      chip.textContent = label;
      chip.title = `Wrap the selection in |${label}:...|`;
      chip.addEventListener("click", () => wrapSelection(label));
      palette.appendChild(chip);
    }

    errorLine.textContent = markupError(text) ?? "";
  }

  function wrapSelection(label) {
    const textarea = rich.textarea;
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? start;
    const selected = textarea.value.slice(start, end);
    const tag = `|${label}:${selected}|`;
    textarea.setRangeText(tag, start, end);
    textarea.focus();
    const innerStart = start + label.length + 1;
    textarea.setSelectionRange(innerStart, innerStart + selected.length);
    rich.autoGrow();
    refresh();
  }

  function unwrapSelection() {
    const textarea = rich.textarea;
    const text = textarea.value;
    const selStart = textarea.selectionStart ?? 0;
    const selEnd = textarea.selectionEnd ?? selStart;
    TAG_RE.lastIndex = 0;
    let match;
    while ((match = TAG_RE.exec(text)) !== null) {
      const start = match.index;
      const end = match.index + match[0].length;
      const overlaps =
        (selStart >= start && selStart <= end) ||
        (selEnd >= start && selEnd <= end) ||
        (selStart <= start && selEnd >= end);
      if (overlaps) {
        textarea.setRangeText(match[2], start, end, "end");
        textarea.focus();
        rich.autoGrow();
        refresh();
        return;
      }
    }
    errorLine.textContent =
      "No tag under the selection — put the caret inside a |tag| first.";
  }

  unwrapBtn.addEventListener("click", unwrapSelection);
  stripBtn.addEventListener("click", () => {
    TAG_RE.lastIndex = 0;
    rich.textarea.value = rich.textarea.value.replace(TAG_RE, "$2");
    rich.autoGrow();
    refresh();
    rich.textarea.focus();
  });

  refresh();
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
      this.addWidget("button", "Annotate...", null, () => openEditor(this));
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
