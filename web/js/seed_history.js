/**
 * File:   seed_history.js
 * Brief:  Seed History node — rolling log of used seeds, displayed with
 *         the core "Preview as Text" widget, copy via its button.
 * Author: Mistress-Lukutar
 * Date:   2026-10-17
 * Version: v0.4.0
 *
 * The backend (nodes/seed_history.py) has no idea about histories: the
 * seed it receives is final.  This extension owns the log:
 *
 *   - The log lives in the node property ``seed_history`` (a plain
 *     array of seeds), which ComfyUI serializes with the workflow —
 *     the history survives reloads without any widget gymnastics.
 *     Logs saved by older builds (kept in a hidden widget value) are
 *     adopted on load.
 *   - Display reuses ComfyUI's own "Preview as Text" machinery
 *     (``window.comfyAPI.textPreviewWidgets``, the very widget the
 *     core PreviewText node renders): a readonly text area below the
 *     parameters, one seed per line, with a copy-to-clipboard button
 *     in its top-right corner.  The Markdown/Plain-text toggle the
 *     helper adds is removed again — plain text is all the log needs.
 *     While the log is empty the panel shows a placeholder line, so
 *     the field is always visible.  If the helper is missing or fails
 *     (frontend version differences), a plain ``textarea`` widget is
 *     used instead — never silently nothing.
 *   - The seed widget's ``serializeValue`` is wrapped: ComfyUI calls it
 *     while building the prompt for a queue, which is exactly the
 *     moment a seed is "used" (after the frontend already applied
 *     control_after_generate).  The outgoing value is prepended to the
 *     log — newest seed first, older ones pushed down — and the log
 *     is trimmed to ``history_size``.
 *
 * Setup runs through TWO independent hooks — the ``onNodeCreated``
 * prototype wrap (the same hook the core PreviewText extension uses,
 * invoked synchronously by LiteGraph.createNode) and the app-level
 * ``nodeCreated`` callback — whichever fires first wins; a guard flag
 * makes the second a no-op.  Every step is isolated: a display failure
 * can no longer take down the seed-capture hook, and everything logs
 * under ``[LukutarNodes.SeedHistory]`` for diagnosis.
 *
 * The node still works without this extension — it is then a plain
 * seed source without a displayed history.
 *
 * Served at /extensions/<pack>/js/seed_history.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.SeedHistory] extension loaded");

const NODE_NAME = "SeedHistory";
const HISTORY_PROPERTY = "seed_history";
const PREVIEW_WIDGET_NAME = "preview_text";
const PREVIEW_MODE_WIDGET = "preview_mode";
const FALLBACK_WIDGET_NAME = "seed_history_log";
const EMPTY_LOG_TEXT = "(no seeds yet — queue a generation)";

function log(msg, ...rest) {
  console.info(`[LukutarNodes.SeedHistory] ${msg}`, ...rest);
}

function warn(msg, ...rest) {
  console.warn(`[LukutarNodes.SeedHistory] ${msg}`, ...rest);
}

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/** Log entries as numbers, whatever the property currently holds. */
function entriesOf(node) {
  const raw = node.properties?.[HISTORY_PROPERTY];
  return Array.isArray(raw) ? raw.map(Number).filter(Number.isFinite) : [];
}

function historyLimit(node) {
  const raw = Number(findWidget(node, "history_size")?.value);
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : 5;
}

function storeEntries(node, entries) {
  node.properties ??= {};
  node.properties[HISTORY_PROPERTY] = entries;
}

/** Text shown in the display: the log, or a placeholder while empty. */
function displayText(node) {
  const entries = entriesOf(node);
  return entries.length ? entries.join("\n") : EMPTY_LOG_TEXT;
}

/** Mirror the log into whichever display widget exists (if any). */
function refreshPreview(node) {
  const text = displayText(node);
  const preview = findWidget(node, PREVIEW_WIDGET_NAME);
  if (preview) {
    preview.value = text;
    return;
  }
  const fallback = findWidget(node, FALLBACK_WIDGET_NAME);
  if (fallback) fallback.value = text;
}

/** Prepend a used seed (newest on top) and trim to history_size. */
function pushSeed(node, seed) {
  if (!Number.isFinite(seed)) return;
  const entries = entriesOf(node);
  if (entries[0] === seed) return;
  entries.unshift(seed);
  const limit = historyLimit(node);
  if (entries.length > limit) {
    entries.splice(limit);
  }
  storeEntries(node, entries);
  refreshPreview(node);
  log("seed captured:", seed, `(log: ${entries.length}/${limit})`);
}

/** Trim the log to the current history_size (history_size changed). */
function trimHistory(node) {
  const entries = entriesOf(node);
  const limit = historyLimit(node);
  if (entries.length > limit) {
    storeEntries(node, entries.slice(0, limit));
  }
  refreshPreview(node);
}

/** Adopt logs saved by older builds, which kept the array in a widget value. */
function migrateLegacyLog(node, info) {
  if (node.properties?.[HISTORY_PROPERTY] !== undefined) return;
  const legacy = info?.widgets_values_named?.[HISTORY_PROPERTY];
  if (Array.isArray(legacy)) {
    storeEntries(node, legacy.map(Number).filter(Number.isFinite));
    log("legacy widget log adopted:", node.properties[HISTORY_PROPERTY].length, "entries");
  }
}

/**
 * Wrap the seed widget's serializeValue — ComfyUI calls it while
 * building the prompt for a queue, after control_after_generate was
 * applied, so the outgoing value is exactly the seed this generation
 * uses.  (Also called for API exports — an accepted false positive.)
 */
function hookSeedCapture(node) {
  const seedWidget = findWidget(node, "seed");
  if (!seedWidget) {
    warn("seed widget not found — capture disabled (converted to input?)");
    return;
  }
  if (seedWidget.__lkSeedHistoryHooked) return;
  seedWidget.__lkSeedHistoryHooked = true;
  const origSerialize = seedWidget.serializeValue;
  seedWidget.serializeValue = function (target, index) {
    const value = origSerialize
      ? origSerialize.call(this, target, index)
      : this.value;
    pushSeed(node, Number(value));
    return value;
  };
}

/** Idempotently hook a widget callback for trims / refreshes. */
function hookWidgetCallback(node, name, onChange) {
  const widget = findWidget(node, name);
  if (!widget || widget.__lkSeedHistoryCbHooked) return;
  widget.__lkSeedHistoryCbHooked = true;
  const origCallback = widget.callback;
  widget.callback = function (value, ...rest) {
    const result = origCallback?.apply(this, [value, ...rest]);
    onChange(node);
    return result;
  };
}

/**
 * Add the display below the parameters.  Primary: the core
 * "Preview as Text" widget (its Markdown toggle is removed — the log
 * is plain text).  Fallback: a plain textarea widget.  Both stay out
 * of prompts and workflow saves.
 */
function setupDisplay(node) {
  const api = window.comfyAPI?.textPreviewWidgets;
  if (api) {
    try {
      api.addTextPreviewWidgets(node);
      const index = node.widgets?.findIndex(
        (w) => w.name === PREVIEW_MODE_WIDGET
      );
      if (index > -1) {
        node.widgets[index].onRemove?.();
        node.widgets.splice(index, 1);
      }
      log("core 'Preview as Text' widget added");
      return;
    } catch (err) {
      warn("textPreviewWidgets failed, falling back to a textarea", err);
    }
  } else {
    warn("comfyAPI.textPreviewWidgets unavailable, using a textarea");
  }
  try {
    const widget = node.addWidget("textarea", FALLBACK_WIDGET_NAME, "", () => {});
    widget.serialize = false;
    widget.options = { ...widget.options, serialize: false };
    log("fallback textarea widget added");
  } catch (err) {
    warn("fallback widget failed too; running without display", err);
  }
}

/** One-time per-node setup; safe to call from several hooks. */
function setupNode(node) {
  if (node.__lkSeedHistoryReady) return;
  node.__lkSeedHistoryReady = true;
  try {
    hookSeedCapture(node);
  } catch (err) {
    warn("seed capture hook failed", err);
  }
  try {
    hookWidgetCallback(node, "history_size", trimHistory);
  } catch (err) {
    warn("history_size callback hook failed", err);
  }
  setupDisplay(node);
  refreshPreview(node);

  // After a workflow load: adopt logs saved by older builds (which
  // kept the array in a widget value) and fill the display.
  const origOnConfigure = node.onConfigure;
  node.onConfigure = function (info) {
    origOnConfigure?.apply(this, arguments);
    migrateLegacyLog(this, info);
    refreshPreview(this);
  };
  log("node ready (id", node.id + ")");
}

/* ------------------------------------------------------------------ */
/* Registration — two independent hook paths, guarded to be idempotent */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.SeedHistory",

  // Path 1: prototype wrap — the same hook the core PreviewText
  // ("Preview as Text") extension uses.  LiteGraph.createNode invokes
  // onNodeCreated synchronously once the INPUT_TYPES widgets (incl.
  // the auto control_after_generate) exist.
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setupNode(this);
      return result;
    };
  },

  // Path 2: app-level callback — covers creation paths that never go
  // through LiteGraph.createNode.
  nodeCreated(node) {
    if (node.type !== NODE_NAME) return;
    setupNode(node);
  },
});
