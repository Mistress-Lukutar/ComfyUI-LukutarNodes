/**
 * File:   group_switch.js
 * Brief:  Group Switch node: one boolean row per canvas group.
 * Author: Mistress-Lukutar
 * Date:   2026-10-17
 * Version: v1.1.0
 *
 * The backend node (nodes/group_switch.py) only republishes the state
 * payload; all panel behaviour lives here. For every group on the
 * canvas the node grows a boolean row (rows appear, reorder and vanish
 * as groups change). A checked row means the group is active; unchecking
 * switches the group off with the chosen method:
 *
 *   bypass  — every node in the group is set to ComfyUI bypass mode
 *   disable — every node in the group is set to ComfyUI never (mute)
 *
 * This is exactly what the native group context menu does ("Bypass Group
 * Nodes" / "Set Group Nodes to Never"), so a group switched off here
 * stays understandable to ComfyUI: muting excludes the nodes from the
 * prompt, bypass wires inputs through. When change_group_color is on,
 * the group's frame is recolored while it is off — purple for bypass,
 * grey for disable — and the original color is restored on re-enable.
 * The original color is remembered in a node property
 * (``lukutar_group_meta``), which is serialized with the workflow — so
 * recolored groups survive reloads and get their color back even when
 * they are re-enabled through the native group menu.
 *
 * Persistence rules (the reason rows are serialize:false):
 *   - The group row widgets are created asynchronously from the live
 *     graph, so their count at load time is unpredictable. They are
 *     marked serialize:false — neither the workflow save nor the
 *     positional widgets_values restore ever assigns them a slot. The
 *     four declared widgets (disable_method, change_group_color,
 *     sort_order, group_states) therefore keep a stable positional
 *     contract and survive reloads untouched.
 *   - Row values are not persisted at all: they are re-derived from the
 *     group's node modes, which ComfyUI saves natively. A group that
 *     was muted when the workflow was saved shows unchecked again.
 *
 * Row order follows the sort_order alphabet (comma-separated title
 * prefixes — a group whose title starts with a token takes that
 * token's position); unlisted groups keep their relative order at the
 * end. The current states are mirrored into the hidden `group_states`
 * widget as JSON, which is the value the backend actually receives.
 *
 * Sync triggers: node create/configure/remove plus a cheap 1s poll, the
 * same pattern as variables.js — current litegraph builds fire no
 * callback on group add/remove/title-rename or on mode changes made
 * through the native group menu.
 *
 * Served at /extensions/<pack>/js/group_switch.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes] group switch extension loaded");

const NODE_NAME = "GroupSwitch";

/** Declared widget names (mirrored in nodes/group_switch.py). */
const W_METHOD = "disable_method";
const W_COLOR = "change_group_color";
const W_SORT = "sort_order";
const W_STATES = "group_states";
const DECLARED = [W_METHOD, W_COLOR, W_SORT, W_STATES];

/** Node property holding per-group frontend bookkeeping. */
const META_PROPERTY = "lukutar_group_meta";

/** LiteGraph node modes — same values the native group menu writes. */
const MODE_ALWAYS = 0;
const MODE_NEVER = 2; // ComfyUI "mute" (disable)
const MODE_BYPASS = 4; // ComfyUI "bypass"

/** Methods the backend accepts, mirrored from core/group_switch.py. */
const DISABLE_METHODS = ["bypass", "disable"];

/** Frame colors for switched-off groups. */
const COLOR_BYPASS = "#a1309b"; // purple, ComfyUI theme
const COLOR_DISABLE = "#888888"; // grey

/** Expando slot on LGraphGroup holding the switch bookkeeping. */
const GROUP_STATE = "__lkGroupSwitch";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name && !w.__lkGroup);
}

function rowWidgets(node) {
  return (node.widgets ?? []).filter((w) => w.__lkGroup);
}

function removeWidget(node, widget) {
  const index = (node.widgets ?? []).indexOf(widget);
  if (index >= 0) node.widgets.splice(index, 1);
  widget.onRemove?.();
}

function setTitle(node, title) {
  if (node.title !== title) node.title = title;
}

function graphChanged() {
  app.graph?.change?.();
  app.canvas?.setDirty?.(true, true);
}

/**
 * Nodes currently inside a group. Uses the same recomputation the
 * native group menu relies on; detatched groups simply have none.
 */
function insideNodes(group) {
  try {
    group.recomputeInsideNodes?.();
  } catch {
    return [];
  }
  return group.nodes ?? [];
}

/**
 * Group state read back from its nodes' modes: off-via-bypass when all
 * nodes are bypassed, off-via-disable when all are muted, on otherwise.
 */
function deriveState(group) {
  const nodes = insideNodes(group);
  if (!nodes.length) return { off: false, method: null };
  let allBypass = true;
  let allNever = true;
  for (const node of nodes) {
    const mode = node.mode ?? MODE_ALWAYS;
    if (mode !== MODE_BYPASS) allBypass = false;
    if (mode !== MODE_NEVER) allNever = false;
  }
  if (allBypass) return { off: true, method: "bypass" };
  if (allNever) return { off: true, method: "disable" };
  return { off: false, method: null };
}

/** Mode a group sits in when switched off via *method*. */
function offMode(method) {
  return method === "bypass" ? MODE_BYPASS : MODE_NEVER;
}

/* ------------------------------------------------------------------ */
/* Per-group bookkeeping in node properties (survives reloads)          */
/* ------------------------------------------------------------------ */

/**
 * ``{[title]: {color, method}}`` — the frame color a group had before
 * the panel recolored it (null color = never recolored). Lives in
 * node.properties, so it is saved and restored with the workflow.
 */
function metaOf(node) {
  const props = (node.properties ??= {});
  const meta = props[META_PROPERTY];
  if (meta && typeof meta === "object") return meta;
  return (props[META_PROPERTY] = {});
}

function metaEntry(node, group) {
  return metaOf(node)[String(group.title)];
}

function setMetaEntry(node, group, entry) {
  const meta = metaOf(node);
  if (entry) meta[String(group.title)] = entry;
  else delete meta[String(group.title)];
}

/** Drop bookkeeping for groups that no longer exist on the canvas. */
function pruneMeta(node, groups) {
  const meta = metaOf(node);
  const titles = new Set(groups.map((group) => String(group.title)));
  for (const title of Object.keys(meta)) {
    if (!titles.has(title)) delete meta[title];
  }
}

/**
 * Reconcile recoloring with the live graph for one group. Handles the
 * two cases the toggle itself cannot: a group re-enabled through the
 * native group menu (restore its original color), and a workflow
 * reloaded while the group was off (re-apply the recolor if it was not
 * saved with the workflow).
 */
function reconcileColor(node, group, changeColor, method) {
  const saved = metaEntry(node, group);
  if (!saved || saved.color == null) return;
  if (!deriveState(group).off) {
    group.color = saved.color;
    setMetaEntry(node, group, null);
    return;
  }
  if (changeColor && group.color === saved.color) {
    const offMethod = saved.method ?? method;
    group.color = offMethod === "bypass" ? COLOR_BYPASS : COLOR_DISABLE;
  }
}

/* ------------------------------------------------------------------ */
/* Switching                                                           */
/* ------------------------------------------------------------------ */

/**
 * Switch a group on or off. The original node modes and frame color are
 * captured on switch-off and restored on switch-on, so nodes the user
 * had muted individually keep that state across a panel toggle. Both
 * survive reloads: modes are saved by ComfyUI per node, the color via
 * the node property above.
 */
function applyGroup(node, group, enabled, method, changeColor) {
  const nodes = insideNodes(group);
  let state = group[GROUP_STATE];
  if (enabled) {
    // How the group is off right now: from the live modes, or from the
    // bookkeeping after this panel switched it off.
    const current = deriveState(group).method ?? state?.method;
    if (state?.savedModes?.size) {
      for (const inner of nodes) {
        const saved = state.savedModes.get(inner);
        if (saved != null) inner.mode = saved;
      }
      state.savedModes.clear();
    } else if (current != null) {
      // No per-node memory (e.g. right after a reload): reset the
      // nodes that sit in the group's current off-mode.
      const mode = offMode(current);
      for (const inner of nodes) {
        if ((inner.mode ?? MODE_ALWAYS) === mode) inner.mode = MODE_ALWAYS;
      }
    }
    const saved = metaEntry(node, group);
    if (saved?.color != null) group.color = saved.color;
    setMetaEntry(node, group, null);
    if (state) {
      state.off = false;
      state.savedColor = null;
    }
  } else {
    if (!state) {
      state = group[GROUP_STATE] = { savedModes: new Map(), method };
    }
    if (!state.savedModes.size) {
      for (const inner of nodes) {
        state.savedModes.set(inner, inner.mode ?? MODE_ALWAYS);
      }
    }
    for (const inner of nodes) inner.mode = offMode(method);
    state.off = true;
    state.method = method;
    if (changeColor) {
      if (metaEntry(node, group)?.color == null) {
        setMetaEntry(node, group, {
          color: group.color ?? null,
          method,
        });
      }
      group.color = offMode(method) === MODE_BYPASS ? COLOR_BYPASS : COLOR_DISABLE;
    } else {
      const saved = metaEntry(node, group);
      if (saved?.color != null) {
        group.color = saved.color;
        setMetaEntry(node, group, null);
      }
    }
  }
  graphChanged();
}

/* ------------------------------------------------------------------ */
/* Ordering                                                            */
/* ------------------------------------------------------------------ */

function parseSortTokens(raw) {
  return String(raw ?? "")
    .split(",")
    .map((tok) => tok.trim().toLowerCase())
    .filter(Boolean);
}

/**
 * Stable order by the sort alphabet of title PREFIXES: the first token
 * a title starts with fixes its position, everything else keeps its
 * relative order at the end. "in,im,de" sorts "Inpaint Fix" before
 * "Image Upscale" before "Detection"/"Detailer" without typing the
 * full titles.
 */
function orderGroups(groups, sortRaw) {
  const tokens = parseSortTokens(sortRaw);
  if (!tokens.length) return [...groups];
  const rank = (group, index) => {
    const title = String(group.title).trim().toLowerCase();
    for (let i = 0; i < tokens.length; i++) {
      if (title.startsWith(tokens[i])) return i;
    }
    return tokens.length + index;
  };
  return groups
    .map((group, index) => ({ group, index }))
    .sort((a, b) => rank(a.group, a.index) - rank(b.group, b.index) || a.index - b.index)
    .map((entry) => entry.group);
}

/** Unique JSON keys for a group list: "Title", "Title (2)", ... */
function uniqueKeys(groups) {
  const seen = new Map();
  return groups.map((group) => {
    const title = String(group.title);
    const count = (seen.get(title) ?? 0) + 1;
    seen.set(title, count);
    return count === 1 ? title : `${title} (${count})`;
  });
}

/* ------------------------------------------------------------------ */
/* State payload widget                                                */
/* ------------------------------------------------------------------ */

/** Mirror the row values into the hidden group_states JSON widget. */
function updateStatesWidget(node) {
  const statesWidget = findWidget(node, W_STATES);
  if (!statesWidget) return;
  const rows = rowWidgets(node);
  const keys = uniqueKeys(rows.map((w) => w.__lkGroup));
  const states = {};
  rows.forEach((widget, index) => {
    states[keys[index]] = widget.value === true;
  });
  const json = JSON.stringify(states);
  if (statesWidget.value !== json) statesWidget.value = json;
}

/* ------------------------------------------------------------------ */
/* Per-node sync                                                       */
/* ------------------------------------------------------------------ */

function syncNode(node) {
  const graph = app.graph;
  const methodWidget = findWidget(node, W_METHOD);
  const colorWidget = findWidget(node, W_COLOR);
  const sortWidget = findWidget(node, W_SORT);
  const statesWidget = findWidget(node, W_STATES);
  if (!graph || !methodWidget || !colorWidget || !sortWidget || !statesWidget) {
    return;
  }

  const groups = graph.groups ?? [];
  const ordered = orderGroups(groups, sortWidget.value);

  // Drop rows whose group is gone from the canvas.
  for (const widget of rowWidgets(node)) {
    if (!ordered.includes(widget.__lkGroup)) removeWidget(node, widget);
  }
  pruneMeta(node, groups);

  // Grow rows for new groups, reflecting the actual group state.
  for (const group of ordered) {
    if (rowWidgets(node).some((w) => w.__lkGroup === group)) continue;
    const widget = node.addWidget(
      "toggle",
      group.title,
      !deriveState(group).off,
      (value) => {
        applyGroup(
          node,
          group,
          value === true,
          findWidget(node, W_METHOD)?.value ?? "bypass",
          findWidget(node, W_COLOR)?.value === true
        );
        updateStatesWidget(node);
        app.canvas?.setDirty?.(true, true);
      }
    );
    widget.__lkGroup = group;
    // Rows are derived from the live graph and must never take a slot
    // in the saved widgets_values (save or positional restore).
    widget.serialize = false;
    widget.options = { ...widget.options, serialize: false };
  }

  // Reorder: rows follow the ordering, declared widgets stay below.
  // Mutate the array in place — other ComfyUI code holds references to
  // node.widgets.
  const rowsByGroup = new Map(
    rowWidgets(node).map((w) => [w.__lkGroup, w])
  );
  const tail = (node.widgets ?? []).filter((w) => !w.__lkGroup);
  const reordered = [
    ...ordered.map((group) => rowsByGroup.get(group)).filter(Boolean),
    ...tail,
  ];
  const current = node.widgets ?? [];
  const sameOrder =
    current.length === reordered.length &&
    reordered.every((w, i) => current[i] === w);
  if (!sameOrder && node.widgets) {
    node.widgets.length = 0;
    node.widgets.push(...reordered);
  }

  // Refresh labels and values from the live graph: catches renames,
  // native group-menu toggles and reloads. Never applies anything here
  // except the color reconciliation below.
  const changeColor = colorWidget.value === true;
  const method = methodWidget.value ?? "bypass";
  let onCount = 0;
  for (const group of ordered) {
    const widget = rowsByGroup.get(group);
    if (!widget) continue;
    const title = String(group.title);
    if (widget.name !== title) widget.name = title;
    const enabled = !deriveState(group).off;
    if (widget.value !== enabled) widget.value = enabled;
    if (enabled) onCount += 1;
    reconcileColor(node, group, changeColor, method);
  }
  updateStatesWidget(node);

  const base = "Group Switch";
  setTitle(
    node,
    ordered.length ? `${base} · ${onCount}/${ordered.length} on` : base
  );
}

function switchNodes() {
  return (app.graph?._nodes ?? []).filter((node) => node.type === NODE_NAME);
}

function syncAll() {
  if (!app.graph) return;
  for (const node of switchNodes()) syncNode(node);
}

let syncTimer = null;
/** Coalesce a burst of events into one pass on the next tick. */
function scheduleSync() {
  if (syncTimer != null) return;
  syncTimer = setTimeout(() => {
    syncTimer = null;
    syncAll();
  }, 0);
}

/* ------------------------------------------------------------------ */
/* Node parameter callbacks                                            */
/* ------------------------------------------------------------------ */

/** Re-apply the panel settings to groups this switch has turned off. */
function onParamChanged(node, name) {
  if (name === W_SORT) {
    scheduleSync();
    return;
  }
  const method = findWidget(node, W_METHOD)?.value ?? "bypass";
  const changeColor = findWidget(node, W_COLOR)?.value === true;
  for (const widget of rowWidgets(node)) {
    const group = widget.__lkGroup;
    // Only groups disabled through this panel are re-applied; groups
    // muted via the native menu keep their own state.
    if (!group[GROUP_STATE]?.off) continue;
    applyGroup(node, group, false, method, changeColor);
  }
  updateStatesWidget(node);
  app.canvas?.setDirty?.(true, true);
}

function wrapParamCallbacks(node) {
  for (const name of [W_METHOD, W_COLOR, W_SORT]) {
    const widget = findWidget(node, name);
    if (!widget || widget.__lkGroupSwitchHooked) continue;
    widget.__lkGroupSwitchHooked = true;
    const callback = widget.callback;
    widget.callback = function (value, ...rest) {
      const result = callback?.apply(this, [value, ...rest]);
      onParamChanged(node, name);
      return result;
    };
  }
}

/* ------------------------------------------------------------------ */
/* Load heal for workflows saved by older pack versions                 */
/* ------------------------------------------------------------------ */

/**
 * Versions before v1.1.0 serialized the group rows, so their saved
 * widgets_values array is [rows..., method, color, sort, states]. On a
 * positional restore the first rows land in the declared widgets —
 * disable_method becomes false, sort_order becomes "false". With the
 * frontend's named restore this heals itself; when only the positional
 * array is available, re-read the declared values from the tail.
 */
function healRestoredValues(node, info) {
  const values = info?.widgets_values;
  if (!Array.isArray(values) || values.length <= DECLARED.length) return;
  const methodWidget = findWidget(node, W_METHOD);
  if (!methodWidget || DISABLE_METHODS.includes(methodWidget.value)) return;
  const [method, color, sort, states] = values.slice(-DECLARED.length);
  if (typeof method === "string" && DISABLE_METHODS.includes(method)) {
    methodWidget.value = method;
  }
  const colorWidget = findWidget(node, W_COLOR);
  if (colorWidget && typeof color === "boolean") colorWidget.value = color;
  const sortWidget = findWidget(node, W_SORT);
  if (sortWidget && typeof sort === "string") sortWidget.value = sort;
  const statesWidget = findWidget(node, W_STATES);
  if (statesWidget && typeof states === "string") statesWidget.value = states;
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.GroupSwitch",
  setup() {
    scheduleSync();
    // No callback fires on group add/remove/rename or on mode changes
    // made through the native group menu; the cheap periodic pass keeps
    // rows and values correct, same as variables.js.
    setInterval(scheduleSync, 1000);
  },
  nodeCreated() {
    scheduleSync();
  },
  afterConfigureGraph() {
    scheduleSync();
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      // The backend reads the JSON payload; it is not a user control.
      // computeSize collapses it on the canvas — the legacy `hidden`
      // flag alone is not honoured by the canvas widget renderer.
      const statesWidget = findWidget(this, W_STATES);
      if (statesWidget) {
        statesWidget.hidden = true;
        statesWidget.computeSize = () => [0, -4];
      }
      wrapParamCallbacks(this);
      scheduleSync();
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      healRestoredValues(this, args[0]);
      scheduleSync();
      return result;
    };

    for (const method of ["onConnectionsChange", "onRemoved"]) {
      const orig = nodeType.prototype[method];
      nodeType.prototype[method] = function (...args) {
        const result = orig?.apply(this, args);
        scheduleSync();
        return result;
      };
    }
  },
});
