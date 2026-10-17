/**
 * File:   bundle_variables.js
 * Brief:  Set Bundle / Get Bundle: one named bundle carrying several
 *         heterogeneously typed slots, wirelessly readable elsewhere.
 * Author: Mistress-Lukutar
 * Date:   2026-10-17
 * Version: v0.12.0
 *
 * The backend nodes (nodes/bundle_variables.py) are typed pass-throughs
 * that receive their slot values via **kwargs; all wiring intelligence
 * lives here, exactly like variables.js does for the single-value pair.
 *
 * Slot growth copies Impact Pack's SEGS Concat pattern: a Set Bundle
 * always shows one free "+ add" input; connecting a wire onto it fixes
 * that slot's type to the source and auto-names it (model, vae, clip,
 * model_2 — derived from the type), then spawns the next free input.
 * Disconnecting an auto-named slot collapses it again; a slot renamed
 * via the node menu (Rename slot) survives disconnects, and so does a
 * connected fallback slot_N. Every named slot is mirrored to an output
 * of the same name, so the Set node is a real bus: source -> Set ->
 * Get -> consumer are all real canvas links and ComfyUI itself owns
 * execution order, caching and API export.
 *
 * A Get Bundle resolves its bundle_name against the active Set Bundles
 * (mute/bypass excluded, like the serializer): with exactly one match
 * it adopts that Set's slot spec — one named input and one typed
 * output per slot — and maintains an invisible real link per slot,
 * tagged and skipped by the shared renderLink patch. A manual wire
 * into a Get input always wins. Several Sets may share one name on
 * alternative branches; two simultaneously active ones leave the Get
 * unlinked and flagged, and the backend raises at queue time.
 *
 * The `slots` spec widget ("name:TYPE,..." ) on both nodes is kept in
 * sync here and serialized into the workflow — the backends read it
 * for loud per-slot errors — but hidden from the UI: it is internal
 * state, the sockets themselves are the interface. Without the web
 * assets the nodes cannot be wired from the UI (the extension owns
 * every socket); hand-written API prompts may pass slot links of any
 * name and order them via the `slots` value or natural slot_N order.
 *
 * Frontend gotcha this extension is built around: current ComfyUI
 * frontends keep EVERY widget as an entry in node.inputs (native
 * widget ⇄ socket wiring; entries carry a `widget` field). Slot logic
 * must therefore filter to real sockets only — widget entries and the
 * reserved bundle_name/slots names are never slots — and rebuilds must
 * never remove or recreate widget entries, or the node loses its
 * widgets. Stale slot_N-named sockets from old saves are collapsed
 * while unconnected.
 *
 * Sync triggers mirror variables.js: node create/configure/remove,
 * connection and widget changes, plus a cheap 1s poll because current
 * litegraph builds fire no callback on mode changes. Each node's sync
 * is try/caught: one broken node must not freeze the healing of the
 * rest.
 *
 * Served at /extensions/<pack>/js/, hence the THREE parent hops to
 * /scripts/app.js. Shared link helpers come from variables.js (one
 * module instance, so the renderLink patch and poll stay global).
 */

import { app } from "../../../scripts/app.js";
import {
  getLink,
  isActive,
  patchLinkRendering,
  tagManaged,
} from "./variables.js";

console.info("[LukutarNodes] bundle variables extension loaded");

const SET_TYPE = "SetBundle";
const GET_TYPE = "GetBundle";
const NAME_WIDGET = "bundle_name";
const SPEC_WIDGET = "slots";
/** The always-present free input that grows the next slot on connect. */
const FREE_SLOT = "+ add";
/** Keep in sync with MAX_SLOTS in nodes/bundle_variables.py. */
const MAX_SLOTS = 16;
/** Widget names that are node inputs, never slot names. */
const RESERVED_NAMES = new Set([NAME_WIDGET, SPEC_WIDGET]);
/** Fallback input names used by hand-written API prompts. */
const FALLBACK_RE = /^slot_(\d+)$/;
/** Types offered by the "Add slot" menu (a wire can add any other). */
const MENU_TYPES = [
  "MODEL",
  "VAE",
  "CLIP",
  "CONDITIONING",
  "CONTROL_NET",
  "IMAGE",
  "LATENT",
  "MASK",
  "UPSCALE_MODEL",
  "STRING",
  "INT",
  "FLOAT",
  "*",
];

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

const isFree = (slot) => slot?.name === FREE_SLOT;
const isFallbackName = (name) => FALLBACK_RE.test(name);
const isReservedName = (name) => RESERVED_NAMES.has(name);
/**
 * True for the widget-backed entries current frontends keep inside
 * node.inputs (native widget ⇄ socket wiring). They are the node's
 * widgets, never slots — and they must never be removed, or the node
 * loses its widgets.
 */
const isWidgetEntry = (slot) => slot?.widget != null;

/** The node's trimmed bundle name ("" while unset). */
function bundleNameOf(node) {
  const widget = node.widgets?.find((w) => w.name === NAME_WIDGET);
  return String(widget?.value ?? "").trim();
}

/** The disabled spec widget, kept in sync with the actual slots. */
function specWidget(node) {
  return node.widgets?.find((w) => w.name === SPEC_WIDGET);
}

/**
 * The real slot sockets: widget entries, the free "+ add" input and
 * reserved widget names excluded. `index` is the slot's position in
 * the full node.inputs array (link/connect calls need that one).
 */
function socketEntries(node) {
  const entries = [];
  (node.inputs ?? []).forEach((slot, index) => {
    if (isWidgetEntry(slot) || isFree(slot) || isReservedName(slot.name)) {
      return;
    }
    entries.push({ slot, index });
  });
  return entries;
}

function specOf(node) {
  return socketEntries(node).map(({ slot }) => ({
    name: slot.name,
    type: slot.type,
  }));
}

function specToText(spec) {
  return spec.map((slot) => `${slot.name}:${slot.type}`).join(",");
}

/** Mirror the spec into the hidden slots widget (no-op when equal). */
function syncSpecWidget(node, spec) {
  const widget = specWidget(node);
  const text = specToText(spec);
  if (widget && widget.value !== text) widget.value = text;
}

/** "CONTROL_NET" -> "control_net"; sanitized for use as a slot name. */
function nameBase(type) {
  const base = String(type ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return base || "value";
}

/** Unique auto name for a type, e.g. model, model_2, model_3. */
function autoName(type, node) {
  const taken = new Set(socketEntries(node).map(({ slot }) => slot.name));
  const base = nameBase(type);
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base}_${i}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/** True when the name still looks auto-derived from the slot's type. */
function isAutoNamed(slot) {
  if (isFallbackName(slot.name)) return false;
  const base = nameBase(slot.type);
  return slot.name === base || slot.name.startsWith(`${base}_`);
}

/** Toast when available, console otherwise. */
function toast(message) {
  app.extensionManager?.toast?.add?.({
    severity: "warn",
    summary: "Lukutar Bundle",
    detail: message,
    life: 4000,
  });
  console.warn(`[LukutarNodes] ${message}`);
}

/** Configure/load/paste rebuild slots directly; never react then. */
function duringGraphLoad() {
  const stack = new Error().stack ?? "";
  return (
    stack.includes("loadGraphData") ||
    stack.includes("pasteFromClipboard") ||
    stack.includes("Subgraph.configure") ||
    stack.includes("convertToSubgraph")
  );
}

/** Full-array index of the input called `name` on `node`, or -1. */
function inputIndexByName(node, name) {
  return node.inputs?.findIndex((s) => s.name === name) ?? -1;
}

/** Full-array index of the output called `name` on `node`, or -1. */
function outputIndexByName(node, name) {
  return node.outputs?.findIndex((o) => o.name === name) ?? -1;
}

/**
 * Make the node's outputs mirror `spec` (one output per slot, same
 * order). No-op while they already match, so downstream links keep
 * their identity while the bundle is untouched.
 */
function rebuildOutputs(node, spec) {
  const outputs = node.outputs ?? [];
  const same =
    outputs.length === spec.length &&
    spec.every(
      (slot, i) =>
        outputs[i]?.name === slot.name && outputs[i]?.type === slot.type
    );
  if (same) return;
  for (let i = outputs.length - 1; i >= spec.length; i--) {
    if (node.outputs[i]?.links?.length) node.disconnectOutput(i);
    node.removeOutput(i);
  }
  spec.forEach((slot, i) => {
    const output = node.outputs[i];
    if (!output) node.addOutput(slot.name, slot.type);
    else {
      output.name = slot.name;
      output.type = slot.type;
    }
  });
}

/**
 * Make the Get node's real input sockets mirror `spec`. Widget-backed
 * entries are skipped entirely (removing one would orphan the node's
 * widgets). A full rebuild fires when the sockets diverge (slot added,
 * removed or renamed on the Set side) and the sync pass then relinks
 * the managed wires.
 */
function rebuildGetInputs(node, spec) {
  const sockets = socketEntries(node);
  const same =
    sockets.length === spec.length &&
    spec.every(
      (slot, i) =>
        sockets[i]?.slot.name === slot.name &&
        sockets[i]?.slot.type === slot.type
    );
  if (same) return;
  for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
    const input = node.inputs[i];
    if (isWidgetEntry(input)) continue;
    if (input.link != null) node.disconnectInput(i);
    node.removeInput(i);
  }
  spec.forEach((slot) => node.addInput(slot.name, slot.type));
}

/**
 * Keep exactly one free trailing input on a Set Bundle and shed stale
 * unconnected slot_N sockets (leftovers from hand-wired or older
 * saves; while unconnected they carry no information).
 */
function normalizeSetInputs(node) {
  const inputs = node.inputs ?? [];
  for (let i = inputs.length - 1; i >= 0; i--) {
    const slot = inputs[i];
    if (isWidgetEntry(slot) || isReservedName(slot.name)) continue;
    if (isFallbackName(slot.name) && slot.link == null) {
      node.removeInput(i);
    }
  }
  const frees = [];
  inputs.forEach((slot, index) => {
    if (isFree(slot)) frees.push(index);
  });
  // Removing only keeps earlier indices valid, so drop all but the
  // last free slot walking backwards.
  for (let i = frees.length - 2; i >= 0; i--) {
    const index = frees[i];
    if (inputs[index].link != null) node.disconnectInput(index);
    node.removeInput(index);
  }
  if (frees.length === 0) node.addInput(FREE_SLOT, "*");
}

/* ------------------------------------------------------------------ */
/* Slot surgery (Set node menus + connection events)                   */
/* ------------------------------------------------------------------ */

function structurallyChanged(node) {
  syncSpecWidget(node, specOf(node));
  scheduleSync();
}

function removeSlot(node, entry) {
  const { slot, index } = entry;
  if (slot.link != null) node.disconnectInput(index);
  node.removeInput(index);
  structurallyChanged(node);
}

function renameSlot(node, entry) {
  const slot = entry.slot;
  let next;
  try {
    next = window.prompt("Slot name:", slot.name);
  } catch {
    toast("Renaming needs the browser prompt dialog");
    return;
  }
  next = String(next ?? "").trim();
  if (!next || next === slot.name) return;
  if (/[,;:]/.test(next)) {
    toast("Slot names must not contain commas, semicolons or colons");
    return;
  }
  if (isReservedName(next) || isFallbackName(next)) {
    toast(`"${next}" is reserved`);
    return;
  }
  if (socketEntries(node).some(({ slot: other }) => other !== slot && other.name === next)) {
    toast(`Slot name "${next}" is already used`);
    return;
  }
  slot.name = next;
  structurallyChanged(node);
}

function addSlot(node, type) {
  if (socketEntries(node).length >= MAX_SLOTS) {
    toast(`A bundle holds at most ${MAX_SLOTS} slots`);
    return;
  }
  const name = autoName(type, node);
  const freeIndex = node.inputs.findIndex(isFree);
  if (freeIndex >= 0) node.removeInput(freeIndex);
  node.addInput(name, type);
  node.addInput(FREE_SLOT, "*");
  structurallyChanged(node);
}

/** Connection events on the Set node: grow and collapse slots. */
function onSetConnectionsChange(node, type, index, connected, linkInfo) {
  if (!linkInfo || type !== 1) return;
  if (duringGraphLoad()) return;
  const slot = node.inputs?.[index];
  if (!slot || isWidgetEntry(slot) || isReservedName(slot.name)) return;

  if (connected) {
    if (!isFree(slot)) {
      // Wiring an existing named slot; nothing structural.
      return;
    }
    slot.type = String(linkInfo.type || "*");
    slot.name = autoName(slot.type, node);
    if (socketEntries(node).length < MAX_SLOTS) node.addInput(FREE_SLOT, "*");
  } else {
    if (isFree(slot) || isFallbackName(slot.name)) return;
    if (!isAutoNamed(slot)) return; // renamed slots survive disconnects
    node.removeInput(index);
  }
  structurallyChanged(node);
}

/** Connection events on the Get node: the sync pass reconciles. */
function onGetConnectionsChange() {
  scheduleSync();
}

/* ------------------------------------------------------------------ */
/* Sync pass                                                           */
/* ------------------------------------------------------------------ */

function syncSetNode(node, setsByName) {
  normalizeSetInputs(node);
  const spec = specOf(node);
  syncSpecWidget(node, spec);
  rebuildOutputs(node, spec);

  let warning = "";
  const name = bundleNameOf(node);
  if (name && isActive(node) && (setsByName.get(name)?.length ?? 0) > 1) {
    warning = " ⚠ duplicate";
  }
  const unconnected = socketEntries(node)
    .filter(({ slot }) => slot.link == null)
    .map(({ slot }) => slot.name);
  if (unconnected.length) warning += ` ⚠ unconnected: ${unconnected.join(", ")}`;
  const base = name ? `Set Bundle · ${name} (${spec.length})` : "Set Bundle";
  if (node.title !== base + warning) node.title = base + warning;
}

/** True when two specs have identical names and types in order. */
function sameSpec(a, b) {
  return (
    a.length === b.length &&
    a.every((slot, i) => slot.name === b[i].name && slot.type === b[i].type)
  );
}

/** Rewire and decorate one Get node against the active Sets. */
function syncGetNode(node, graph, setsByName) {
  const name = bundleNameOf(node);
  const sets = name ? setsByName.get(name) : undefined;
  const active = sets?.length === 1 ? sets[0] : null;

  if (active) {
    const target = specOf(active);
    const drifted =
      !sameSpec(specOf(node), target) ||
      (specWidget(node)?.value ?? "") !== specToText(target);
    if (drifted) {
      syncSpecWidget(node, target);
      rebuildGetInputs(node, target);
    }
  }

  const spec = specOf(node);
  rebuildOutputs(node, spec);

  let warning = "";
  if (!name) {
    warning = " ⚠ name it";
  } else if (!sets || sets.length === 0) {
    warning = " ⚠ no active Set";
  } else if (sets.length > 1) {
    warning = " ⚠ duplicate name";
  }

  // spec[k] corresponds to the k-th real socket; `entry.index` is its
  // position in the full node.inputs array (widget entries included).
  const entries = socketEntries(node);
  for (let k = 0; k < entries.length; k++) {
    const { slot: input, index } = entries[k];
    const existing =
      input.link != null ? getLink(graph, input.link) : null;
    const origin = existing
      ? graph.getNodeById?.(existing.origin_id)
      : null;
    if (origin && origin.type !== SET_TYPE) continue; // manual wire wins
    if (!active) {
      if (existing) node.disconnectInput(index);
      continue;
    }
    // Managed slot: (re)link from the active Set's matching output.
    const slotName = spec[k]?.name;
    const outIndex = outputIndexByName(active, slotName);
    const source =
      outIndex >= 0 ? active.inputs[inputIndexByName(active, slotName)] : null;
    if (!source || source.link == null) {
      if (existing) node.disconnectInput(index);
      warning += ` ⚠ Set slot not connected: ${slotName}`;
      continue;
    }
    if (existing && origin === active && existing.origin_slot === outIndex) {
      tagManaged(existing); // already wired to the right Set output
      continue;
    }
    if (existing) node.disconnectInput(index);
    active.connect(outIndex, node, index);
    const link = input.link != null ? getLink(graph, input.link) : null;
    if (link) tagManaged(link);
    else warning += ` ⚠ failed to link: ${slotName}`;
  }

  const base = name ? `Get Bundle · ${name} (${spec.length})` : "Get Bundle";
  if (node.title !== base + warning) node.title = base + warning;
}

/** Full idempotent pass over the graph. */
function syncBundles() {
  const graph = app.graph;
  const nodes = graph?._nodes;
  if (!nodes) return;

  const setsByName = new Map();
  for (const node of nodes) {
    if (node.type !== SET_TYPE || !isActive(node)) continue;
    const name = bundleNameOf(node);
    if (!name) continue;
    if (!setsByName.has(name)) setsByName.set(name, []);
    setsByName.get(name).push(node);
  }
  for (const node of nodes) {
    try {
      if (node.type === SET_TYPE) syncSetNode(node, setsByName);
      else if (node.type === GET_TYPE) syncGetNode(node, graph, setsByName);
    } catch (error) {
      // One broken node must not freeze the healing of the rest.
      console.warn("[LukutarNodes] bundle sync failed for a node", error);
    }
  }
  app.canvas?.setDirty?.(true, true);
}

let syncTimer = null;
/** Coalesce a burst of events into one pass on the next tick. */
function scheduleSync() {
  if (syncTimer != null) return;
  syncTimer = setTimeout(() => {
    syncTimer = null;
    syncBundles();
  }, 0);
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.BundleVariables",
  setup() {
    // Hide the per-slot managed links (shared idempotent patch from
    // variables.js; its 1s poll reasons apply here too).
    patchLinkRendering();
    scheduleSync();
    setInterval(scheduleSync, 1000);
  },
  nodeCreated() {
    scheduleSync();
  },
  afterConfigureGraph() {
    scheduleSync();
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== SET_TYPE && nodeData?.name !== GET_TYPE) return;
    const isSet = nodeData.name === SET_TYPE;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const spec = specWidget(this);
      if (spec) {
        spec.disabled = true;
        // Hidden from the UI, still serialized: the sockets are the
        // interface, the spec is internal state.
        spec.computeSize = () => [0, -4];
      }
      const nameWidget = this.widgets?.find(
        (w) => w.name === NAME_WIDGET
      );
      if (nameWidget && !nameWidget.lukutarBundleHooked) {
        nameWidget.lukutarBundleHooked = true;
        const callback = nameWidget.callback;
        nameWidget.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          scheduleSync();
          return cbResult;
        };
      }
      // The backend shape ships MAX_SLOTS generic outputs; the sync
      // pass rebuilds them from the spec within a tick.
      while (this.outputs.length) {
        this.removeOutput(this.outputs.length - 1);
      }
      if (isSet) normalizeSetInputs(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      const spec = specWidget(this);
      if (spec) {
        spec.disabled = true;
        spec.computeSize = () => [0, -4];
      }
      if (isSet) normalizeSetInputs(this);
      scheduleSync();
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
      const result = onConnectionsChange?.apply(this, arguments);
      if (isSet) onSetConnectionsChange(this, type, index, connected, linkInfo);
      else onGetConnectionsChange(this, type, index, connected, linkInfo);
      return result;
    };

    if (isSet) {
      const onConnectInput = nodeType.prototype.onConnectInput;
      nodeType.prototype.onConnectInput = function (targetIndex) {
        if (onConnectInput?.apply(this, arguments) === false) return false;
        const slot = this.inputs?.[targetIndex];
        if (
          slot &&
          !isWidgetEntry(slot) &&
          isFree(slot) &&
          socketEntries(this).length >= MAX_SLOTS
        ) {
          toast(`A bundle holds at most ${MAX_SLOTS} slots`);
          return false;
        }
        return true;
      };

      const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        const result = getExtraMenuOptions?.apply(this, arguments);
        const node = this;
        for (const entry of socketEntries(node)) {
          options.push({
            content: `Rename slot "${entry.slot.name}"`,
            callback: () => renameSlot(node, entry),
          });
          options.push({
            content: `Remove slot "${entry.slot.name}"`,
            callback: () => removeSlot(node, entry),
          });
        }
        options.push({
          content: "Add slot",
          submenu: {
            options: MENU_TYPES.map((type) => ({
              content: type === "*" ? "* (any)" : type,
              callback: () => addSlot(node, type),
            })),
          },
        });
        return result;
      };
    }

    // onRemoved fires before the node leaves the graph; the deferred
    // pass then sees the graph without it.
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function (...args) {
      const result = onRemoved?.apply(this, args);
      scheduleSync();
      return result;
    };
  },
});
