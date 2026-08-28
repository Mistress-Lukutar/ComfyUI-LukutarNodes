/**
 * File:   variables.js
 * Brief:  Invisible auto-connections between Set Variable / Get Variable
 *         nodes: named workflow variables without dragging wires.
 * Author: Mistress-Lukutar
 * Date:   2026-08-28
 * Version: v0.9.0
 *
 * The backend nodes (nodes/variables.py) are plain pass-throughs; all
 * wiring intelligence lives here. The extension maintains a REAL canvas
 * link from the Set node's `value` output to every matching Get node's
 * `value` input. A real link serializes into the prompt like any other
 * wire, which is the point: ComfyUI itself then owns the execution
 * order and output caching, exported API workflows keep working
 * headless, and a manually wired value input (which always wins) is
 * just a regular link. The managed links are tagged and skipped by a
 * renderLink patch, so nothing shows on the canvas.
 *
 * Why links instead of a prompt rewrite: current ComfyUI frontends
 * have no extension-level queue hooks (beforeQueued/onQueued are gone)
 * and app.graphToPrompt is no longer intercepted by the queue path, so
 * rewriting the serialized prompt is not reliable. Real links need no
 * hooks at all and work with every serializer version.
 *
 * Name resolution runs over the live graph but only considers ACTIVE
 * nodes (mute/bypass are excluded, exactly like the serializer does):
 * several Set nodes may share one name on alternative branches —
 * "generated or loaded" — as long as one branch is active. A name set
 * by two active Sets (or a Set whose input is disconnected) leaves the
 * Get unlinked and flagged ⚠; the backend then raises a descriptive
 * error at queue time.
 *
 * Sync triggers: node create/configure/remove, connection and widget
 * changes. Current litegraph builds fire no callback on mode changes,
 * so a cheap 1s poll keeps the links correct after muting a branch.
 *
 * Served at /extensions/<pack>/js/variables.js, hence the THREE parent
 * hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes] variables extension loaded");

const SET_NAME = "SetVariable";
const GET_NAME = "GetVariable";
const NAME_WIDGET = "var_name";
const VALUE_INPUT = "value";
/** Flag on managed link objects; not serialized — re-tagged after load. */
const LINK_FLAG = "lukutarVariable";
/** LiteGraph.NEVER — ComfyUI "mute"; such nodes never reach the prompt. */
const MODE_NEVER = 2;
/** LiteGraph.BYPASS — ComfyUI "bypass"; serialized away too. */
const MODE_BYPASS = 4;

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

/** Read a link by id across litegraph versions (Map or object _links). */
function getLink(graph, linkId) {
  if (linkId == null || !graph) return null;
  if (graph.getLink) return graph.getLink(linkId);
  if (graph._links instanceof Map) return graph._links.get(linkId) ?? null;
  return graph._links?.[linkId] ?? null;
}

/** The node's trimmed variable name ("" while unset). */
function varNameOf(node) {
  const widget = node.widgets?.find((w) => w.name === NAME_WIDGET);
  return String(widget?.value ?? "").trim();
}

/** Active = not muted and not bypassed. */
function isActive(node) {
  return node.mode !== MODE_NEVER && node.mode !== MODE_BYPASS;
}

/** Index of the `value` input slot, or -1. */
function valueInputIndex(node) {
  return node.inputs?.findIndex((i) => i.name === VALUE_INPUT) ?? -1;
}

/** Type name feeding the node's value input, or null (hides "*"). */
function typeOnInput(node, graph) {
  const idx = valueInputIndex(node);
  if (idx < 0) return null;
  const slot = node.inputs[idx];
  if (slot.link == null) return null;
  const link = getLink(graph, slot.link);
  if (!link) return null;
  const origin = graph.getNodeById?.(link.origin_id);
  const type = origin?.outputs?.[link.origin_slot]?.type;
  return type && type !== "*" ? type : null;
}

function setTitle(node, title) {
  if (node.title !== title) node.title = title;
}

/* ------------------------------------------------------------------ */
/* Link management                                                     */
/* ------------------------------------------------------------------ */

function tagManaged(link) {
  link[LINK_FLAG] = true;
  // Belt and suspenders with the renderLink patch: an alpha-only color
  // keeps the link invisible even if the patch ever stops applying.
  link.color = "rgba(0,0,0,0)";
}

/**
 * The managed link occupying the node's value input, or null. A link
 * coming from a Set Variable output is adopted as managed after a
 * workflow reload (the flag itself is not serialized); any other
 * origin is a manual wire and always wins.
 */
function managedLinkOn(node, valueIdx, graph) {
  const slot = node.inputs?.[valueIdx];
  if (!slot || slot.link == null) return null;
  const link = getLink(graph, slot.link);
  if (!link) return null;
  const origin = graph.getNodeById?.(link.origin_id);
  if (link[LINK_FLAG] || origin?.type === SET_NAME) {
    tagManaged(link);
    return link;
  }
  return null;
}

function dropManaged(node, valueIdx, managed) {
  if (managed) node.disconnectInput?.(valueIdx);
}

/** Decorate one Set node; setsByName maps name -> active Set nodes. */
function syncSetNode(node, graph, setsByName) {
  const name = varNameOf(node);
  const base = name ? `Set · ${name}` : "Set Variable";
  let warning = false;
  if (name && isActive(node) && (setsByName.get(name)?.length ?? 0) > 1) {
    warning = true;
  }
  const type = typeOnInput(node, graph);
  setTitle(node, base + (type ? ` [${type}]` : "") + (warning ? " ⚠" : ""));
}

/** Rewire and decorate one Get node. */
function syncGetNode(node, graph, setsByName) {
  const valueIdx = valueInputIndex(node);
  if (valueIdx < 0) return;
  const name = varNameOf(node);
  const base = name ? `Get · ${name}` : "Get Variable";
  let warning = null;
  let type = null;

  const managed = managedLinkOn(node, valueIdx, graph);
  const slot = node.inputs[valueIdx];
  if (managed == null && slot.link != null) {
    // Manual wire from a non-Set node: keep it untouched.
    type = typeOnInput(node, graph);
  } else {
    const sets = name ? setsByName.get(name) : undefined;
    if (!name) {
      dropManaged(node, valueIdx, managed);
    } else if (!sets || sets.length === 0) {
      dropManaged(node, valueIdx, managed);
      warning = "no active Set";
    } else if (sets.length > 1) {
      dropManaged(node, valueIdx, managed);
      warning = "duplicate name";
    } else {
      const setNode = sets[0];
      const setIdx = valueInputIndex(setNode);
      if (setIdx < 0 || setNode.inputs[setIdx].link == null) {
        dropManaged(node, valueIdx, managed);
        warning = "Set input not connected";
      } else if (managed && managed.origin_id === setNode.id) {
        type = typeOnInput(setNode, graph); // already wired to the right Set
      } else {
        dropManaged(node, valueIdx, managed);
        setNode.connect(0, node, valueIdx);
        const link = managedLinkOn(node, valueIdx, graph);
        if (link) type = typeOnInput(setNode, graph);
        else warning = "failed to connect";
      }
    }
  }
  setTitle(
    node,
    base + (type ? ` [${type}]` : "") + (warning ? ` ⚠ ${warning}` : "")
  );
}

/** Full idempotent pass over the graph. */
function syncVariables() {
  const graph = app.graph;
  const nodes = graph?._nodes;
  if (!nodes) return;

  const setsByName = new Map();
  for (const node of nodes) {
    if (node.type !== SET_NAME || !isActive(node)) continue;
    const name = varNameOf(node);
    if (!name) continue;
    if (!setsByName.has(name)) setsByName.set(name, []);
    setsByName.get(name).push(node);
  }
  for (const node of nodes) {
    if (node.type === SET_NAME) syncSetNode(node, graph, setsByName);
    else if (node.type === GET_NAME) syncGetNode(node, graph, setsByName);
  }
  app.canvas?.setDirty?.(true, true);
}

let syncTimer = null;
/** Coalesce a burst of events into one pass on the next tick. */
function scheduleSync() {
  if (syncTimer != null) return;
  syncTimer = setTimeout(() => {
    syncTimer = null;
    syncVariables();
  }, 0);
}

/* ------------------------------------------------------------------ */
/* Rendering: managed links stay invisible                             */
/* ------------------------------------------------------------------ */

function patchLinkRendering() {
  const proto =
    typeof LGraphCanvas !== "undefined"
      ? LGraphCanvas.prototype
      : app.canvas?.constructor?.prototype;
  if (!proto || proto.lukutarVarsPatched) return;
  const orig = proto.renderLink;
  if (typeof orig !== "function") return;
  proto.lukutarVarsPatched = true;
  proto.renderLink = function (ctx, a, b, link) {
    if (link?.[LINK_FLAG]) return;
    return orig.apply(this, arguments);
  };
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.Variables",
  setup() {
    patchLinkRendering();
    scheduleSync();
    // No node callback fires on mode changes in current litegraph
    // builds; the cheap periodic pass keeps the invisible links
    // correct after mute/bypass of alternative branches.
    setInterval(scheduleSync, 1000);
  },
  nodeCreated() {
    scheduleSync();
  },
  afterConfigureGraph() {
    scheduleSync();
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== SET_NAME && nodeData?.name !== GET_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const widget = this.widgets?.find((w) => w.name === NAME_WIDGET);
      if (widget && !widget.lukutarVarsHooked) {
        widget.lukutarVarsHooked = true;
        const callback = widget.callback;
        widget.callback = function (...args) {
          const cbResult = callback?.apply(this, args);
          scheduleSync();
          return cbResult;
        };
      }
      scheduleSync();
      return result;
    };

    // onRemoved fires before the node leaves the graph; the deferred
    // pass then sees the graph without it.
    for (const method of ["onConfigure", "onConnectionsChange", "onRemoved"]) {
      const orig = nodeType.prototype[method];
      nodeType.prototype[method] = function (...args) {
        const result = orig?.apply(this, args);
        scheduleSync();
        return result;
      };
    }
  },
});
