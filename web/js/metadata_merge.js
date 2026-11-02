/**
 * File:   metadata_merge.js
 * Brief:  MergeMetadata node — dynamic input slot growth (one fixed type).
 * Author: Mistress-Lukutar
 * Date:   2026-11-02
 * Version: v0.1.0
 *
 * The backend declares zero inputs; this extension grows real input
 * sockets named meta_1, meta_2, … (max 16) all of type LUKUTAR_METADATA.
 * A single free "+ add" input is always present — connecting a wire onto
 * it claims the slot and spawns the next free input. Disconnecting an
 * auto-named slot collapses it, then normalize ensures exactly one free
 * slot remains.
 *
 * Served at /extensions/<pack>/js/metadata_merge.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.MetadataMerge] extension loaded");

/** Node types that share the metadata-merge slot-growth behaviour. */
const NODE_NAMES = ["MergeMetadata", "A1111Parameters"];

/** Prefix for auto-named metadata input sockets. */
const INPUT_PREFIX = "meta_";
/** Maximum number of metadata slots (keep in sync with backend). */
const MAX_SLOTS = 16;
/** Socket type for every metadata input. */
const SLOT_TYPE = "LUKUTAR_METADATA";
/** The always-present free input that grows the next slot on connect. */
const FREE_SLOT = "+ add";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function isFree(input) {
  return input?.name === FREE_SLOT;
}

/** Real socket inputs only — skip widget entries (input.widget != null). */
function slotInputs(node) {
  return (node.inputs ?? []).filter((input) => input.widget == null);
}

/** Set of N parsed from meta_N names on real sockets. */
function slotNumbers(node) {
  const nums = new Set();
  for (const input of slotInputs(node)) {
    const m = /^meta_(\d+)$/.exec(input.name);
    if (m) nums.add(Number(m[1]));
  }
  return nums;
}

/**
 * Normalize: remove every unconnected auto-named meta_N input, then
 * ensure exactly one trailing free "+ add" input exists. Never touches
 * widget entries or exceeds the node's output sockets.
 */
function normalize(node) {
  const inputs = node.inputs ?? [];

  // Remove unconnected auto-named meta_N inputs (stale from old saves).
  for (let i = inputs.length - 1; i >= 0; i--) {
    const input = inputs[i];
    if (input.widget != null) continue; // skip widget entries
    if (isFree(input)) continue;
    if (!input.name.startsWith(INPUT_PREFIX)) continue;
    if (input.link == null) {
      node.removeInput(i);
    }
  }

  // Ensure exactly one free "+ add" input exists.
  let hasFree = false;
  for (let i = 0; i < inputs.length; i++) {
    if (isFree(inputs[i])) {
      hasFree = true;
      break;
    }
  }
  if (!hasFree) {
    node.addInput(FREE_SLOT, SLOT_TYPE);
  }
}

/** Find the full-array index of the free "+ add" input, or -1. */
function findFreeIndex(node) {
  return (node.inputs ?? []).findIndex(isFree);
}

/* ------------------------------------------------------------------ */
/* Connection events                                                   */
/* ------------------------------------------------------------------ */

function handleSlotConnection(node, type, index, connected, linkInfo) {
  if (!linkInfo || type !== 1) return; // type 1 = input connection

  const slot = node.inputs?.[index];
  if (!slot || slot.widget != null) return; // skip widget entries

  if (connected) {
    if (!isFree(slot)) return; // wiring an existing named slot — nothing structural

    // Claim the lowest unused meta_N number.
    const nums = slotNumbers(node);
    let n;
    for (n = 1; n <= MAX_SLOTS; n++) {
      if (!nums.has(n)) break;
    }
    if (n > MAX_SLOTS) return; // reached the limit

    slot.name = `${INPUT_PREFIX}${n}`;
    slot.type = SLOT_TYPE;

    // Spawn the next free "+ add" input while there is still room for
    // another slot AFTER this claim (named slots = nums.size + 1 now).
    if (nums.size + 1 < MAX_SLOTS) {
      node.addInput(FREE_SLOT, SLOT_TYPE);
    }
  } else {
    // A meta_N lost its link — collapse it, then normalize.
    const m = /^meta_(\d+)$/.exec(slot.name);
    if (m) {
      const idx = node.inputs.indexOf(slot);
      if (idx >= 0) node.removeInput(idx);
      normalize(node);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.MergeMetadata",
  afterConfigureGraph() {
    // After a workflow load, clean up stale unconnected slots and ensure
    // the free "+ add" input exists on every matching node.
    (app.graph?._nodes ?? []).forEach((node) => {
      if (NODE_NAMES.includes(node.type)) normalize(node);
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.includes(nodeData?.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      normalize(this);
      return result;
    };

    // Chain connection-change handler to grow/collapse slots.
    const origConnChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
      const result = origConnChange?.apply(this, arguments);
      handleSlotConnection(this, type, index, connected, linkInfo);
      return result;
    };
  },
});
