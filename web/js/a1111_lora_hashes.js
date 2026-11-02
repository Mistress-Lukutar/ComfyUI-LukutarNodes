/**
 * File:   a1111_lora_hashes.js
 * Brief:  A1111LoraHashes node — hide unused lora_N combo widgets, show one free slot.
 * Author: Mistress-Lukutar
 * Date:   2026-11-02
 * Version: v0.1.0
 *
 * The backend (nodes/a1111.py, A1111LoraHashesNode) declares lora_1 .. lora_8
 * combo widgets with "None" as the sentinel for unused slots.  This extension
 * hides every unclaimed slot and shows exactly one free trailing slot so the
 * user can keep adding LoRA inputs without seeing eight blank combos.
 *
 * Served at /extensions/<pack>/js/a1111_lora_hashes.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.A1111LoraHashes] extension loaded");

const NODE_NAME = "A1111LoraHashes";
const PREFIX = "lora_";
const MAX_SLOTS = 8;
const LORA_NONE = "None";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * Apply visibility to all lora_N combo widgets.
 *
 * A slot is "claimed" when its value is a non-empty string and !== LORA_NONE.
 * Visible set = every claimed slot + the FIRST unclaimed slot (the free one).
 * Hidden  = remaining slots, with hidden=true + computeSize=()=>[0,-4].
 */
function applyVisibility(node) {
  const values = [];

  for (let i = 1; i <= MAX_SLOTS; i++) {
    const w = findWidget(node, PREFIX + i);
    if (!w) continue;
    values.push({ index: i, widget: w, value: w.value });
  }

  // Determine which slots are claimed.
  const claimed = new Set();
  for (const v of values) {
    const s = String(v.value).trim();
    if (s && s !== LORA_NONE) {
      claimed.add(v.index);
    }
  }

  // Find the first unclaimed slot to serve as the free trailing input.
  let freeIndex = -1;
  for (let i = 1; i <= MAX_SLOTS; i++) {
    if (!claimed.has(i)) {
      freeIndex = i;
      break;
    }
  }

  // Apply visibility: visible slots get their computeSize removed; hidden ones
  // get hidden=true + a tiny computeSize so they don't take canvas space.
  for (const v of values) {
    const isVisible = claimed.has(v.index) || v.index === freeIndex;

    if (isVisible) {
      v.widget.hidden = false;
      delete v.widget.computeSize;
    } else {
      v.widget.hidden = true;
      v.widget.computeSize = () => [0, -4];
    }
  }
}

let syncTimer = null;
/** Coalesce a burst of events into one pass on the next tick. */
function scheduleSync() {
  if (syncTimer != null) return;
  syncTimer = setTimeout(() => {
    syncTimer = null;
    (app.graph?._nodes ?? []).forEach((node) => {
      if (node.type === NODE_NAME) applyVisibility(node);
    });
  }, 0);
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.A1111LoraHashes",
  afterConfigureGraph() {
    // After a workflow load, re-apply visibility so hidden slots stay hidden.
    (app.graph?._nodes ?? []).forEach((node) => {
      if (node.type === NODE_NAME) applyVisibility(node);
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      // Wrap every lora_N widget callback so changes trigger re-visibility.
      for (let i = 1; i <= MAX_SLOTS; i++) {
        const w = findWidget(this, PREFIX + i);
        if (!w || w.__lkA1111LoraHashesHooked) continue;
        w.__lkA1111LoraHashesHooked = true;
        const callback = w.callback;
        w.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          applyVisibility(this);
          return cbResult;
        };
      }

      // Initial pass — hide unused slots on a fresh node.
      applyVisibility(this);

      // Snap the node height down to what the visible widgets need.
      const fitted = this.computeSize(this.size[0]);
      if (fitted[1] < this.size[1]) this.size[1] = fitted[1];

      return result;
    };

    // onConfigure fires after widgets_values are restored from a workflow;
    // re-apply visibility so the correct slots become visible/hidden again.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      applyVisibility(this);
      return result;
    };
  },

  setup() {
    scheduleSync();
    // Newer frontends can change combo values without invoking widget.callback;
    // the cheap periodic pass keeps slot visibility in sync.
    setInterval(scheduleSync, 1000);
  },
});
