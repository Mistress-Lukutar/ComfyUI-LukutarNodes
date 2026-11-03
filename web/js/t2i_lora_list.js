/**
 * File:   t2i_lora_list.js
 * Brief:  T2ILoraList node — hide unused lora_N/weight_N pairs, show one free pair.
 * Author: Mistress-Lukutar
 * Date:   2026-11-03
 * Version: v0.1.0
 *
 * The backend (nodes/t2i.py, T2ILoraListNode) declares lora_1 .. lora_8
 * combo widgets paired with weight_1 .. weight_8 floats.  This extension
 * hides every unclaimed pair and shows exactly one free trailing pair so
 * the user can keep adding LoRAs without seeing eight blank rows.
 *
 * Served at /extensions/<pack>/js/t2i_lora_list.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.T2ILoraList] extension loaded");

const NODE_NAME = "T2ILoraList";
const LORA_PREFIX = "lora_";
const WEIGHT_PREFIX = "weight_";
const MAX_SLOTS = 8;
const LORA_NONE = "None";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * A slot is "claimed" when its combo value is a non-empty string other
 * than LORA_NONE.  Visible set = every claimed pair + the FIRST
 * unclaimed pair (the free one).
 */
function applyVisibility(node) {
  const slots = [];

  for (let i = 1; i <= MAX_SLOTS; i++) {
    const lora = findWidget(node, LORA_PREFIX + i);
    const weight = findWidget(node, WEIGHT_PREFIX + i);
    if (!lora) continue;
    slots.push({ index: i, lora, weight, value: lora.value });
  }

  const claimed = new Set();
  for (const s of slots) {
    const v = String(s.value).trim();
    if (v && v !== LORA_NONE) {
      claimed.add(s.index);
    }
  }

  let freeIndex = -1;
  for (let i = 1; i <= MAX_SLOTS; i++) {
    if (!claimed.has(i)) {
      freeIndex = i;
      break;
    }
  }

  for (const s of slots) {
    const isVisible = claimed.has(s.index) || s.index === freeIndex;
    for (const w of [s.lora, s.weight]) {
      if (!w) continue;
      if (isVisible) {
        w.hidden = false;
        delete w.computeSize;
      } else {
        w.hidden = true;
        w.computeSize = () => [0, -4];
      }
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
  name: "LukutarNodes.T2ILoraList",
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
        const w = findWidget(this, LORA_PREFIX + i);
        if (!w || w.__lkT2ILoraListHooked) continue;
        w.__lkT2ILoraListHooked = true;
        const callback = w.callback;
        w.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          applyVisibility(this);
          return cbResult;
        };
      }

      // Initial pass — hide unused pairs on a fresh node.
      applyVisibility(this);

      // computeSize() must be called no-arg; passing the width as a
      // number throws in strict mode and aborts node creation.
      const fitted = this.computeSize();
      if (Array.isArray(fitted) && fitted[1] < this.size[1]) {
        this.size[1] = fitted[1];
      }

      return result;
    };

    // onConfigure fires after widgets_values are restored from a
    // workflow; re-apply visibility so the correct slots reappear.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      applyVisibility(this);
      return result;
    };
  },

  setup() {
    scheduleSync();
    // Newer frontends can change combo values without invoking
    // widget.callback; the cheap periodic pass keeps visibility in sync.
    setInterval(scheduleSync, 1000);
  },
});
