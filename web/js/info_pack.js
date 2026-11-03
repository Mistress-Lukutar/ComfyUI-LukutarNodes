/**
 * File:   info_pack.js
 * Brief:  InfoPack node — hide unused key_N/value_N pairs, show one free pair.
 * Author: Mistress-Lukutar
 * Date:   2026-11-03
 * Version: v0.1.0
 *
 * The backend (nodes/checkpoint_bundle.py, InfoPackNode) declares key_1..key_4
 * paired with value_1..value_4.  A pair is claimed when its key is a
 * non-empty string.  This extension shows every claimed pair plus one
 * free trailing pair and hides the rest, mirroring the LoRA List
 * interaction.
 *
 * Served at /extensions/<pack>/js/info_pack.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.InfoPack] extension loaded");

const NODE_NAME = "InfoPack";
const KEY_PREFIX = "key_";
const VALUE_PREFIX = "value_";
const MAX_PAIRS = 4;

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * Visible set = every claimed pair (non-empty key) + the FIRST
 * unclaimed pair (the free one).
 */
function applyVisibility(node) {
  const pairs = [];

  for (let i = 1; i <= MAX_PAIRS; i++) {
    const key = findWidget(node, KEY_PREFIX + i);
    const value = findWidget(node, VALUE_PREFIX + i);
    if (!key) continue;
    pairs.push({ index: i, key, value, claimed: String(key.value).trim() !== "" });
  }

  let freeIndex = -1;
  for (const p of pairs) {
    if (!p.claimed) {
      freeIndex = p.index;
      break;
    }
  }

  for (const p of pairs) {
    const isVisible = p.claimed || p.index === freeIndex;
    for (const w of [p.key, p.value]) {
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
  name: "LukutarNodes.InfoPack",
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

      // Wrap every key_N widget callback so edits trigger re-visibility.
      for (let i = 1; i <= MAX_PAIRS; i++) {
        const w = findWidget(this, KEY_PREFIX + i);
        if (!w || w.__lkInfoPackHooked) continue;
        w.__lkInfoPackHooked = true;
        const callback = w.callback;
        w.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          applyVisibility(this);
          return cbResult;
        };
      }

      // Initial pass — show only the first free pair on a fresh node.
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
    // workflow; re-apply visibility so the correct pairs reappear.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      applyVisibility(this);
      return result;
    };
  },

  setup() {
    scheduleSync();
    // Newer frontends can change text values without invoking
    // widget.callback; the cheap periodic pass keeps visibility in sync.
    setInterval(scheduleSync, 1000);
  },
});
