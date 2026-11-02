/**
 * File:   metadata_blocks.js
 * Brief:  MetadataBlocks node — dynamically show/hide name/value pairs
 *         based on the num_blocks parameter.
 * Author: Mistress-Lukutar
 * Date:   2026-11-02
 * Version: v0.1.0
 *
 * The backend (nodes/metadata.py, MetadataBlocksNode) always declares
 * 16 name/value pairs; this extension only controls their visibility
 * on the canvas.
 * Pairs beyond num_blocks are hidden with the pack's established idiom
 * (hidden + computeSize → [0, -4]) so they still serialize correctly
 * and never take a slot in widgets_values.
 *
 * Served at /extensions/<pack>/js/metadata_blocks.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.MetadataBlocks] extension loaded");

const NODE_NAME = "MetadataBlocks";
const NUM_BLOCKS_WIDGET = "num_blocks";
const MAX_BLOCKS = 16;

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * Apply visibility to all name/value pair widgets based on num_blocks.
 * Visible pairs: delete computeSize so they render normally.
 * Hidden pairs: set hidden=true + computeSize=()=>[0,-4] (pack idiom).
 */
function applyVisibility(node) {
  const numBlocksWidget = findWidget(node, NUM_BLOCKS_WIDGET);
  const count = Math.min(16, Math.max(1, numBlocksWidget?.value ?? 1));

  for (let i = 1; i <= MAX_BLOCKS; i++) {
    const nameW = findWidget(node, `name_${i}`);
    const valueW = findWidget(node, `value_${i}`);
    const visible = i <= count;

    if (visible) {
      if (nameW) {
        nameW.hidden = false;
        delete nameW.computeSize;
      }
      if (valueW) {
        valueW.hidden = false;
        delete valueW.computeSize;
      }
    } else {
      if (nameW) {
        nameW.hidden = true;
        nameW.computeSize = () => [0, -4];
      }
      if (valueW) {
        valueW.hidden = true;
        valueW.computeSize = () => [0, -4];
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
  name: "LukutarNodes.MetadataBlocks",
  afterConfigureGraph() {
    // Re-apply visibility after a workflow load so hidden pairs stay
    // hidden when num_blocks was restored from the saved widgets_values.
    (app.graph?._nodes ?? []).forEach((node) => {
      if (node.type === NODE_NAME) applyVisibility(node);
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      // Attach callback to num_blocks so visibility re-applies when changed.
      const numBlocksWidget = findWidget(this, NUM_BLOCKS_WIDGET);
      if (numBlocksWidget && !numBlocksWidget.__lkMetadataBlocksHooked) {
        numBlocksWidget.__lkMetadataBlocksHooked = true;
        const callback = numBlocksWidget.callback;
        numBlocksWidget.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          applyVisibility(this);
          return cbResult;
        };
      }

      // Initial pass — show the default number of blocks.
      applyVisibility(this);

      // The node is constructed at full height (all 16 pairs declared);
      // snap the initial height down to what the visible widgets need.
      // computeSize(out) expects a 2-slot output array; passing the
      // width as a number throws in strict mode and aborts node
      // creation entirely.
      const fitted = this.computeSize();
      if (Array.isArray(fitted) && fitted[1] < this.size[1]) {
        this.size[1] = fitted[1];
      }

      return result;
    };

    // onConfigure fires after widgets_values are restored from a workflow;
    // re-apply visibility so the correct pairs become visible again.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      applyVisibility(this);
      return result;
    };
  },

  setup() {
    scheduleSync();
    // Newer frontends can change INT widget values without invoking
    // widget.callback; the cheap periodic pass keeps the visible pair
    // count in sync, same as group_switch.js.
    setInterval(scheduleSync, 1000);
  },
});
