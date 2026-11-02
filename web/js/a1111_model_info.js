/**
 * File:   a1111_model_info.js
 * Brief:  A1111ModelInfo node — refill model_name combo when model_type changes.
 * Author: Mistress-Lukutar
 * Date:   2026-11-02
 * Version: v0.1.0
 *
 * The backend (nodes/a1111.py, A1111ModelInfoNode) ships a default model_name
 * list for "Checkpoint". When the user switches type to VAE/LoRA/Embedding this
 * extension fetches /models/<folder> and refills the combo.
 *
 * Served at /extensions/<pack>/js/a1111_model_info.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.info("[LukutarNodes.A1111ModelInfo] extension loaded");

const NODE_NAME = "A1111ModelInfo";
const MODEL_TYPE_WIDGET = "model_type";
const MODEL_NAME_WIDGET = "model_name";

/** Map model type labels to the corresponding ComfyUI folder slug. */
const TYPE_TO_FOLDER = {
  Checkpoint: "checkpoints",
  VAE: "vae",
  LoRA: "loras",
  Embedding: "embeddings",
};

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * Fetch the model list for a given type and refill the model_name combo.
 * Returns true on success (list was updated), false otherwise.
 */
async function refillChoices(node, modelType) {
  const folder = TYPE_TO_FOLDER[modelType];
  if (!folder) return false;

  try {
    const resp = await api.fetchApi("/models/" + folder);
    if (!resp.ok) {
      console.warn("[LukutarNodes.A1111ModelInfo] HTTP " + resp.status + " for /models/" + folder);
      return false;
    }
    const list = await resp.json();
    if (!Array.isArray(list)) {
      console.warn("[LukutarNodes.A1111ModelInfo] /models/" + folder + " did not return an array");
      return false;
    }

    // Defensive: keep only strings.
    const valid = list.filter((e) => typeof e === "string");

    const nameWidget = findWidget(node, MODEL_NAME_WIDGET);
    if (!nameWidget) {
      console.warn("[LukutarNodes.A1111ModelInfo] model_name widget not found");
      return false;
    }

    nameWidget.options.values = valid;
    if (valid.length > 0 && !valid.includes(nameWidget.value)) {
      nameWidget.value = valid[0];
    }
    node.setDirtyCanvas?.(true, true);
    return true;
  } catch (err) {
    console.warn("[LukutarNodes.A1111ModelInfo] Error refilling choices for " + modelType + ": " + err.message);
    return false;
  }
}

/**
 * Sync marker: the model_type value that was used to fill choices.
 * On first boot (marker undefined) we just store without refilling
 * — the backend already ships the checkpoint list matching default type.
 */
function sync(node) {
  const typeWidget = findWidget(node, MODEL_TYPE_WIDGET);
  if (!typeWidget) return;

  const storedType = typeWidget.__lkModelType;
  const currentType = typeWidget.value;

  // First sync ever: store marker without refilling (backend already has Checkpoint list).
  if (storedType === undefined) {
    typeWidget.__lkModelType = currentType;
    return;
  }

  // Type changed since last refill → re-fetch.
  if (currentType !== storedType) {
    typeWidget.__lkModelType = currentType;
    refillChoices(node, currentType);
  }
}

let syncTimer = null;
/** Coalesce a burst of events into one pass on the next tick. */
function scheduleSync() {
  if (syncTimer != null) return;
  syncTimer = setTimeout(() => {
    syncTimer = null;
    (app.graph?._nodes ?? []).forEach((node) => {
      if (node.type === NODE_NAME) sync(node);
    });
  }, 0);
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.A1111ModelInfo",
  afterConfigureGraph() {
    // After a workflow load, re-sync all A1111ModelInfo nodes.
    (app.graph?._nodes ?? []).forEach((node) => {
      if (node.type === NODE_NAME) scheduleSync();
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      // Wrap the model_type widget callback so changes trigger a refill.
      const typeWidget = findWidget(this, MODEL_TYPE_WIDGET);
      if (typeWidget && !typeWidget.__lkA1111ModelInfoHooked) {
        typeWidget.__lkA1111ModelInfoHooked = true;
        const callback = typeWidget.callback;
        typeWidget.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          sync(this);
          return cbResult;
        };
      }

      // Store initial marker without refilling (backend ships Checkpoint list).
      if (typeWidget && typeWidget.__lkModelType === undefined) {
        typeWidget.__lkModelType = typeWidget.value;
      }

      return result;
    };

    // onConfigure fires after widgets_values are restored from a workflow.
    // If the stored type differs from what was used to fill choices, re-fetch.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = onConfigure?.apply(this, args);
      scheduleSync();
      return result;
    };
  },

  setup() {
    scheduleSync();
    // Newer frontends can change combo values without invoking widget.callback;
    // the cheap periodic pass keeps model_name in sync.
    setInterval(scheduleSync, 1000);
  },
});
