/**
 * File:   save_image.js
 * Brief:  LukutarSaveImage node — conditionally show/hide the metadata
 *         input socket based on whether the selected format supports it.
 * Author: Mistress-Lukutar
 * Date:   2026-11-02
 * Version: v0.1.0
 *
 * The backend always declares an optional `metadata` input of type
 * LUKUTAR_METADATA. This extension adds/removes that socket depending
 * on the format widget's value: formats that embed metadata (PNG, JPG,
 * JXL, WEBP, TIFF) get the socket; BMP (which lacks metadata support)
 * hides it by removing the unconnected socket. The backend re-declares
 * the socket on every workflow configure, so apply() is idempotent.
 *
 * Supported formats are read from formatWidget.options.metadata_formats
 * when present; otherwise only "BMP" is considered metadataless.
 *
 * Served at /extensions/<pack>/js/save_image.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";

console.info("[LukutarNodes.SaveImage] extension loaded");

const NODE_NAME = "LukutarSaveImage";
const FORMAT_WIDGET = "format";
const METADATA_INPUT = "metadata";
const METADATA_TYPE = "LUKUTAR_METADATA";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * Whether the given format string supports metadata embedding.
 * Uses formatWidget.options.metadata_formats when present; otherwise
 * falls back to assuming only "BMP" lacks support.
 */
function supportsMetadata(format, metadataFormats) {
  if (Array.isArray(metadataFormats)) {
    return metadataFormats.includes(format);
  }
  // Fallback: only BMP lacks metadata support.
  return format !== "BMP";
}

/** The optional metadata input socket (real socket, not a widget entry). */
function metadataInput(node) {
  return (node.inputs ?? []).find(
    (input) => input.name === METADATA_INPUT && input.widget == null
  );
}

/* ------------------------------------------------------------------ */
/* Apply logic                                                         */
/* ------------------------------------------------------------------ */

/**
 * Add or remove the metadata input socket based on format capability.
 * Idempotent: calling twice has no additional effect.
 *   - Format supports metadata + no socket → add it.
 *   - Format lacks metadata + unconnected socket → remove it.
 *   - Format lacks metadata + connected socket → keep it (backend warns).
 */
function apply(node) {
  const formatWidget = findWidget(node, FORMAT_WIDGET);
  const formatValue = formatWidget?.value ?? "PNG";
  const supportedFormats = formatWidget?.options?.metadata_formats;
  const ok = supportsMetadata(formatValue, supportedFormats);

  const input = metadataInput(node);

  if (ok) {
    // Format supports metadata — ensure the socket exists.
    if (!input) {
      node.addInput(METADATA_INPUT, METADATA_TYPE);
    }
  } else {
    // Format lacks metadata — hide unconnected sockets only.
    if (input && input.link == null) {
      const idx = (node.inputs ?? []).indexOf(input);
      if (idx >= 0) node.removeInput(idx);
    }
    // If connected, keep the link; the backend warns at save time.
  }
}

/* ------------------------------------------------------------------ */
/* Registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "LukutarNodes.SaveImage",
  afterConfigureGraph() {
    // Re-apply socket visibility after a workflow load (format may have
    // changed between saves).
    (app.graph?._nodes ?? []).forEach((node) => {
      if (node.type === NODE_NAME) apply(node);
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);

      // Hook format widget callback so apply() runs on every format change.
      const formatWidget = findWidget(this, FORMAT_WIDGET);
      if (formatWidget && !formatWidget.__lkSaveImageHooked) {
        formatWidget.__lkSaveImageHooked = true;
        const callback = formatWidget.callback;
        formatWidget.callback = (...args) => {
          const cbResult = callback?.apply(this, args);
          apply(this);
          return cbResult;
        };
      }

      // Initial pass — show/hide socket for the default format.
      apply(this);
      return result;
    };
  },
});
