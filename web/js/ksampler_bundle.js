/**
 * File:   ksampler_bundle.js
 * Brief:  KSampler (Bundle) — make the decoded result copyable without
 *         duplicating the preview picture.
 * Author: Mistress-Lukutar
 * Date:   2026-11-04
 * Version: v0.1.0
 *
 * The node displays its result through the binary preview channel:
 * the final decoded frame replaces the last latent noise frame on the
 * node.  The backend additionally saves every decoded frame as a temp
 * PNG and emits it as the "executed" output ui.  This extension makes
 * that output useful without showing it twice:
 *
 * - `node.hideOutputImages = true` keeps the v2 output section hidden.
 *   The v2 output display prefers live preview frames over executed
 *   outputs, so rendering the section would show the same picture a
 *   second time below the preview (the v1 canvas renderer draws
 *   node.imgs once and is unaffected by the flag).
 * - `node.imgs` is fed with /view URLs of the temp files on
 *   `onExecuted`, which is what the right-click menu (Open Image /
 *   Copy Image / Save Image) is built from.  Open and Save then work
 *   against the real full-resolution temp file instead of the JPEG
 *   preview blob.
 *
 * Served at /extensions/<pack>/js/ksampler_bundle.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.info("[LukutarNodes.KSamplerBundle] extension loaded");

const NODE_NAME = "BundleKSampler";

/** Build the /view URL for an executed image reference. */
function viewUrl(image) {
  const params = new URLSearchParams();
  params.set("filename", image.filename ?? "");
  params.set("subfolder", image.subfolder ?? "");
  params.set("type", image.type ?? "temp");
  return api.apiURL(`/view?${params.toString()}`);
}

app.registerExtension({
  name: "Lukutar.BundleKSampler",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalNodeCreated?.apply(this, arguments);
      this.hideOutputImages = true;
      return result;
    };

    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      originalOnExecuted?.apply(this, arguments);
      const images = output?.images;
      if (!Array.isArray(images) || images.length === 0) return;
      this.previewMediaType = "image";
      this.imgs = images
        .filter((image) => image && image.filename)
        .map((image) => {
          const img = new Image();
          img.src = viewUrl(image);
          return img;
        });
    };
  },
});
