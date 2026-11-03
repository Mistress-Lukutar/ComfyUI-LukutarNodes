/**
 * File:   paste_keep_mask.js
 * Brief:  "Paste (Clipspace, Keep Mask)" and "Paste Image (Keep Mask)"
 *         context-menu items for image upload nodes (Load Image &
 *         friends): paste a new image into the widget but keep the mask
 *         already stored in the current file's alpha channel.
 * Author: Mistress-Lukutar
 * Date:   2026-11-03
 * Version: v0.13.0
 *
 * ComfyUI has no separate mask storage: the MaskEditor bakes the
 * painted mask into the ALPHA CHANNEL of the file the image widget
 * points at (alpha = 255 - painted), and LoadImage reads its MASK
 * output back from that alpha. The stock pastes merely replace the
 * file — clipspace paste points the widget at the clipspace file, and
 * the Nodes-2.0 "Paste Image" uploads the clipboard image — so the old
 * mask dies with the old file. These items download the current file,
 * lift its alpha, re-bake it on top of the new image (stretched if the
 * sizes differ) and upload the result as a new input file — the same
 * flow the MaskEditor itself uses on save.
 *
 * Two items:
 * - "Paste (Clipspace, Keep Mask)" pastes the clipspace's SELECTED
 *   image, ignoring img_paste_mode — the keep-mask sibling of the
 *   stock "Paste (Clipspace)".
 * - "Paste Image (Keep Mask)" pastes the first image on the system
 *   clipboard — the keep-mask sibling of the Nodes-2.0 "Paste Image"
 *   (which is what the new frontend's paste flow is built on). Works
 *   on the legacy frontend too.
 *
 * If the current file carries no mask at all (no alpha, or alpha
 * fully opaque) the item degrades to an ordinary paste: the clipspace
 * item just points the widget at the clipspace reference and nothing
 * is uploaded; the clipboard item uploads the pasted image as-is.
 *
 * The combined PNG is encoded from the raw RGBA buffer instead of
 * canvas.toBlob because canvases store pixels premultiplied: toBlob
 * would zero the RGB under every masked (transparent) pixel. Mirrors
 * the frontend's own encodeRgbaAsPng (utils/pngEncodeUtil.ts).
 *
 * Menu placement: both items are added via getExtraMenuOptions, which
 * BOTH frontends honour — the legacy one renders them directly in the
 * node menu, the Vue-based Nodes-2.0 menu converts them into its
 * "Extensions" section (like other packs' items). The media section
 * with icons ("Paste Image" & co.) is not extensible — those items
 * are built Vue-side and tagged through an internal WeakMap no
 * extension can reach — hence the Extensions placement. Items appear
 * whenever the clip source can be resolved: the clipspace item needs
 * a clipspace image (clipspace.images, or the snapshotted image
 * widget value — recent frontends only copy node OUTPUTS into
 * clipspace.images, which input nodes often don't have), the
 * clipboard item is always offered on image-upload nodes. Pure
 * convenience — with the extension disabled everything keeps working
 * and the stock paste behaviour is unchanged.
 *
 * Served at /extensions/<pack>/js/paste_keep_mask.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app, ComfyApp } from "../../../scripts/app.js";

console.info("[LukutarNodes] paste-keep-mask extension loaded");

/** Context-menu labels, in the stock paste items' naming style. */
const CLIPSPACE_LABEL = "Paste (Clipspace, Keep Mask)";
const CLIPBOARD_LABEL = "Paste Image (Keep Mask)";
/** Prefix for the uploaded combined file (MaskEditor uses clipspace-*). */
const FILE_PREFIX = "clipspace-paste-keep-mask-";
/** One shared encoder for PNG chunk type bytes. */
const TEXT_ENCODER = new TextEncoder();

/* ------------------------------------------------------------------ */
/* Straight-alpha PNG encoder (mirror of the frontend's pngEncodeUtil) */
/* ------------------------------------------------------------------ */

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  out.set(TEXT_ENCODER.encode(type), 4);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** zlib-format deflate (what a PNG IDAT needs) via CompressionStream. */
async function zlibDeflate(bytes) {
  const stream = new Blob([bytes])
    .stream()
    .pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Encode raw RGBA pixels as a PNG without any canvas round-trip.
 * @param {{width: number, height: number, data: Uint8ClampedArray}} image
 * @returns {Promise<Blob>}
 */
async function encodeRgbaAsPng(image) {
  const { width, height, data } = image;
  const bytesPerRow = width * 4;

  const raw = new Uint8Array(height * (bytesPerRow + 1));
  for (let y = 0; y < height; y++) {
    const rowStart = y * (bytesPerRow + 1);
    raw[rowStart] = 0; // scanline filter: None
    raw.set(data.subarray(y * bytesPerRow, (y + 1) * bytesPerRow), rowStart + 1);
  }

  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, width);
  ihdrView.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type: RGBA

  const idat = await zlibDeflate(raw);

  return new Blob(
    [
      PNG_SIGNATURE,
      pngChunk("IHDR", ihdr),
      pngChunk("IDAT", idat),
      pngChunk("IEND", new Uint8Array(0)),
    ],
    { type: "image/png" }
  );
}

/* ------------------------------------------------------------------ */
/* File-reference helpers                                              */
/* ------------------------------------------------------------------ */

/**
 * Turn {filename, subfolder, type} into the annotated widget string
 * ("subfolder/name [type]"), the format the stock paste produces.
 */
function annotatedPath(ref) {
  return (
    (ref.subfolder ? ref.subfolder + "/" : "") +
    ref.filename +
    (ref.type ? " [" + ref.type + "]" : "")
  );
}

/**
 * Parse a widget value back into a file reference. Accepts both the
 * annotated string form and the {filename, subfolder, type} object
 * form; returns null when the value points at no file.
 */
function parseWidgetFileRef(value) {
  if (value && typeof value === "object") {
    if (typeof value.filename === "string") {
      return {
        filename: value.filename,
        subfolder: value.subfolder || "",
        type: value.type || "input",
      };
    }
    return null;
  }
  if (typeof value !== "string" || !value) {
    return null;
  }
  // Split at the LAST " [" so file names with spaces survive.
  let filename = value;
  let type = "input";
  const bracket = value.lastIndexOf(" [");
  if (bracket !== -1 && value.endsWith("]")) {
    filename = value.slice(0, bracket);
    type = value.slice(bracket + 2, -1) || "input";
  }
  const slash = filename.lastIndexOf("/");
  const subfolder = slash === -1 ? "" : filename.slice(0, slash);
  if (slash !== -1) {
    filename = filename.slice(slash + 1);
  }
  return { filename, subfolder, type };
}

/** /api/view URL for a file reference (same origin, no CORS trouble). */
function viewURL(ref) {
  const query = new URLSearchParams({
    filename: ref.filename,
    subfolder: ref.subfolder,
    type: ref.type,
  });
  return app.api.apiURL("/view") + "?" + query.toString();
}

async function fetchAsBitmap(ref) {
  const response = await app.api.fetchApi(viewURL(ref));
  if (!response.ok) {
    throw new Error(
      "failed to fetch " + annotatedPath(ref) + " (" + response.status + ")"
    );
  }
  return createImageBitmap(await response.blob());
}

/* ------------------------------------------------------------------ */
/* The paste-keep-mask operation                                       */
/* ------------------------------------------------------------------ */

/**
 * Draw a bitmap (optionally rescaled) and return its pixels. Reading
 * pixels through a canvas only ever loses RGB under transparency —
 * the ALPHA channel survives exactly, and that is all we take from
 * the old image. For the new image the source is opaque, so its RGB
 * is exact too.
 */
function drawToImageData(bitmap, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

/** True when at least one pixel is not fully opaque, i.e. a mask exists. */
function hasMask(imageData) {
  const d = imageData.data;
  for (let i = 3; i < d.length; i += 4) {
    if (d[i] !== 255) {
      return true;
    }
  }
  return false;
}

/** Upload a PNG blob as a new input file; returns its server reference. */
async function uploadImage(blob) {
  const formData = new FormData();
  formData.append("image", blob, FILE_PREFIX + Date.now() + ".png");
  formData.append("type", "input");
  const response = await app.api.fetchApi("/upload/image", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error("upload failed (" + response.status + ")");
  }
  const data = await response.json();
  if (!data || typeof data.name !== "string") {
    throw new Error("upload response missing 'name'");
  }
  return {
    filename: data.name,
    subfolder: data.subfolder || "",
    type: data.type || "input",
  };
}

/** Point the widget at a file, refreshing preview and serialization. */
function applyImageToWidget(node, widget, ref) {
  const value = annotatedPath(ref);
  widget.value = value;
  if (node.properties) {
    node.properties[widget.name] = value;
  }
  if (node.widgets_values && node.widgets) {
    const index = node.widgets.indexOf(widget);
    if (index >= 0) {
      node.widgets_values[index] = value;
    }
  }
  node.images = [ref];

  // Preview refresh, exactly like the stock paste / old showImage.
  const img = new Image();
  img.src = viewURL(ref);
  node.imgs = [img];
  node.imageIndex = 0;
  const outputs = app.nodeOutputs?.[String(node.id)];
  if (outputs) {
    outputs.images = node.images;
  }
  app.canvas.setDirty(true);
}

/** Surface a failure as a toast (Nodes-2.0) plus the console. */
function reportError(message, error) {
  console.error("[LukutarNodes] " + message, error);
  app.extensionManager?.toast?.addAlert?.(message + ": " + error);
}

/**
 * Download the widget's current file, lift its alpha (the painted
 * mask), re-bake it over `newBitmap`'s pixels and upload the combined
 * PNG. Returns the uploaded file's reference, or null when the
 * current file carries no mask (the caller then pastes plainly).
 */
async function bakeMaskAndUpload(node, widget, newBitmap) {
  const width = newBitmap.width;
  const height = newBitmap.height;

  // Old alpha (the mask), stretched to the new image's dimensions.
  let oldAlpha = null;
  const currentRef = parseWidgetFileRef(widget.value);
  if (currentRef) {
    try {
      const oldBitmap = await fetchAsBitmap(currentRef);
      const oldPixels = drawToImageData(oldBitmap, width, height);
      if (hasMask(oldPixels)) {
        oldAlpha = oldPixels.data;
      }
    } catch (error) {
      console.warn(
        "[LukutarNodes] paste-keep-mask: current image unavailable, " +
          "pasting without mask:",
        error
      );
    }
  }

  if (!oldAlpha) {
    return null;
  }

  // New RGB + old alpha, encoded straight (no premultiplication).
  const newPixels = drawToImageData(newBitmap, width, height);
  const merged = new Uint8ClampedArray(newPixels.data);
  for (let i = 3; i < merged.length; i += 4) {
    merged[i] = oldAlpha[i];
  }

  const blob = await encodeRgbaAsPng({ width, height, data: merged });
  const ref = await uploadImage(blob);
  applyImageToWidget(node, widget, ref);
  return ref;
}

/**
 * Paste the clipspace image into the widget, keeping the current
 * file's alpha (the painted mask). Falls back to a plain paste when
 * there is no mask to keep.
 */
async function pasteKeepMask(node, widget, clipRef) {
  try {
    const newBitmap = await fetchAsBitmap(clipRef);
    if (!(await bakeMaskAndUpload(node, widget, newBitmap))) {
      // Nothing to keep — behave like the stock paste.
      applyImageToWidget(node, widget, clipRef);
    }
  } catch (error) {
    reportError(CLIPSPACE_LABEL + " failed", error);
  }
}

/**
 * First image bitmap on the system clipboard, or null when the
 * clipboard holds no image (Permissions/Focus errors propagate).
 */
async function readClipboardBitmap() {
  const items = await navigator.clipboard.read();
  for (const item of items) {
    const imageType = item.types.find((type) => type.startsWith("image/"));
    if (!imageType) {
      continue;
    }
    const blob = await item.getType(imageType);
    return createImageBitmap(blob);
  }
  return null;
}

/**
 * Paste the clipboard image into the widget, keeping the current
 * file's alpha (the painted mask). The Nodes-2.0 menu's own
 * "Paste Image" is the no-mask sibling of this item. Falls back to
 * uploading the pasted image as-is when there is no mask to keep.
 */
async function pasteClipboardKeepMask(node, widget) {
  try {
    const newBitmap = await readClipboardBitmap();
    if (!newBitmap) {
      console.info("[LukutarNodes] paste-keep-mask: no image on clipboard");
      return;
    }
    if (!(await bakeMaskAndUpload(node, widget, newBitmap))) {
      // Nothing to keep — behave like the stock clipboard paste:
      // upload the image itself and point the widget at it.
      const pixels = drawToImageData(
        newBitmap,
        newBitmap.width,
        newBitmap.height
      );
      const blob = await encodeRgbaAsPng({
        width: newBitmap.width,
        height: newBitmap.height,
        data: pixels.data,
      });
      const ref = await uploadImage(blob);
      applyImageToWidget(node, widget, ref);
    }
  } catch (error) {
    reportError(CLIPBOARD_LABEL + " failed", error);
  }
}

/* ------------------------------------------------------------------ */
/* Clip-source resolution                                              */
/* ------------------------------------------------------------------ */

/**
 * The frontend's clipspace. Always reached through ComfyApp: the
 * legacy frontends keep it as a plain property, the Vue-based ones
 * as a static getter onto the new clipspace module — both are read
 * the same way from an extension.
 */
function resolveClipspace() {
  return ComfyApp.clipspace ?? null;
}

/**
 * The file reference a clipspace paste would apply to this node's
 * image widget: the clipspace's selected image, or — when the
 * clipspace holds no images (recent frontends copy node OUTPUTS
 * there, which input nodes without a run don't have) — the
 * snapshotted image widget value, exactly what the stock
 * "Paste (Clipspace)" restores. Null when nothing is pasteable.
 */
function resolveClipRef(clipspace, inputNames) {
  if (!clipspace) {
    return null;
  }
  const images = clipspace.images;
  if (images?.length) {
    const index = Math.min(
      Math.max(clipspace.selectedIndex ?? 0, 0),
      images.length - 1
    );
    return images[index];
  }
  const widgetSnapshot = clipspace.widgets?.find((w) =>
    inputNames.includes(w.name)
  );
  return widgetSnapshot ? parseWidgetFileRef(widgetSnapshot.value) : null;
}

/* ------------------------------------------------------------------ */
/* Context-menu registration                                           */
/* ------------------------------------------------------------------ */

/** Input names of this node def that carry image_upload: true. */
function imageUploadInputNames(nodeData) {
  const names = [];
  for (const section of [nodeData?.input?.required, nodeData?.input?.optional]) {
    if (!section) {
      continue;
    }
    for (const [name, spec] of Object.entries(section)) {
      if (Array.isArray(spec) && spec[1]?.image_upload === true) {
        names.push(name);
      }
    }
  }
  return names;
}

function findImageWidget(node, inputNames) {
  // "imageUpload" is the legacy frontend's widget type; the current
  // frontend keeps the plain combo, so fall back to the input name.
  return node.widgets?.find(
    (w) => w.type === "imageUpload" || inputNames.includes(w.name)
  );
}

app.registerExtension({
  name: "Lukutar.PasteKeepMask",

  beforeRegisterNodeDef(nodeType, nodeData) {
    const inputNames = imageUploadInputNames(nodeData);
    if (!inputNames.length) {
      return;
    }

    const original = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      const result = original?.apply(this, arguments);
      const widget = findImageWidget(this, inputNames);
      if (!widget) {
        return result;
      }
      const node = this;

      // The keep-mask sibling of the stock "Paste (Clipspace)": shown
      // whenever the clipspace holds anything pasteable (the same
      // condition the stock item uses, relaxed to widget values).
      const clipRef = resolveClipRef(resolveClipspace(), inputNames);
      if (clipRef) {
        options.push({
          content: CLIPSPACE_LABEL,
          callback: () => {
            void pasteKeepMask(node, widget, clipRef);
          },
        });
      }

      // The keep-mask sibling of the Nodes-2.0 "Paste Image" (the new
      // frontend's paste flow): offered on every image-upload node,
      // in both menus, exactly like the stock item.
      options.push({
        content: CLIPBOARD_LABEL,
        callback: () => {
          void pasteClipboardKeepMask(node, widget);
        },
      });

      return result;
    };
  },
});
