# ComfyUI-LukutarNodes

A collection of custom nodes for
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) by Mistress-Lukutar.

The pack is organized as a home for standalone image-processing nodes:
each node lives in its own module under `nodes/`, backed by a pure
numpy/OpenCV engine under `core/` (no ComfyUI imports, unit-testable
anywhere) and thin tensor-conversion helpers under `utils/`.

## Nodes

- **Color Match (Frequency Split)** — recolor a processed image from a
  reference (`Lukutar/Image`).
- **SEGS BBox Overlay** — draw Impact Pack SEGS detections on an image
  (`Lukutar/Image`).
- **SEGS Set Crop Size** — refit Impact Pack SEGS crop regions to an
  absolute target size, e.g. 512×512 (`Lukutar/Image`).
- **Prompt Annotate** — inline `|label: text|` region markup in one
  prompt (`Lukutar/Prompt`).
- **Annotations to Wildcard (LAB)** — annotations → Impact Pack `[LAB]`
  wildcard text (`Lukutar/Prompt`).
- **Annotation Segment** — extract one label's prompt text
  (`Lukutar/Prompt`).
- **Annotation Labels** — the label set as one comma-separated string
  (`Lukutar/Prompt`).
- **Annotations to String** — annotations → plain prompt string, tag-free or raw markup (Lukutar/Prompt).
- **Annotation Segment Edit** — pass-through per-label edits on
  ANNOTATIONS: prepend/append/remove text, new/delete segments
  (`Lukutar/Prompt`).
- **Set Variable / Get Variable** — named workflow variables: publish a
  value once, read it anywhere without dragging wires
  (`Lukutar/Variables`).
- **Set Bundle / Get Bundle** — a named bundle of several typed values
  (model + vae + clip + …): grow slots by wiring, read the whole bundle
  anywhere (`Lukutar/Variables`).
- **Group Switch** — one boolean row per canvas group: switch whole
  groups on/off (bypass or mute) from a single panel node
  (`Lukutar/Groups`).
- **Seed History** — a seed source with the built-in
  `seed`/`control_after_generate` widgets plus a rolling log of used
  seeds rendered as copyable text (`Lukutar/Seeds`).
- **Save Image (Metadata)** — save IMAGE batches to disk as PNG, JPG,
  WEBP, TIFF or BMP with optional LUKUTAR_METADATA embedding; passes the
  input batch and metadata through as `images` + `metadata` outputs for
  mid-graph chaining (`Lukutar/Image`).
- **Save Image JXL** — dedicated JPEG-XL saver with lossy and lossless
  modes and full encoder control (loss, effort, decode speed, threads,
  metadata via EXIF box); passes the input batch and metadata through as
  `images` + `metadata` outputs for mid-graph chaining (`Lukutar/Image`).
- **Metadata Blocks** — build a LUKUTAR_METADATA dict from up to 16
  editable name/value pairs (`Lukutar/Metadata`).
- **Standard Metadata** — emit workflow / prompt / datetime entries as
  a LUKUTAR_METADATA dict (`Lukutar/Metadata`).
- **Merge Metadata** — merge several LUKUTAR_METADATA blocks into one
  without duplicates (same-name blocks join; names matching an A1111
  settings entry fold into that block's text) via dynamic
  SEGS-Concat-style inputs
  (`Lukutar/Metadata`).
- **A1111 Parameters** — assemble an A1111-style `parameters` infotext
  from positive/negative prompts, version, and metadata-block settings
  (`Lukutar/Metadata`).
- **A1111 Sampling** — emit sampling parameters (steps, sampler, CFG,
  seed, size, clip skip) as a metadata block
  (`Lukutar/Metadata`).
- **A1111 Hires Fix** — emit high-resolution-fix parameters (upscaler,
  upscale factor, hires steps, denoising) as a metadata block
  (`Lukutar/Metadata`).
- **A1111 Model Info** — look up a model file's SHA-256 prefix and emit
  type-specific name/hash entries with on-disk caching
  (`Lukutar/Metadata`).
- **A1111 Lora Hashes** — hash up to eight LoRA files and emit a single
  `"Lora hashes"` entry in A1111 format
  (`Lukutar/Metadata`).
- **Load Checkpoint (Bundle)** — load a checkpoint with optional LoRAs,
  encode both prompts with selectable weight semantics and emit the
  bundle plus its portable info entity (`Lukutar/Bundle`).
- **LoRA List** — collect several LoRAs with weights into one list
  socket for Load Checkpoint (Bundle) (`Lukutar/Bundle`).
- **KSampler (Bundle)** — sample the bundle like the core KSampler, with
  a live latent preview and an optional VAE decode whose image replaces
  the noise preview; the result is saved to temp so it can be
  opened/copied/saved from the node's right-click menu
  (`Lukutar/Bundle`).
- **Info to A1111 Parameters** — convert the info entity into the
  A1111 `parameters` metadata block, filtering out pack-only fields and
  normalising values to the official A1111 infotext format
  (`Lukutar/Info`).
- **Info Unpack** — read the standard info entries as separate
  string outputs (`Lukutar/Info`).
- **Info Pack** — assemble an info entity from an optional base
  plus key/value pairs (`Lukutar/Info`).
- **Checkpoint Bundle Unpack** / **Checkpoint Bundle Pack** — split a
  LUKUTAR_BUNDLE into standard sockets and reassemble it
  (`Lukutar/Bundle`).

> **Why KSampler (Bundle)?** The core KSampler shows you noise: in the
> legacy (v1) frontend the progress preview is a *latent* frame — a grey,
> noisy picture that never becomes the clean result — and in the v2
> frontend the finished image is rendered *a second time* below the
> preview, so the node ends up showing the same picture twice. **KSampler
> (Bundle)** fixes both: the decoded image is pushed through the same
> preview channel the latent frames use, so it *replaces* the noise
> frame in place — one clean picture, no duplicates, in both frontends.
> The frames are additionally saved to temp and wired into
> `node.imgs` by the web extension, so the result is copyable via the
> right-click menu like any PreviewImage output.
- **Paste (Clipspace, Keep Mask)** / **Paste Image (Keep Mask)** —
  context-menu items for image upload nodes: paste a clipspace image
  or a clipboard image while keeping the node's painted mask
  (frontend extension, no node).

### Color Match (Frequency Split)

Restores the color distribution of a **reference** image onto a
**processed** image (for example an SD upscale) while keeping the
processed image's detail.

The node splits the input into frequency layers with a Gaussian blur
(`sigma` = cutoff frequency), transfers the color statistics of the
reference onto the low-frequency layer, and recombines it with the
untouched high-frequency detail of the input.

**Typical use case.** SD iterative upscaling drifts colors through
repeated VAE encode/decode cycles. Feed the pre-drift image as
`reference` and the upscaled result as `image` to bring the colors back
without losing the upscaled detail.

**Color transfer methods:**

- `reinhard` — transfers per-channel mean/std in LAB space from the
  reference low layer onto the input low layer. The default; preserves
  the input's tonal structure while adopting the reference's color
  envelope.
- `replace` — swaps the input's low-frequency layer for the reference's
  verbatim. Stronger effect; useful when the drift is severe.

**Auto-tune.** With `auto_tune` enabled the node grid-searches sigma
from `sigma_min` to `sigma_max` in `sigma_step` increments, scores each
candidate with a fixed evaluation blur (`eval_sigma`) and reuses the
winning sigma for the whole batch. The `envelope` metric compares
blurred layers only (color/tonal accuracy, detail ignored); `full`
compares raw images (color + structure). The winning sigma is returned
as the `sigma` output so it can be wired into other nodes.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Processed image to recolor (e.g. the upscale). |
| reference | IMAGE | — | Image with the target color distribution (the original). |
| auto_tune | BOOLEAN | manual | Grid-search sigma on the first frame and reuse the best value for the whole batch. |
| sigma | FLOAT | 30.0 | Gaussian blur radius (frequency cutoff). Ignored when auto_tune is on. |
| method | COMBO | reinhard | `reinhard`: transfer LAB mean/std. `replace`: use the reference low layer verbatim. |
| sigma_min | FLOAT | 5.0 | Auto-tune grid start. |
| sigma_max | FLOAT | 60.0 | Auto-tune grid end. |
| sigma_step | FLOAT | 5.0 | Auto-tune grid step. |
| eval_sigma | FLOAT | 20.0 | Fixed blur radius for auto-tune scoring; prevents degenerate minima at high sigma. |
| metric | COMBO | envelope | `envelope`: color only. `full`: color + structure. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Recolored batch, same resolution as `image`. |
| sigma | FLOAT | Sigma actually used (best grid hit when auto-tuning). |

#### Batch behaviour

- `reference` with a single frame is broadcast to every frame of `image`.
- Matching batch sizes are processed pairwise.
- Mismatched larger batches raise an error on the node.
- Auto-tune searches on the first frame pair only, then applies the
  winning sigma to the whole batch (consistent processing for video).

#### Progress reporting

The node drives the ComfyUI progress bar: per frame in manual mode, per
sigma candidate plus per frame while auto-tuning. Every auto-tune
candidate and its score are also echoed to the ComfyUI console log.

### SEGS BBox Overlay

Draws [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
SEGS detections on an image the way YOLO demos do: an outlined bounding
box per segment with a filled caption plate (class name, optionally
with the confidence percentage) and an optional semi-transparent tint
of the segment masks.

Feed it the output of `SEGM Detector (SEGS)` (e.g. the bbox detector
used by ADetailer) plus the image the detector ran on. The `segs` input
is passed through unchanged, so the node can be inserted between the
detector and a Detailer pipeline to preview exactly what will be
repainted.

**Colors.** In `auto` mode each class gets a stable color from a
built-in vivid palette (same class = same color, regardless of
detection order). In `single` mode every detection is drawn with one
user color (`color_r`/`color_g`/`color_b`, only used by that mode).

**Size mismatch.** When the image resolution differs from the one
recorded in SEGS (e.g. the detection ran on an upscaled copy but you
preview the original), all coordinates and masks are rescaled
proportionally — automatically.

Captions use OpenCV's built-in Hershey font, so labels are limited to
ASCII characters (detector class names like `face` or `hand` are fine).

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to draw on, e.g. the one the detector ran on. |
| segs | SEGS | — | Segments from an Impact Pack detector (SEGM Detector (SEGS)). |
| label_format | COMBO | label+confidence | `label`: class name only. `label+confidence`: append the score, e.g. `face 91%`. |
| draw_masks | BOOLEAN | masks | Tint the segment masks (`boxes only` to disable). |
| mask_alpha | FLOAT | 0.45 | Mask tint strength, 0–1. |
| thickness | INT | 0 | Box border width in pixels; 0 = auto from the image height. |
| font_scale | FLOAT | 1.0 | Multiplier on the auto caption text size. |
| color_mode | COMBO | auto | `auto`: stable color per class from the palette. `single`: one user color. |
| color_r / color_g / color_b | INT | 0 / 255 / 0 | Single color mode's RGB channels. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Annotated batch, same resolution as `image`. |
| segs | SEGS | The input SEGS passed through unchanged. |

#### Batch behaviour

- SEGS describe a single image, so the same detections are drawn on
  every frame of the batch (useful for previewing a video pass).

### SEGS Set Crop Size

Refits every [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
SEGS segment's crop region to an **absolute** target size, e.g.
512×512, instead of the detector's relative `crop_factor`.

Why: `crop_factor` scales with the detection, so the same factor yields
128×128 crops for small segments and 1024×1024 for large ones, and the
actual crop sizes are often sampler-unfriendly (552×239). This node
pins the crop size so every segment is processed at a resolution you
choose.

The bbox, label, confidence and the mask content are untouched — only
the crop rectangle (and the mask's alignment to it) changes, so the
node can sit between a detector (e.g. SEGM Detector (SEGS)) and a
Detailer pipeline. The added context area around the mask is
zero-filled in the re-cut mask, exactly like Impact's own crop padding.

**Modes.**

- `exact` — the crop region is exactly `width`×`height`, centered on
  the bbox. If the bbox itself is larger than the target, the region
  grows just enough to contain it (rounded up to a multiple of
  `round_to`) so a detection is never cut off; the size is then larger
  than requested and a warning is logged.
- `aspect` — the bbox is scaled uniformly around its center so its
  longer side equals `max(width, height)`, proportions kept and sizes
  rounded up to multiples of `round_to` (400×173 @ 512 → 512×224).
  Never scales below 1×, so the bbox always fits.

**Clamping.** Regions are centered on the bbox center and shifted to
stay inside the image; a target larger than the image is clamped to
the image size (with a warning).

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| segs | SEGS | — | Segments from an Impact Pack detector (SEGM Detector (SEGS)). |
| width | INT | 512 | Target crop width in pixels. |
| height | INT | 512 | Target crop height in pixels. |
| mode | COMBO | exact | `exact`: exactly width×height (grown only if the bbox is larger). `aspect`: uniform scale so the bbox's longer side equals max(width, height). |
| round_to | INT | 8 | Round fitted sizes up to multiples of this (SD-friendly); applies to aspect-mode sizes and bbox-forced growth, the exact target is verbatim. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| segs | SEGS | SEGS with refitted crop regions; bboxes and mask content unchanged. |

### Prompt Annotate

Annotates **one prompt** with region labels for label-driven
per-region workflows (e.g. classifier segments → Detailer inpainting).
The prompt stays a single text that can still drive the base
generation; only the added markup selects which parts belong to which
region.

**Markup.** ``|label1,label2: text|`` tags mark `text` as belonging to
`label1` and `label2`:

```
masterpiece, |body:1girl, thin|, |face:blue eyes, smirk|, |body,hair:red hair|, |body:stands|, |background:outdoors, park|
```

- Tags are **flat, never nested** — interleaved regions use repeated
  labels (`body` appears twice above); one tag may carry several labels.
- Labels are free-form (`letters`, `digits`, `_`) and should match your
  classifier's label vocabulary; there is no fixed class list.
- Text outside tags is the **common part**, implicitly labelled `all`.
- The `clean_prompt` output strips all markup:
  `masterpiece, 1girl, thin, blue eyes, smirk, red hair, stands,
  outdoors, park` — feed it to the base generation.
- Empty tags (`|face:|`) are dropped with their dangling separators.
- `|` cannot appear in the prompt text itself; malformed tags fail the
  node with the offending position.

**Web editor.** With the web assets loaded (they ship with the pack),
the node's prompt field itself becomes a **rich input**: the markup is
highlighted live while typing — `|label:` parts dimmed, span text in a
stable pastel color per label (dark-theme friendly). The field fills the
node's widget area and follows node resizes; typing past the available
height grows the node, but only up to a 400 px cap — past it the field
stops growing and scrolls instead. The mouse wheel scrolls the field
while it has somewhere to scroll; once the field is at its top/bottom
edge (or has no overflow at all) the wheel zooms the canvas underneath.
The built-in text widget is only
hidden, so the value keeps serializing into the workflow normally.
Standard ComfyUI text shortcuts work in the rich input: Ctrl+Enter queues
the workflow and Ctrl+Up/Down adjusts attention weight around the caret.
The node works without the web assets too — the markup is plain text and
can be typed by hand.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | STRING | — | Prompt with `\|label: text\|` markup (multiline); labels are free-form, typed right into the markup. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| annotations | ANNOTATIONS | Parsed spans; consumed by the nodes below. |
| clean_prompt | STRING | The same prompt with all markup removed. |

### Annotations to Wildcard (LAB)

Converts annotations into the **label-mode wildcard** consumed by
Impact Pack Detailer (SEGS)-style `wildcard` inputs:

```
[LAB]
[ALL] masterpiece,
[body] 1girl, thin, red hair, stands
[face] blue eyes, smirk
[hair] red hair
[background] outdoors, park
```

The common part becomes the `[ALL]` line; multi-label spans are
duplicated into each of their label lines. Impact Pack concatenates
`[ALL]` and the matching label value with no separator, so the `[ALL]`
line ends with a comma whenever label lines follow.

#### Inputs

| Name | Type | Description |
|------|------|-------------|
| annotations | ANNOTATIONS | Annotations from Prompt Annotate. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| wildcard | STRING | `[LAB]`-format wildcard text for Detailer (SEGS). |

### Annotation Segment

Extracts one label's prompt text, e.g. to drive a per-region inpaint
directly. With `include_common` on, the unmarked common part is
prepended (`masterpiece, blue eyes, smirk` for `label=face` above).

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| annotations | ANNOTATIONS | — | Annotations from Prompt Annotate. |
| label | STRING | face | Label to extract; unknown labels fail the node listing the available ones. |
| include_common | BOOLEAN | common + label | Prepend the unmarked common part of the prompt. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| text | STRING | Prompt text for the selected label. |

### Annotation Labels

Renders the annotation's label set as one comma-separated string —
`body, face, hair, background` for the example above — e.g. to feed a
label picker or to log which regions a workflow covers. Labels keep
first-appearance order and are deduplicated. The implicit `all` label
of the unmarked common part is included only with `include_common` on;
with no tagged regions and `include_common` off the output is empty.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| annotations | ANNOTATIONS | — | Annotations from Prompt Annotate. |
| include_common | BOOLEAN | regions only | Include the implicit `all` label of the unmarked common part. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| labels | STRING | Comma-separated label list, e.g. `body, face, hair`. |

### Annotations to String

Converts **ANNOTATIONS** back into a plain prompt string — bridges the
annotations wire into any **STRING** input (e.g. after an Annotation
Segment Edit chain, or when only the annotations socket reaches a
branch). The `mode` combo picks between the tag-free output and the
original markup text:

- `clean` — the same text as Prompt Annotate's `clean_prompt` output
  (what CLIP receives);
- `raw` — the exact original text with the ``|label: text|`` markup
  intact.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| annotations | ANNOTATIONS | — | Annotations from Prompt Annotate. |
| mode | COMBO | clean | `clean`: the tag-free prompt as sent to CLIP; `raw`: the original text with the ``|label: text|`` markup. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| text | STRING | The selected prompt string. |

### Annotation Segment Edit

Pass-through editor for the annotations: takes **ANNOTATIONS in and
emits edited ANNOTATIONS out**, so one shared Prompt Annotate can feed
several workflow branches, each with its own tweaks (edits chain
freely). The `label` field accepts **several labels comma-separated**
(`face, body`) and every mode applies to all of them:

- `prepend` — put the tags before each label's text
  (`face` + `detailed eyes` → `detailed eyes, blue eyes, smirk`);
- `append` — put them after it (`… → blue eyes, smirk, smile`);
- `remove` — delete the listed comma-separated tags from the labels'
  text, matched exactly after trimming (`smirk` → `blue eyes`);
- `new` — append a fresh span `|labels: text|` to the end of the
  prompt with the typed labels and text (e.g. `hands, weapon` +
  `delicate fingers` → a new `|hands,weapon: delicate fingers|` tag);
  every listed label must not exist yet;
- `delete` — remove the labels with their content entirely; the text
  field is unused (the web extension grays it out in this mode).

Notes on the semantics:

- A label spread over several spans is edited at its first span
  (`prepend`) / last span (`append`); `remove` applies to all of them.
- The `all` label edits the unmarked common part **in place** — the
  added text stays unmarked, no `|all:|` tag appears; deleting `all`
  removes the common text, keeping the separators between the
  surviving spans.
- `delete` only strips the label from a multi-label span
  (`|body,hair: red hair|` − `hair` → `|body: red hair|`); a span left
  without any label disappears with its text.
- A span shared by several labels holds one text, so `prepend` /
  `append` / `remove` through any of its labels change that shared
  text for all of them; a span emptied by `remove` disappears from the
  annotation (and its labels from Annotation Labels / the wildcard).
- Removing tags that are not present changes nothing — the annotation
  passes through untouched. The same applies to **blank text** (except
  in `delete`): the edit becomes a no-op.
- Unknown labels, unknown modes and blank label lists fail the node
  (unknown labels list the available ones; `new` fails on labels that
  already exist).

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| annotations | ANNOTATIONS | — | Annotations from Prompt Annotate. |
| label | STRING | face | Label(s) to edit, comma-separated for several; every mode applies to all of them. Unknown labels fail the node listing the available ones. |
| mode | COMBO | prepend | `prepend` / `append` / `remove` / `new` / `delete` — see above. |
| text | STRING | — | Tags to add, the comma-separated tags to remove, or the new span's text; blank text = no edit; unused in `delete` mode. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| annotations | ANNOTATIONS | The edited annotations; same type as the input, feeds the other annotation nodes. |

### Set Variable / Get Variable

Named workflow variables: publish a value under a name once and read it
anywhere in the graph — no wire across groups and collapsed subgraphs
(`Lukutar/Variables`). The type is arbitrary (IMAGE, MODEL,
CONDITIONING, SEGS, strings…): the value is passed through untouched,
so ComfyUI's own execution order and output caching apply, and the node
titles show the detected type (`Set · img_t2i [IMAGE]`).

The web extension connects each Get to its Set with an **invisible real
link** (it serializes into the prompt like a normal wire, it just never
renders on the canvas). Because the connection is real, execution
ordering, caching, exported API workflows and headless runs all work
without any extra machinery. A manual wire into the Get's `value` input
overrides the name; without the web assets the link can simply be wired
by hand.

#### Set Variable — Inputs / Outputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| var_name | STRING | — | Variable name, e.g. `img_t2i`; Get Variable nodes with this name receive the value. |
| value | * | — | The value to publish; any type. |

| Name | Type | Description |
|------|------|-------------|
| value | * | Pass-through of the input; may be wired normally. |

#### Get Variable — Inputs / Outputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| var_name | STRING | — | Variable name to read, e.g. `img_t2i`. |
| value | * | — | Connected automatically (invisible); a manual wire overrides the name. |

| Name | Type | Description |
|------|------|-------------|
| value | * | The variable's value, with its original type. |

#### Rules

- Names are global to the workflow. Several Set nodes **may** share one
  name on alternative branches as long as only one branch is active —
  the "generated or loaded" pattern: muting/bypassing a branch removes
  its nodes from the prompt, so exactly one Set survives. This is the
  key difference from KJNodes' Set/Get, which rejects duplicates on the
  canvas outright.
- Two **simultaneously active** Sets with one name are ambiguous: the
  Get stays unlinked (⚠ on both nodes) and the queue fails with a
  descriptive error. Rename them or mute one branch.
- Branches merged by a switch node should use a single Set **after**
  the switch instead.
- Variables live within one queue run; values are not persisted
  between runs and cannot be "reassigned" sequentially
  (set → set → get reads the last) — that is a duplicate-name conflict.
- Without the web extension, wire the Get's `value` input manually (Set
  output → Get input works fine headless).

### Set Bundle / Get Bundle

A bundle is one named bag holding **several heterogeneously typed
values** — e.g. a model, its VAE and its CLIP under a single name like
`sdxl` — published once and readable anywhere (`Lukutar/Variables`).
The single-value Set/Get Variable pair above stays the right tool for
one value; the bundle pair is for moving a whole "kit" of nodes across
the graph in one go.

Slots grow **SEGS-Concat style** (Impact Pack): a Set Bundle always
shows one free `+ add` input. Dropping a wire onto it fixes that
slot's type to the source (MODEL, VAE, CLIP, IMAGE, anything — any
custom type included) and auto-names it from the type (`model`, `vae`,
`clip`, `model_2`…), then the next free input appears. Disconnecting an
auto-named slot collapses it again. The node menu offers more:

- **Rename slot "…"** — give the slot a meaningful name (e.g.
  `sdxl_model`); renamed slots survive disconnects, so a custom name
  is never lost by accident. Names must be unique, may not contain
  commas/semicolons/colons and may not be `bundle_name`, `slots` or
  `slot_<n>` (reserved).
- **Remove slot "…"** — explicit removal.
- **Add slot ▸** — pre-create an unconnected slot of a chosen type
  (for wiring later; the type list there is fixed, wires accept any
  type).

Every named slot is mirrored to an output of the same name, so the Set
node is a real bus: the wires source → Set → Get → consumer are all
real links, and execution order, caching and API export stay ComfyUI's
own — same principle as the variable pair. A Get Bundle resolves its
`bundle_name` against the active Set Bundles and mirrors the Set's
slots: one named input (the anchor of an invisible real link per slot)
and one typed output per slot, appearing/vanishing as the Set's slots
change. A manual wire into a Get input overrides the name. Titles show
the state: `Set Bundle · sdxl (3)`, `Get Bundle · sdxl (3)`, plus ⚠
warnings (`duplicate`, `no active Set`, `unconnected: vae`).

Both nodes carry a `slots` widget — the spec (`model:MODEL,vae:VAE,…`)
the extension keeps in sync. It is hidden from the UI (internal state;
the sockets themselves are the interface) but serialized into the
workflow and API export; the backends read it for loud per-slot errors
at queue time (a declared but unwired slot fails with
`Set Bundle [sdxl]: slot(s) not connected: vae` instead of a cryptic
None downstream).

#### Set Bundle — Inputs / Outputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| bundle_name | STRING | — | Bundle name, e.g. `sdxl`; Get Bundle nodes with this name receive all slots. |
| slots | STRING | — | Hidden spec, managed automatically; do not edit. |
| *(slots)* | any | — | One input per slot, added by wiring or the node menu. |

| Name | Type | Description |
|------|------|-------------|
| *(slots)* | any | One output per slot, mirroring the inputs by name. |

#### Get Bundle — Inputs / Outputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| bundle_name | STRING | — | Bundle name to read, e.g. `sdxl`. |
| slots | STRING | — | Hidden spec mirror; updated automatically from the Set side. |
| *(slots)* | any | — | One input per slot, connected by invisible links; manual wires win. |

| Name | Type | Description |
|------|------|-------------|
| *(slots)* | any | One output per slot, with the slot's type. |

#### Rules

- The same duplicate-name rules as the variable pair apply: several Set
  Bundles may share one name on alternative branches, but two
  simultaneously active Sets with one name leave Gets unlinked (⚠) and
  fail the queue. A Get follows the spec of the single active Set —
  branches with different slot specs reshape every linked Get on
  switch, dropping its wiring, so keep the specs identical across
  branches.
- Slot values flow in spec order; renaming a slot renames it on every
  linked Get. Removing a slot removes the matching Get sockets; wires
  hanging on removed outputs are dropped (re-plug them).
- A bundle holds at most 16 slots.
- The bundle pair needs the web extension: every socket is grown and
  linked by it, so without the web assets the nodes cannot be wired
  from the UI (use the single-value Set/Get Variable pair instead).
  Hand-written API prompts work regardless — pass the slot links under
  their names and order them via the `slots` spec, or omit the spec
  and name the links `slot_1..slot_16` (values then pass through in
  natural name order).

### Group Switch

A panel node that mirrors the canvas groups: one boolean row per group
(`Lukutar/Groups`). A checked row means the group is active; unchecking
switches the group off with the chosen method. Rows appear, reorder and
vanish automatically as groups are added, renamed, reordered or deleted
on the canvas — the node has no inputs, and its single `states` output
carries the current `{group title: enabled}` dictionary (order follows
the prefix alphabet below).

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| disable_method | bypass / disable | bypass | How an unchecked group is switched off. `bypass` sets the group's nodes to ComfyUI bypass mode (inputs wire through); `disable` sets them to never/mute (nodes leave the prompt). Same mechanism as the native group context menu. |
| change_group_color | BOOLEAN | true | Recolor the frame of a switched-off group: purple for bypass, grey for disable; the original color is restored on re-enable. |
| sort_order | STRING | — | Comma-separated title prefixes for the row order, e.g. `inp,im,det`. A group whose title starts with a token (case-insensitive) takes that token's position — letters or short combinations are enough, no full titles needed; several groups may share a token (they keep their relative order); unlisted groups follow at the end. |
| group_states | STRING | `{}` | Current row states as JSON — managed automatically by the web extension and hidden on the node; edit it only for API workflows without the frontend. |

| Name | Type | Description |
|------|------|-------------|
| states | GROUP_STATES | Ordered `{group title: enabled}` dictionary; wire it anywhere `*` is accepted (e.g. Set Variable) or leave it unconnected. |

#### Behaviour

- Toggling a row applies immediately: node modes are captured first and
  restored on re-enable, so nodes you had muted individually keep that
  state across a panel toggle. Groups switched off through the native
  group menu are shown as-is but are not re-applied by parameter changes.
- The panel state survives reloads. The rows are derived from the live
  group node modes (which ComfyUI saves natively), and the declared
  parameters keep a stable serialization, so `disable_method`,
  `change_group_color`, `sort_order` and the states dictionary come back
  exactly as saved. The original frame color of a recolored group is
  remembered too, so a group re-enabled after a reload — even through
  the native group menu — gets its own color back.
- The row values always reflect the live graph: toggling a group via the
  native group menu (or a second Group Switch node) updates the row on
  the next sync tick.
- Only the root canvas is mirrored; groups inside subgraphs are not
  listed.
- Duplicate group titles are disambiguated in the output keys as
  `Title`, `Title (2)`, …
- Without the web assets the node still loads and validates; drive it by
  setting the `group_states` JSON (and optionally `sort_order`) manually.

### Seed History

A zero-input seed generator (`Lukutar/Seeds`): the same `seed` +
`control_after_generate` widget pair as the built-in seed inputs, plus a
`history_size` log. The seed wires into any sampler via its INT output.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| seed | INT | 0 | Starting seed value (0 … 2⁶⁴−1). |
| history_size | INT | 5 | How many past seeds the rolling log keeps (1–1000). |

| Name | Type | Description |
|------|------|-------------|
| seed | INT | The seed for the current generation; connect it to any `seed`/`noise_seed` input. |

#### Behaviour

- `control_after_generate` is added by the ComfyUI frontend to every
  `seed` widget and applied there (fixed / increment / decrement /
  randomize). It is deliberately not a backend input — the frontend
  never sends it, so API workflows submit only `seed` (and
  `history_size`).
- The web extension keeps a rolling log of the seeds actually sent to
  the backend (newest first, older seeds pushed down, trimmed to
  `history_size`) and renders it
  below the parameters, one seed per line, using the same widget the
  core "Preview as Text" node displays its text with (plain text only
  — the Markdown toggle it ships with is removed). While the log is
  empty the panel shows a placeholder line, so the field is always
  visible. The log is copied with the copy button in the panel's
  top-right corner. Consecutive duplicates are collapsed, so
  re-queueing with a fixed seed does not fill the log with the same
  number.
- The log is stored in a node property, so the history survives
  reloads. Logs saved by older builds (kept in a hidden widget value)
  are adopted on load.
- If the frontend does not expose the "Preview as Text" helper, the
  extension falls back to a plain text widget; without the web assets
  the node still works as a plain seed source — the history is simply
  not displayed.

### Load Checkpoint (Bundle)

One-node checkpoint loader for an A1111-style txt2img pipeline
(`Lukutar/Bundle`). Loads the checkpoint, applies an optional weighted
LoRA list, encodes both prompts, creates the empty latent and emits
two entities:

- **bundle** (new socket type **LUKUTAR_BUNDLE**) — a dict carrying
  `model`, `clip`, `vae`, `pos_cond`, `neg_cond`, `latent` and `img`
  (initially `None`); consumed by the KSampler (Bundle) and re-emitted
  by it.
- **info** (new socket type **LUKUTAR_INFO**) — an ordered
  `dict[str, str]` describing every parameter for later A1111 metadata
  conversion (see "Info entity" below).

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| ckpt_name | COMBO | — | Checkpoint file. |
| clip_skip | INT | -2 | CLIP layer to stop at; -1 is the standard output layer. |
| vae | COMBO | baked | `baked` uses the checkpoint's own VAE; any other entry loads a standalone VAE file. |
| positive / negative | STRING (multiline) | "" | Prompts, encoded with the weighted prompt engine. |
| width / height | INT | 1024 | Latent geometry in pixels, step 8. |
| token_normalization | COMBO | mean | `none`, `mean`, `length`, `length+mean` — how token weights are normalized before interpretation. |
| weight_interpretation | COMBO | comfy | `comfy`, `A1111`, `compel`, `comfy++`, `down_weight` — how `(text:weight)` syntax is applied. |
| batch_size | INT | 1 | Latent batch size. |

#### Inputs / Outputs

| Direction | Name | Type | Description |
|-----------|------|------|-------------|
| input (optional) | lora_list | LUKUTAR_LORA_LIST | Weighted LoRA list from LoRA List. |
| output | bundle | LUKUTAR_BUNDLE | The generation bundle. |
| output | info | LUKUTAR_INFO | Textual info entity. |

#### Weighted prompt engine

`comfy` is ComfyUI's native weighting. `A1111` encodes the unweighted
prompt, scales the embedding from zero per token and renormalizes to
the base mean (A1111 semantics). `compel`, `comfy++` and `down_weight`
mix masked prompt variants, matching the BNK-style advanced encoders.
SD1.x/SD2.x (`l` key) and SDXL (`l` + `g`, pooled from `g`) take the
full engine; other model compositions (Flux, SD3, ...) fall back to
ComfyUI's native weighting with a log note.

#### Info entity

The loader fills the LUKUTAR_INFO mapping with, in order: `prompt`,
`negative_prompt`, `Model`, `Model hash`, `VAE`, `Clip skip`, `Lora`
(`name:weight` comma-joined), `Lora hashes` (A1111 `name: hash` style),
`Size` (`"{width} x {height}"`), `Batch size`, `Token normalization`
and `Weight interpretation`. Later pipeline nodes append their own
entries to a copy of this mapping; the input mapping is never mutated.

#### Behaviour

- The latent is created with the loaded model's own channel count, so
  SD1.x, SDXL and newer architectures all work from the same node.
- Each LoRA is applied to both model and CLIP with the same weight;
  zero-weight entries are skipped. Files are cached per node instance.
- The checkpoint and LoRA hashes are computed through the same
  on-disk-cached SHA-256 engine as the A1111 nodes.
- The bundle's `img` slot starts as `None`; the Checkpoint Bundle
  Unpack node reports it as an empty IMAGE tensor until a sampler fills
  it.

### LoRA List

Collects up to eight LoRAs with per-LoRA weights into one
**LUKUTAR_LORA_LIST** socket for Load Checkpoint (Bundle)
(`Lukutar/Bundle`). No
inputs. Each row is a `lora_N` combo (with `None` marking an unused
slot) and a `weight_N` float in [-3.0, 3.0], step 0.01, applied to both
model and CLIP.

The web extension `web/js/lora_list.js` shows every claimed pair
plus exactly one free trailing pair and hides the rest (same interaction
as A1111 Lora Hashes). Without the web assets the node still works —
all eight pairs are simply visible.

Zero-weight and unused entries are dropped by the backend; the output
is a tuple of `(name, weight)` pairs in slot order.

### Checkpoint Bundle Unpack / Checkpoint Bundle Pack

Convert between the LUKUTAR_BUNDLE entity and standard ComfyUI sockets
(`Lukutar/Bundle`).

**Checkpoint Bundle Unpack** — input `bundle` (LUKUTAR_BUNDLE), outputs
`model` (MODEL), `clip` (CLIP), `vae` (VAE), `pos_cond` (CONDITIONING),
`neg_cond` (CONDITIONING), `latent` (LATENT) and `img` (IMAGE). The
image output is an empty `(0, h, w, 3)` tensor (geometry from the
bundle latent) while no sampler has decoded one yet.

**Checkpoint Bundle Pack** — inputs `model`, `clip`, `vae`, `pos_cond`,
`neg_cond`, `latent` and optional `img` (IMAGE), output `bundle`
(LUKUTAR_BUNDLE).

### KSampler (Bundle)

Samples the bundle's latent exactly like the core KSampler and
re-emits the bundle plus the info entity (`Lukutar/Bundle`).

#### Inputs / Outputs

| Direction | Name | Type | Description |
|-----------|------|------|-------------|
| input | bundle | LUKUTAR_BUNDLE | Bundle from Load Checkpoint (Bundle) (or another sampler). |
| input | info | LUKUTAR_INFO | Info entity to extend. |
| output | bundle | LUKUTAR_BUNDLE | Copy of the input bundle with the sampled latent. |
| output | info | LUKUTAR_INFO | Copy of the input info with the sampling entries appended. |

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| seed | INT | 0 | Noise seed, with the standard `control_after_generate` behavior. |
| steps | INT | 20 | Denoising step count. |
| cfg | FLOAT | 7.0 | Classifier-free guidance scale (0–100). |
| sampler_name | COMBO | — | ComfyUI sampler list. |
| scheduler | COMBO | — | ComfyUI scheduler list. |
| denoise | FLOAT | 1.0 | Denoise strength; below 1.0 keeps the input latent structure (img2img). |
| vae_decode | BOOLEAN | True | Decode the finished samples to IMAGE. |

#### Preview behaviour

- While sampling, the node shows the standard ComfyUI latent preview —
  the same progress stream every core sampler emits, honouring the
  server's preview-method settings (TAESD / Latent2RGB).
- With `vae_decode` enabled, the finished samples are decoded through
  the bundle's VAE, stored into the bundle's `img` slot, and the first
  frame is pushed through the same binary preview channel the latent
  previews use — so the frontend **replaces** the last noise frame
  with the decoded image. This matters because the v2 frontend's
  output display prefers live preview frames over executed outputs
  (which are only revoked on the next run), and the legacy canvas
  renderer draws `node.imgs`, which is fed exclusively by preview
  frames.
- The decoded frames are also saved as temp PNGs (same layout as
  ComfyUI's own PreviewImage: `LukutarBundleKSampler_temp_00001_.png`,
  …, in the ComfyUI temp directory) and emitted as the `executed`
  output ui. The `web/js/ksampler_bundle.js` extension consumes them:
  it keeps the v2 output section hidden (`node.hideOutputImages` — the
  preview channel already shows the same picture, and the section
  would render it a second time) and fills `node.imgs` with `/view`
  URLs of the temp files, which is what the right-click **Open Image /
  Copy Image / Save Image** menu entries are built from — so the
  picture can be copied or saved like any PreviewImage output. Without
  the web extension the pack stays functional but the v2 node shows
  the picture twice (the legacy canvas renderer is unaffected).
- With `vae_decode` disabled the bundle's `img` slot stays as it was
  (empty on a fresh loader bundle), no temp files are written and the
  last latent frame remains as the node preview.
- The sampling result is never cached: an `OUTPUT_NODE` re-executes on
  every queue, matching A1111's re-run semantics.
- A failed temp-preview save or preview push is logged and does not
  abort sampling.

#### Info entries appended

`Seed`, `Steps`, `CFG scale`, `Sampler`, `Schedule type`,
`Denoising strength`, `VAE decode` (`enabled`/`disabled`) — appended to
a copy of the input info; the input mapping is never mutated.

### Info to A1111 Parameters

Converts a LUKUTAR_INFO entity into the A1111 `parameters` metadata
block (`Lukutar/Info`). One input (`info`), one output (`metadata`,
LUKUTAR_METADATA): `{"parameters": text}` where *text* is the full
A1111 infotext — prompt line, optional `Negative prompt:` line, then
the settings block in info order — composed by the same
`build_parameters` engine the A1111 nodes use. The prompt keys are
never emitted as settings pairs; an empty info yields an empty
metadata dict. The output plugs into Merge Metadata, Save Image
(Metadata) or any LUKUTAR_METADATA consumer.

#### A1111 compatibility filtering

Entries with no official A1111 infotext counterpart are removed and
A1111-convention values are normalised (`a1111_settings` in
`core/generation_info.py`), so external A1111 / Forge PNG-info parsers
never see pack-specific data:

- Dropped keys: `Batch size`, `Lora`, `Token normalization`, `Weight
  interpretation`, `VAE decode` — A1111's infotext format has no such
  fields (LoRA weights live in `<lora:name:weight>` prompt tags, hashes
  in `Lora hashes`).
- Dropped values: `VAE: baked` (ComfyUI-ese; A1111 expects a VAE file
  name). A real VAE name is kept.
- `Clip skip` is emitted as the positive layers-to-skip count
  (`-2` → `Clip skip: 2`); a count of `1` (no skipping) drops the key,
  like A1111 itself.
- `Size` `1024 x 1536` is normalised to A1111's space-free
  `1024x1536` (parsers regex `(\d+)x(\d+)`).
- `Sampler` / `Schedule type` values are translated to native A1111
  names (`euler_ancestral` → `Euler a`, `sgm_uniform` → `SGM Uniform`,
  …); names outside the translation maps pass through unchanged.
- `Lora hashes` entry names lose their model-file extension
  (`Mawshot (1).safetensors: 9ec5889be9` → `Mawshot (1):
  9ec5889be9`) — A1111 matches LoRAs by extension-free name/hash.
- Entries are reordered into the canonical A1111 infotext order
  (`Steps`, `Sampler`, `Schedule type`, `CFG scale`, `Seed`, `Size`,
  `Model hash`, `Model`, … `Lora hashes`, …); pairs such as `Model
  hash, Model` stay adjacent for parsers that match them as a unit.
- Any other (user-added) info entries pass through verbatim, after the
  known keys.

### Info Unpack

Reads the standard info entries as named STRING outputs
(`Lukutar/Info`): `model_name`, `model_hash`, `vae_name`, `lora`,
`lora_hashes`, `prompt`, `negative`, `size`, `batch_size`,
`clip_skip`, `token_normalization`, `weight_interpretation`, `seed`,
`steps`, `cfg`, `sampler`, `scheduler`, `denoise`, `vae_decode`.
Missing entries yield `""`. The trailing `metadata` output is a copy
of the whole mapping as LUKUTAR_METADATA.

### Info Pack

Assembles a LUKUTAR_INFO entity (`Lukutar/Info`): an optional
`info_in` base extended by up to four `key_N`/`value_N` string pairs
(later pairs win, empty keys are skipped, values are stripped). The
web extension `web/js/info_pack.js` shows every claimed pair plus
one free trailing pair and hides the rest; without the web assets the
node still works with all four pairs visible.

### Save Image (Metadata)

Saves an IMAGE batch to disk, one file per frame, in a chosen format
(`Lukutar/Image`). Optionally embeds LUKUTAR_METADATA key-value pairs
into the file headers where the format supports it. The node is an
`OUTPUT_NODE` (always re-executes), so it never returns images — files
are written directly to the filesystem.

A new socket type, **LUKUTAR_METADATA**, carries an ordered set of named
metadata blocks (`dict[str, str]`). It is produced by the three metadata
nodes and consumed by this save node.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| images | IMAGE | — | Batch of frames to save. |
| path | STRING | "output" | Target directory. Relative paths resolve against the ComfyUI root (`folder_paths.base_path`); absolute paths are used as-is. Missing directories are created automatically. |
| filename | STRING | "img" | Base file name (extension added automatically). Forbidden characters for Windows filenames are replaced with underscores. |
| format | COMBO | PNG | Output format: `PNG`, `JPG`, `WEBP`, `TIFF`, `BMP`. |
| quality | INT | 95 | Compression quality, 1–100. Used by `JPG` and `WEBP`; ignored by `PNG`, `TIFF`, `BMP`. For JPEG-XL use the dedicated **Save Image JXL** node (see below). |

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| metadata | LUKUTAR_METADATA | — | Optional metadata blocks to embed. Hidden when the selected format cannot store metadata (e.g. BMP). |

#### Behaviour

- One file per frame is written. Multi-frame batches receive indexed
  names: `filename_00000`, `filename_00001`, … (zero-padded five digits).
  A single-frame batch keeps the bare filename without an index.
- When a target file already exists, a collision suffix `_00001`,
  `_00002`, … is appended so repeated runs never overwrite.
- Metadata embedding per format:
  - **PNG** — one `tEXt` chunk per key/value pair.
  - **JPG / WEBP** — EXIF UserComment tag (0x9286); the payload
    is `UNICODE\0` + UTF-16-encoded JSON of the metadata dict, matching
    A1111-style readers.
  - **TIFF** — ImageDescription tag (270) holding the JSON-encoded
    metadata dict.
  - **BMP** — no metadata embedding.
- If the selected format cannot store metadata and the `metadata` input
  is unconnected, the web extension hides the socket entirely. A still-
  connected link triggers a backend warning at save time and degrades
  gracefully (image saved without metadata).
- If embedding metadata fails for any reason, the node retries saving
  the image without metadata and logs a warning; it never raises on
  that single frame.
- The node preview (`ui.images`) lists saved files only when they land
  inside ComfyUI's output directory.
- Without the web extension the node is fully functional — the metadata
  socket is always declared and can be wired manually.
- The pass-through outputs let you chain this saver mid-graph: downstream
  nodes receive the original batch and metadata without extra wiring.

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| images | IMAGE | The input batch, passed through untouched. |
| metadata | LUKUTAR_METADATA | The input metadata blocks, passed through; `None` when the metadata input is not connected (Merge Metadata skips `None` blocks). |

### Save Image JXL

Dedicated JPEG-XL saver backed by the `pillow-jxl-plugin` package
(`Lukutar/Image`). Supports both lossy and lossless modes with full
encoder control. Pass-through outputs (`images`, `metadata`) let you chain
	the saver mid-graph without rewiring; it is an `OUTPUT_NODE` that always
	re-executes (mirrors ComfyUI's own SaveImage behaviour).

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| images | IMAGE | — | Batch of frames to save. |
| path | STRING | "output" | Target directory. Relative paths resolve against the ComfyUI root (`folder_paths.base_path`); absolute paths are used as-is. Missing directories are created automatically. |
| filename | STRING | "img" | Base file name (extension added automatically). Forbidden characters for Windows filenames are replaced with underscores. |
| lossless | BOOLEAN | off | Lossless (mathematically exact) encoding. When on, output pixels are identical to the input and `quality` is ignored. |
| quality | INT | 90 | JPEG-style lossy quality, 0–100. `100` ≈ visually lossless. Ignored in lossless mode. |
| effort | INT | 7 | Encoder effort, 1–9. Higher values compress better but run slower. |
| decoding_speed | INT | 0 | Decode-speed tier, 0–4. `0` = smallest files / slowest decode; `4` = fastest decode / larger files. |
| use_container | BOOLEAN | off | Wrap the codestream in the JXL container format. |
| use_original_profile | BOOLEAN | off | Keep the original color profile (skip YCbCr conversion). |
| num_threads | INT | -1 | Encoder threads; `-1` = automatic (up to 64). |
| compress_metadata | BOOLEAN | off | Compress the embedded metadata boxes. |

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| metadata | LUKUTAR_METADATA | — | Optional metadata blocks written into the file as an EXIF box. |

#### Behaviour

- One `.jxl` file per frame is written. Multi-frame batches receive
  indexed names: `filename_00000`, `filename_00001`, … (zero-padded
  five digits). A single-frame batch keeps the bare filename without
  an index.
- When a target file already exists, a collision suffix `_00001`,
  `_00002`, … is appended so repeated runs never overwrite.
- **Lossless mode.** When `lossless` is on the encoder forces quality
  to 100; output pixels are mathematically identical to the input. The
  test suite asserts pixel-exact round-trips in lossless mode.
- Metadata is embedded as a JXL EXIF box — the same LUKUTAR_METADATA
  dict format used by the generic saver, stored as JSON inside the EXIF
  UserComment tag (`UNICODE\0` + UTF-16).
- Without the web extension the node is fully functional. Node preview
  (`ui.images`) shows saved files only when they land inside ComfyUI's
  output directory.
- **Dependency.** `pillow-jxl-plugin` is listed in
  `requirements.txt` (alongside `opencv-python`). ComfyUI-Manager /
  comfy-cli install it automatically with the pack. The wheels statically
  bundle libjxl, so no system `libjxl` or other JXL package is needed.
  If the plugin is missing at execution time the node raises a clear
  `RuntimeError` with the install command.
- The pass-through outputs let you chain this saver mid-graph: downstream
  nodes receive the original batch and metadata without extra wiring.

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| images | IMAGE | The input batch, passed through untouched. |
| metadata | LUKUTAR_METADATA | The input metadata blocks, passed through; `None` when the metadata input is not connected (Merge Metadata skips `None` blocks). |

### Metadata Blocks

Builds a LUKUTAR_METADATA dict from up to 16 editable name/value pairs
(`Lukutar/Metadata`). Names are single-line strings; values support
multiline text. The `num_blocks` parameter controls how many pairs are
actually processed (default 1, range 1–16). Pairs with an empty (after
trimming) name are skipped. Duplicate names resolve **later-wins**.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| num_blocks | INT | 1 | How many name/value pairs to process (1–16). The web extension hides pairs beyond this count on the canvas. |

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| name_1 … name_16 | STRING | "" | Block label for each pair. Empty names are skipped. |
| value_1 … value_16 | STRING (multiline) | "" | Block content for each pair. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Ordered `dict[str, str]` of the processed pairs. |

#### Behaviour

- Names and values are trimmed of leading/trailing whitespace before
  processing. Non-string values are coerced via `str()`.
- The node statically declares all 16 name/value pair inputs so it
  works fully without the web extension. The extension
  (`web/js/metadata_blocks.js`) hides pairs beyond `num_blocks` at
  runtime by setting `hidden=true` and `computeSize=()=>[0,-4]`, which
  is the pack's established idiom — hidden widgets still serialize, so
  values survive workflow save/load. Without the extension all 16 pairs
  remain visible and the node functions normally.

### Standard Metadata

Emits standard ComfyUI metadata entries as a LUKUTAR_METADATA dict
(`Lukutar/Metadata`). Three toggles control which keys are emitted, in
a fixed order: `prompt`, `workflow`, `datetime`. Keys whose toggle is
off (or whose hidden input is missing) are omitted entirely.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| include_workflow | BOOLEAN | true | Include the workflow JSON (ComfyUI's hidden EXTRA_PNGINFO). |
| include_prompt | BOOLEAN | true | Include the execution prompt JSON (ComfyUI's hidden PROMPT input — the entire graph, not just this node's widgets). |
| include_datetime | BOOLEAN | false | Add the current local date/time. |
| datetime_format | STRING | "%Y-%m-%d %H:%M:%S" | strftime format used when `include_datetime` is on. Blank falls back to the default. |

#### Hidden Inputs

| Name | Type | Description |
|------|------|-------------|
| prompt | PROMPT | ComfyUI's full execution prompt dict (the entire graph). |
| extra_pnginfo | EXTRA_PNGINFO | ComfyUI's hidden extra PNG info dict. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Ordered `dict[str, str]` with the selected keys. |

#### Behaviour

- `prompt` — JSON-serialised ComfyUI execution prompt (the hidden
  `PROMPT` input containing the entire graph).
- `workflow` — JSON-serialised `extra_pnginfo["workflow"]`.
- `datetime` — current **local** time formatted per `datetime_format`.
- Keys are always emitted in this fixed order when their flag is on:
  `prompt`, then `workflow`, then `datetime`.
- Values are coerced via `str()` during JSON serialisation to handle
  non-string types inside the prompt/workflow dicts.

### Merge Metadata

Merges several LUKUTAR_METADATA blocks into one **without duplicating
existing data** (blocks processed in slot-number order)
(`Lukutar/Metadata`). The backend declares zero inputs; slots are grown
client-side by `web/js/metadata_merge.js` using the same SEGS-Concat /
Set-Bundle growth mechanism.

#### Parameters

None — all inputs are dynamic.

#### Inputs (dynamic)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| + add | LUKUTAR_METADATA | — | Always-present free input. Connecting a wire onto it claims the lowest unused `meta_N` slot and spawns the next free input. Disconnecting an auto-named slot collapses it. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Merged dict; blocks processed in `meta_1`, `meta_2`, … order, entries resolved per the smart-merge rules below. |

#### Behaviour

- Slots are named `meta_1`, `meta_2`, … (up to 16). The backend sorts
  incoming kwargs by natural `meta_N` order and merges sequentially.
- Every incoming entry (blank names/values are skipped) is resolved by
  three tiers, in order:
  1. **Same-name block** — the name already exists in the result: the
     values are joined (comma for single-line values, newline when
     either side is multiline). Two `Lora hashes` blocks therefore
     become one combined block.
  2. **Settings-line fold** — the name appears as a `"Key: value"`
     entry inside another block's *last* line (the A1111 infotext
     settings line). The data is folded into that entry with A1111
     quoting honoured (values containing commas are quoted), so e.g. a
     `Lora hashes` block merges into the `parameters` block's
     `Lora hashes: "…"` entry instead of becoming a separate block.
  3. **New block** — otherwise the entry is appended as a new block.
- `None` inputs are skipped during merge. All names/values are coerced
  via `str()` and stripped; input mappings are never mutated.
- Without the web extension the node is not wirable from the canvas
  (same trade-off as Set/Get Bundle). Hand-written API prompts may pass
  `meta_1`, `meta_2`, … links, which the backend orders by natural
  `meta_N` order.

### A1111 Parameters

Builds an A1111-style generation-parameter infotext string and a
corresponding LUKUTAR_METADATA block (`Lukutar/Metadata`). The output
is byte-compatible with A1111 / Forge PNG-info parsers: the positive
prompt line, an optional `"Negative prompt:"` line, and a settings
block where every key/value pair is rendered as `key: value`, joined by
`, `. Values that contain a comma, newline or colon are JSON-encoded
(`quote_a1111`); everything else is emitted verbatim.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| positive | STRING (multiline) | "" | Positive prompt text; may span multiple lines. Always present in the output. |
| negative | STRING (multiline) | "" | Negative prompt text. The `"Negative prompt:"` line is omitted when this field is empty. |
| version | STRING | "" | Value for the trailing `"Version:"` entry (e.g. `"forge-0.9.1"`). Omitted when empty. |

#### Inputs (dynamic)

| Name | Type | Default | Description |
|------|------|---------|-------------|
| + add | LUKUTAR_METADATA | — | Always-present free input. Connecting a wire onto it claims the lowest unused `meta_N` slot and spawns the next free input. Slots are processed in `meta_1`, `meta_2`, … order, which determines settings-key order in the output. Disconnecting an auto-named slot collapses it. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Single-entry dict `{"parameters": <infotext>}` when the infotext is non-empty; empty dict when the whole text would be blank (both prompts empty and no settings). |
| parameters_text | STRING | The raw A1111-style infotext string, including the prompt lines and settings block. |

#### Behaviour

- Settings keys come from whatever metadata blocks are wired into the
  `meta_N` inputs; their order in the output follows slot number
  (`meta_1` before `meta_2`, …). A `"Version"` key is appended last if
  the `version` parameter is non-empty.
- The infotext format: `<positive>\n<Negative prompt: <negative>>\n<key: value, key: value>` — each settings entry uses `quote_a1111()` on its value, which JSON-encodes values containing `,`, `\n`, or `:` and leaves simple strings untouched.
- The `"Negative prompt:"` line appears only when the `negative` field
  is non-empty (truthy check, no strip).
- When the resulting text is entirely empty (no positive, no negative,
  no settings), the `metadata` output key `"parameters"` is omitted and
  `parameters_text` is `""`.

#### Recommended graph

Wire **A1111 Sampling**, **A1111 Hires Fix**, **A1111 Model Info** and
**A1111 Lora Hashes** into the metadata slots of this node. Feed the seed
from a **Seed History** node (the `seed` widget on A1111 Sampling is
display-only; its `control_after_generate` is added by the frontend).
Pass positive/negative prompts from **Prompt Annotate**'s clean STRING
output or type them directly. Merge the resulting `"parameters"` metadata
block with Standard Metadata (or any other blocks) through **Merge
Metadata** before a save node.

### A1111 Sampling

Emits standard A1111 settings entries for sampler configuration as a
LUKUTAR_METADATA block (`Lukutar/Metadata`). Keys are emitted in the
fixed A1111 order: Steps, Sampler, CFG scale, Seed, [Size], [Clip skip].

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| steps | INT | 20 | Number of inference steps (1–150). |
| sampler | COMBO | DPM++ 2M Karras | Sampler algorithm. The combo lists 25 common A1111 samplers; arbitrary names can be added via the node's context menu → "convert widget to input". |
| cfg | FLOAT | 7.0 | Classifier-free guidance scale (0–30, step 0.1). Rendered without trailing zeros (`7` not `7.0`). |
| seed | INT | 0 | Generation seed (0 … 2⁶⁴−1). The frontend adds its own `control_after_generate` widget; declaring it server-side would conflict with the Seed History node's handling. |
| width | INT | 1024 | Image width in pixels (0–16384, step 8). `0` on either dimension omits the "Size" entry entirely. |
| height | INT | 1024 | Image height in pixels (0–16384, step 8). |
| clip_skip | INT | 0 | CLIP skip value (0–12). `0` = omit the "Clip skip" entry; values ≥ 1 are emitted as integers. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Dict with keys `Steps`, `Sampler`, `CFG scale`, `Seed`, optionally `Size` (when both width and height are > 0) and optionally `Clip skip` (when ≥ 1). CFG scale uses Python's `:g` format specifier so trailing zeros are dropped. |

### A1111 Hires Fix

Emits standard A1111 settings entries for high-resolution fix as a
LUKUTAR_METADATA block (`Lukutar/Metadata`). Keys are emitted in this
fixed order: Hires upscale, Hires upscaler, [Hires steps], Denoising strength.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| upscaler | STRING | 4x-UltraSharp | Upscaler display name (free text). Matches whatever A1111/Forge shows in its upscaler dropdown. |
| upscale | FLOAT | 2.0 | Upscaling factor (1–8, step 0.05). Rendered as `"2.0"` with one decimal place (A1111 convention). |
| hires_steps | INT | 10 | Hires pass steps (0–150). `0` = omit the "Hires steps" entry. |
| denoising | FLOAT | 0.45 | Denoising strength for the hires pass (0–1, step 0.01). Rendered as `"0.45"` with one decimal place. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Dict with keys `Hires upscale` (float string), `Hires upscaler`, optionally `Hires steps` (integer when ≥ 1), and `Denoising strength` (float string). Values use `str(float(value))` so integer inputs like `2` produce `"2.0"`. |

### A1111 Model Info

Looks up a model file by type, computes its SHA-256 prefix, and emits
type-specific name/hash entries matching A1111 conventions
(`Lukutar/Metadata`). Four model types are supported: Checkpoint, VAE,
LoRA, and Embedding.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| model_type | COMBO | Checkpoint | Which model folder to list. The web extension refills the file combo when this changes; without it the combo shows checkpoints only. |
| model_name | COMBO | (first checkpoint) | Model file to look up. Populated from `folder_paths` at node load time. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| model_name | STRING | Display name: checkpoint/LoRA/embedding stem (extension removed), VAE name with extension preserved — matching A1111's own convention. |
| model_hash | STRING | First 10 hex characters of the file's SHA-256 digest. |
| metadata | LUKUTAR_METADATA | Type-specific entries: Checkpoint → `"Model hash"` + `"Model"`; VAE → `"VAE hash"` + `"VAE"`; LoRA → `"Lora hashes": "name: hash"`; Embedding → `"TI hashes": "name: hash"`. |

#### Behaviour

- **Hash computation.** The SHA-256 is computed by streaming the file in
  1 MiB chunks (never loads the full file into memory). Hashing a
  multi-GB checkpoint takes tens of seconds **once**; subsequent uses
  read from an on-disk cache.
- **Cache.** A `ModelHashCache` singleton persists entries to
  `a1111_hash_cache.json` in the pack root directory (one level above
  `core/`). Entries are keyed by absolute path and validated against the
  file's current mtime and size — if either changes, the cache entry is
  invalidated and the hash is recomputed. The cache writes atomically
  (temp file + rename).
- **Web extension.** The frontend (`web/js/a1111_model_info.js`) listens
  for `model_type` changes and refills the `model_name` combo via
  `/models/<folder>` — so switching from Checkpoint to LoRA shows only
  `.safetensors`/`.sft` files in the loras folder. Without the extension
  the combo always lists checkpoints (the backend default).
- **Runtime requirement.** The node raises a `RuntimeError` when called
  outside the ComfyUI runtime (`folder_paths` is `None`), since it needs
  the model directory lookup.

### A1111 Lora Hashes

Computes SHA-256 prefixes for up to eight LoRA files and emits a single
`"Lora hashes"` metadata entry in A1111 format
(`Lukutar/Metadata`). The `"Lora hashes"` value is a raw, comma-separated
list of `"name: hash"` pairs; quoting (JSON-encoding) is applied only when
the **A1111 Parameters** node composes the infotext via `quote_a1111` —
byte-compatible with A1111's own parser.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| lora_1 … lora_8 | COMBO | None | LoRA file from the `loras` folder. `"None"` (the first entry) marks an unused slot. The web extension hides unused slots and always shows one free trailing slot; without it all eight combos are visible. |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| metadata | LUKUTAR_METADATA | Single-entry dict `{"Lora hashes": "name: hash, name: hash"}` (raw; A1111 quoting is applied once when the **A1111 Parameters** node composes the infotext). Empty dict when all slots are `"None"` or unconnected. |
| lora_hashes | STRING | The raw comma-joined `"name: hash"` string (not pre-quoted), for inspection or manual editing. |

#### Behaviour

- Slots are processed in numbered order (`lora_1` through `lora_8`).
  Entries with `"None"` or blank values are skipped entirely. If no
  real slots are active the node returns empty outputs without touching
  `folder_paths`.
- **Why not multiple Model Info nodes?** Each A1111 Model Info emits a
  `"Lora hashes"` key, and **Merge Metadata** folds duplicate names
  together — so wiring two Model Info nodes would append the second
  node's hash to the first node's entry. This node aggregates all
  hashes into a single raw value
  in a single metadata block (A1111 quoting is applied once by the
  **A1111 Parameters** node), avoiding that conflict entirely.
- **Runtime requirement.** The node raises a `RuntimeError` when called
  outside the ComfyUI runtime and at least one non-"None" slot is active.
- **Web extension.** The frontend (`web/js/a1111_lora_hashes.js`) hides
  every unclaimed slot and shows exactly one free trailing combo, so the
  user can keep adding LoRA inputs without seeing eight blank dropdowns.

### Paste (Clipspace, Keep Mask) / Paste Image (Keep Mask)

Not nodes — two context-menu items the pack's web extension adds to
image-upload nodes (`Load Image`, `Load Image (impact)`, anything with
an `image_upload` input).

ComfyUI stores a painted mask in the **alpha channel of the very file**
the image widget points at; the stock pastes swap the file —
`Paste (Clipspace)` points the widget at the clipspace file, and the
Nodes-2.0 `Paste Image` uploads the clipboard image — so the mask dies
with it. These items paste the new image but re-bake the current
file's alpha (i.e. the mask) on top of it, uploading the combined PNG
as a new input file — the same save flow the MaskEditor uses.

Two items:

- **Paste (Clipspace, Keep Mask)** — pastes the clipspace's selected
  image, ignoring `img_paste_mode`; the keep-mask sibling of the stock
  `Paste (Clipspace)`.
- **Paste Image (Keep Mask)** — pastes the first image on the system
  clipboard; the keep-mask sibling of the Nodes-2.0 `Paste Image`
  (which is what the new frontend's paste flow is built on). Works on
  the legacy frontend too.

#### Behaviour

- On the Vue-based Nodes-2.0 menu both items sit in the **Extensions**
  section at the bottom of the menu (next to other packs' items); on
  the legacy frontend, directly in the node menu. The media section
  with icons (`Paste Image` & co.) is built Vue-side and cannot be
  extended by third-party packs.
- **Paste (Clipspace, Keep Mask)** appears whenever the clipspace
  holds anything pasteable: its selected image, or — recent frontends
  only copy node *outputs* into `clipspace.images`, which input nodes
  without a run don't have — the snapshotted image widget value,
  exactly what the stock `Paste (Clipspace)` restores.
- **Paste Image (Keep Mask)** is always offered on image-upload nodes,
  like the stock `Paste Image`; it no-ops with a console note when the
  clipboard holds no image.
- If the sizes differ, the mask is stretched to the new image's
  dimensions.
- If the current file has no mask (no alpha / fully opaque), the item
  degrades to an ordinary paste — the clipspace item just points the
  widget at the clipspace reference and nothing is uploaded; the
  clipboard item uploads the pasted image as-is.
- The pasted image's own alpha (if any) is discarded; only the node's
  existing mask is kept.
- The combined PNG is encoded straight from the RGBA buffer, so RGB
  under semi-transparent mask areas is preserved bit-exactly (a plain
  canvas re-encode would black it out).
- Requires no backend node — with the extension disabled everything
  keeps working, the stock paste behaviour is unchanged.

## Installation

**Via ComfyUI-Manager:** Custom Nodes Manager → *Install via git URL* →
`https://github.com/Mistress-Lukutar/ComfyUI-LukutarNodes`

**Manual:**

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/Mistress-Lukutar/ComfyUI-LukutarNodes
```
Restart ComfyUI afterwards. `requirements.txt` is picked up automatically
by ComfyUI-Manager (`opencv-python`; torch and numpy ship with ComfyUI).

## Development

```bash
# Engine unit tests (any python with numpy, opencv-python and pytest)
pytest

# Node loading + end-to-end smoke test with ComfyUI's own python
"<ComfyUI>/python_embeded/python.exe" tests/smoke_test_comfyui_load.py
```

Layout:

```
core/    pure numpy/OpenCV engines, no ComfyUI imports
nodes/   ComfyUI node classes (INPUT_TYPES, tensor glue)
utils/   torch tensor conversion helpers
web/js/  frontend extension (rich prompt annotator input)
tests/   pytest suite + ComfyUI loader smoke test
```

The image nodes appear in ComfyUI under the `Lukutar/Image` category,
the prompt annotation nodes under `Lukutar/Prompt`.

## License

MIT — see [LICENSE](LICENSE).
