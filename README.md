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
- **Annotation Segment Edit** — pass-through per-label edits on
  ANNOTATIONS: prepend/append/remove text, new/delete segments
  (`Lukutar/Prompt`).
- **Set Variable / Get Variable** — named workflow variables: publish a
  value once, read it anywhere without dragging wires
  (`Lukutar/Variables`).
- **Paste (Clipspace, Keep Mask)** — context-menu item for image upload
  nodes: paste a clipspace image while keeping the node's painted mask
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
height grows the node. The built-in text widget is only
hidden, so the value keeps serializing into the workflow normally. The
**Annotate...** button opens a popup editor with the same
live-highlighted input, larger, plus a palette of the labels already
used in the text — click one to wrap the current selection. The node
works without the web assets too — the markup is plain text and can be
typed by hand.

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

### Paste (Clipspace, Keep Mask)

Not a node — a context-menu item the pack's web extension adds to
image-upload nodes (`Load Image`, `Load Image (impact)`, anything with
an `image_upload` input).

ComfyUI stores a painted mask in the **alpha channel of the very file**
the image widget points at; the stock `Paste (Clipspace)` swaps the
file, and the mask dies with it. This item pastes the clipspace's
selected image but re-bakes the current file's alpha (i.e. the mask)
on top of it, uploading the combined PNG as a new input file — the
same save flow the MaskEditor uses.

#### Behaviour

- The item appears in the node's context menu whenever the clipspace
  holds an image. On the current ComfyUI frontend it sits in the
  **Extensions** section at the bottom of the menu (next to other
  packs' items); on the legacy frontend, directly in the node menu.
- If the sizes differ, the mask is stretched to the new image's
  dimensions.
- If the current file has no mask (no alpha / fully opaque), the item
  degrades to an ordinary clipspace paste — nothing is uploaded.
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
web/js/  frontend extension (prompt annotator popup editor)
tests/   pytest suite + ComfyUI loader smoke test
```

The image nodes appear in ComfyUI under the `Lukutar/Image` category,
the prompt annotation nodes under `Lukutar/Prompt`.

## License

MIT — see [LICENSE](LICENSE).
