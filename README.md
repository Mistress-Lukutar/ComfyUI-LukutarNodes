# ComfyUI-LukutarNodes

A collection of custom nodes for
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) by Mistress-Lukutar.

The pack is organized as a home for standalone image-processing nodes:
each node lives in its own module under `nodes/`, backed by a pure
numpy/OpenCV engine under `core/` (no ComfyUI imports, unit-testable
anywhere) and thin tensor-conversion helpers under `utils/`.

## Nodes

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
stable pastel color per label (dark-theme friendly). The field grows
with its content automatically. The built-in text widget is only
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
| annotations | ANNOTATIONS | Parsed spans; consumed by the two nodes below. |
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
