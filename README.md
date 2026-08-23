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
untouched high-frequency detail of the input:

```
input ──┬─ Gaussian(σ) ──> low ──> color transfer <── Gaussian(σ) ── reference
        └───────────────────────> high ──────────────────┐
                                                        ├── + ──> output
```

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

## Installation

**Via ComfyUI-Manager:** Custom Nodes Manager → *Install via git URL* →
`https://github.com/Mistress-Lukutar/ComfyUI-LukutarNodes`

**Manual:**

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/Mistress-Lukutar/ComfyUI-LukutarNodes
```

Or, to develop straight from a checkout (Windows, no copying):

```bat
mklink /J "<ComfyUI>\custom_nodes\ComfyUI-LukutarNodes" "C:\_Source\ComfyUI-LukutarNodes"
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
tests/   pytest suite + ComfyUI loader smoke test
```

The node appears in ComfyUI under the `Lukutar/Image` category.

## License

MIT — see [LICENSE](LICENSE).
