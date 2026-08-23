# ComfyUI-LukutarNodes

Custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
by Mistress-Lukutar.

## Nodes

### Color Match (Frequency Split)

Restores the color distribution of a **reference** image onto a
**processed** image (for example an SD upscale) while keeping the processed
image's detail.

The node splits the input into frequency layers with a Gaussian blur
(`sigma` = cutoff frequency), transfers the color statistics of the
reference onto the low-frequency layer, and recombines it with the
untouched high-frequency detail of the input:

```
input ──┬─ Gaussian(σ) ──> low ──> color transfer <── Gaussian(σ) ── reference
        └───────────────────────> high ──────────────────┐
                                                        ├── + ──> output
```

Typical use case: SD iterative upscaling drifts colors through repeated VAE
encode/decode cycles. Feed the pre-drift image as `reference` and the
upscaled result as `image` to bring the colors back without losing the
upscaled detail.

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
- Auto-tune searches on the first frame pair only, then applies the winning
  sigma to the whole batch (consistent processing for video).

## Installation

**Via ComfyUI-Manager:** Custom Nodes Manager → *Install via git URL* →
URL of this repository.

**Manual:**

```bash
cd <ComfyUI>/custom_nodes
git clone <url-of-this-repo>
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
"C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe" tests/smoke_test_comfyui_load.py
```

Layout: `core/` — pure numpy/OpenCV engines, no ComfyUI imports;
`nodes/` — ComfyUI node classes; `utils/` — tensor conversion helpers.

## License

MIT — see [LICENSE](LICENSE).
