# AGENTS.md — ComfyUI-LukutarNodes

ComfyUI custom node pack by Mistress-Lukutar. Nodes:
**Color Match (Frequency Split)** — restores a reference image's color
distribution onto a processed image (e.g. an SD upscale) via Gaussian
frequency separation; **SEGS BBox Overlay** — draws Impact Pack
SEGS detections on an image, YOLO-demo style; **SEGS Set Crop Size** —
refits each segment's crop region to an absolute target size (exact
W×H or aspect-scaled to the target's longer side, rounded up to a
multiple, bbox always contained, cropped mask re-cut/zero-padded); and
the prompt
annotation set — **Prompt Annotate** (inline `|label: text|` markup in
one prompt, outputs ANNOTATIONS + clean STRING), **Annotations to
Wildcard (LAB)** (converts ANNOTATIONS to Impact Pack `[LAB]`
wildcard text), **Annotation Segment** (extracts one label's text),
**Annotation Labels** (labels as one comma-separated string) and
**Annotation Segment Edit** (pass-through ANNOTATIONS → ANNOTATIONS
editor: prepend/append/remove per label, new/delete segments;
comma-separated multi-label input); and the workflow-variable pair
**Set Variable** / **Get Variable** (publish any value under a name,
read it anywhere; `web/js/variables.js` maintains an invisible real
link between the pair — real so ordering/caching/API export are
ComfyUI's own; name resolution is mute-aware: several Sets may share a
name across alternative branches, one active).
The nodes' behavioral contract (inputs, batch semantics, auto-tune,
progress reporting, markup grammar) is specified in `README.md`; keep
the README in sync with any behavior change.

## Layout & architecture boundaries

- `core/` — pure numpy/OpenCV processing engines. **No ComfyUI imports,
  no torch** — must stay importable and unit-testable on any plain
  python that has numpy + opencv.
- `nodes/` — ComfyUI node classes (`INPUT_TYPES`, tensor glue, progress
  bar). `comfy` imports must be lazy and guarded with
  `try: from comfy... except ImportError: return None` so the module
  imports outside the ComfyUI runtime (pattern: `_make_progress_bar` in
  `nodes/color_match.py`). The prompt-annotation nodes and the
  variable nodes (`nodes/variables.py`) need neither torch nor comfy
  and stay pure-python.
- `web/js/` — frontend extensions served via `WEB_DIRECTORY` in the root
  `__init__.py` (rich highlighted input + popup editor for Prompt
  Annotate; invisible Set/Get Variable auto-connection; "Paste
  (Clipspace, Keep Mask)" context-menu item that re-bakes the current
  file's alpha-mask onto a pasted clipspace image, see README). Vanilla
  ES modules, no build step; the pack must stay fully functional
  without them (for Set/Get that means wiring the value input manually).
- `utils/` — torch IMAGE-tensor ⇄ numpy frame conversion helpers
  (B,H,W,3 float [0,1] RGB ⇄ HWC float32 [0,255] RGB).
- `tests/unit/` — engine tests, torch-free. `tests/smoke_test_comfyui_load.py`
  — node loading + end-to-end run, ComfyUI's embedded python only.

Adding a node: engine in `core/`, node class in `nodes/`, then register
it in `nodes/__init__.py` (`NODE_CLASS_MAPPINGS` +
`NODE_DISPLAY_NAME_MAPPINGS`). ComfyUI menu categories: `Lukutar/Image`
for image nodes, `Lukutar/Prompt` for the prompt-annotation nodes,
`Lukutar/Variables` for the variable pair.

## Commands

```bash
# Unit tests — any python with numpy, opencv-python, pytest (no torch)
pytest

# Lint (ruff: line-length 88, rules E,F,W,I,UP,B,SIM)
ruff check .

# Type check — plain `mypy core` FAILS (see gotchas); use:
mypy --explicit-package-bases core

# Node load + e2e smoke test — MUST use ComfyUI's embedded python
"C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe" tests/smoke_test_comfyui_load.py
```

The pack is **not pip-installed**; it is cloned into `ComfyUI/custom_nodes`.
`requirements.txt` carries only `opencv-python` — torch and numpy come
from the ComfyUI runtime. The `[project]`/`[tool.comfy]` tables in
`pyproject.toml` are registry + package metadata (no pip semantics);
pytest/ruff/mypy config lives in its tool sections.

## Releases (ComfyUI registry)

Published by CI, never by hand. Pushing a `v*` tag (or pressing "Run
workflow" in the Actions tab) makes `.github/workflows/publish_action.yml`
run Comfy-Org/publish-node-action with the `REGISTRY_ACCESS_TOKEN` repo
secret and publish the version written in `pyproject.toml` to the
registry — publisher `mistress-lukutar`, node id `lukutar-nodes`
(registry.comfy.org/publishers/mistress-lukutar).

```bash
# checks green first: pytest, ruff, mypy, smoke test (see Commands)
# bump the version in BOTH __init__.py.__version__ and pyproject.toml
git commit -m "chore(release): bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

- Verify: `gh run list --workflow=publish_action.yml` (expect success),
  then `curl -s https://api.comfy.org/nodes/lukutar-nodes/versions`
  must list the new version.
- A published version can never be republished — the registry answers
  400 "The node version already exists". A forgotten bump = red run.
- The registry node name (`lukutar-nodes` in `pyproject.toml`) is
  immutable after the first publish; `DisplayName` is changeable.
- The workflow deliberately does NOT fire on `pyproject.toml` edits:
  the file changes for non-release reasons too. The tag is the release.
- The package is every git-tracked file minus `.comfyignore` — when
  adding dev-only files, extend `.comfyignore` so they don't ship.

## Import / pytest gotchas

- The repo folder name contains a hyphen and is **not a valid python
  package name**. Consequences:
  - mypy errors with "not a valid Python package name" unless run with
    `--explicit-package-bases`.
  - Unit tests import `core` through a stub parent package aliased
    `comfyui_lukutar_nodes` (`tests/conftest.py`), which deliberately
    does **not** execute the root `__init__.py` (it would pull in torch).
- pytest runs with `--import-mode=importlib` and root `conftest.py` sets
  `collect_ignore = ["__init__.py"]` — don't remove either; pytest 9
  would otherwise try to import the root `__init__` and fail.
- Root `__init__.py` is dual-mode: imported as a package it exports the
  real node mappings; imported as a bare module (no package context) it
  exposes empty mappings instead of failing. ComfyUI never takes the
  bare-module branch.

## Conventions

- Python >= 3.10. Every module starts with the header docstring
  (`File / Brief / Author / Date / Version`) followed by
  `from __future__ import annotations`.
- Google-style docstrings (`Args:` / `Returns:`); `#:` doc comments on
  module constants; combo choices exposed as module-level string tuples
  (e.g. `MATCH_METHODS` in `core/color_matcher.py`) and re-exported via
  the package `__init__` `__all__`.
- Use module loggers (`logging.getLogger(__name__)`), not print.
- Version is tracked in **both** `__init__.py.__version__` and
  `pyproject.toml` — bump them together (release flow: see Releases).
