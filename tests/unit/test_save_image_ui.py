'''
File:   test_save_image_ui.py
Brief:  Unit tests for the ``build_ui_entries`` helper in nodes/save_image.py.
Author: Mistress-Lukutar
Date:   2026-11-03
Version: v0.1.0

Tests that saved files under the output and temp directories produce
correct ComfyUI preview entries with proper subfolder and type fields.
'''

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from comfyui_lukutar_nodes.nodes import save_image as _si_mod

# ======================================================================
# Helpers                                                                #
# ======================================================================


def _make_fake_fp(output_sub: Path, temp_sub: Path) -> MagicMock:
    '''Return a ``MagicMock`` mimicking ``folder_paths``.

    Args:
        output_sub: Directory that acts as the ComfyUI output root.
        temp_sub: Directory that acts as the ComfyUI temp root.

    Returns:
        A ``MagicMock`` with ``get_output_directory`` and
        ``get_temp_directory`` returning string paths rooted at
        *output_sub* / *temp_sub*.
    '''
    fake_fp = MagicMock()
    fake_fp.get_output_directory.return_value = str(output_sub)
    fake_fp.get_temp_directory.return_value = str(temp_sub)
    return fake_fp


# ======================================================================
# File under output dir                                                  #
# ======================================================================


def test_entry_bare_output(tmp_path: Path) -> None:
    '''A file directly in the output directory emits subfolder "".

    Regression: the old code returned an empty list for files not under
    ``output`` — this test pins the baseline that files *under* output
    still work.
    '''
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    file_path = output_dir / "img.png"
    file_path.write_bytes(b"data")

    fake_fp = _make_fake_fp(output_dir, tmp_path / "temp")

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([file_path])

    assert len(result) == 1
    entry = result[0]
    assert entry["filename"] == "img.png"
    assert entry["subfolder"] == ""
    assert entry["type"] == "output"


def test_entry_nested_output(tmp_path: Path) -> None:
    '''A file nested under output emits the relative posix subfolder.

    ``<output>/sub/x.png`` → subfolder ``"sub"``.
    '''
    output_dir = tmp_path / "output"
    nested = output_dir / "sub"
    nested.mkdir(parents=True)
    file_path = nested / "x.png"
    file_path.write_bytes(b"data")

    fake_fp = _make_fake_fp(output_dir, tmp_path / "temp")

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([file_path])

    assert len(result) == 1
    entry = result[0]
    assert entry["filename"] == "x.png"
    assert entry["subfolder"] == "sub"
    assert entry["type"] == "output"


# ======================================================================
# File under temp dir (regression case for the reported bug)             #
# ======================================================================


def test_entry_bare_temp(tmp_path: Path) -> None:
    '''A file directly in the temp directory emits type "temp", subfolder "".

    Regression: the old code only emitted entries for files under the
    output directory, so files saved to ``temp`` produced an empty list
    and no preview appeared.
    '''
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    file_path = temp_dir / "preview.png"
    file_path.write_bytes(b"data")

    fake_fp = _make_fake_fp(tmp_path / "output", temp_dir)

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([file_path])

    assert len(result) == 1
    entry = result[0]
    assert entry["filename"] == "preview.png"
    assert entry["subfolder"] == ""
    assert entry["type"] == "temp"


def test_entry_nested_temp(tmp_path: Path) -> None:
    '''A file nested under temp emits the correct subfolder and type "temp".

    ``<temp>/cache/img.png`` → subfolder ``"cache"``.
    '''
    temp_dir = tmp_path / "temp"
    nested = temp_dir / "cache"
    nested.mkdir(parents=True)
    file_path = nested / "img.png"
    file_path.write_bytes(b"data")

    fake_fp = _make_fake_fp(tmp_path / "output", temp_dir)

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([file_path])

    assert len(result) == 1
    entry = result[0]
    assert entry["filename"] == "img.png"
    assert entry["subfolder"] == "cache"
    assert entry["type"] == "temp"


# ======================================================================
# File outside both directories                                          #
# ======================================================================


def test_entry_outside_both_skipped(tmp_path: Path) -> None:
    '''A file under an arbitrary path (not output or temp) is skipped.

    ComfyUI's ``/view`` endpoint cannot serve it, so the entry list must
    remain empty — same contract as the original code.
    '''
    stray_dir = tmp_path / "stray"
    stray_dir.mkdir()
    file_path = stray_dir / "orphan.png"
    file_path.write_bytes(b"data")

    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()

    fake_fp = _make_fake_fp(output_dir, temp_dir)

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([file_path])

    assert result == []


# ======================================================================
# HAS_COMFY False                                                        #
# ======================================================================


def test_no_comfy_returns_empty() -> None:
    '''When HAS_COMFY is False, build_ui_entries returns [].'''
    with patch.object(_si_mod, "HAS_COMFY", False):
        result = _si_mod.build_ui_entries([Path("/tmp/img.png")])

    assert result == []


# ======================================================================
# Nested batch: mixed output + temp                                      #
# ======================================================================


def test_mixed_batch_preserves_order_and_types(tmp_path: Path) -> None:
    '''A mixed list with one file under output and one under temp yields
    two entries with correct types, preserving input order.

    Regression: the old code only emitted "output" entries, so a batch
    spanning both directories would lose the temp files from the preview.
    '''
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    output_dir.mkdir()
    temp_dir.mkdir()

    out_file = output_dir / "frame0.png"
    out_file.write_bytes(b"data0")

    temp_file = temp_dir / "frame1.png"
    temp_file.write_bytes(b"data1")

    fake_fp = _make_fake_fp(output_dir, temp_dir)

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([out_file, temp_file])

    assert len(result) == 2
    # First entry: output file.
    assert result[0]["filename"] == "frame0.png"
    assert result[0]["subfolder"] == ""
    assert result[0]["type"] == "output"
    # Second entry: temp file.
    assert result[1]["filename"] == "frame1.png"
    assert result[1]["subfolder"] == ""
    assert result[1]["type"] == "temp"


# ======================================================================
# Degenerate: output dir == temp dir                                     #
# ======================================================================


def test_output_equals_temp_first_candidate_wins(tmp_path: Path) -> None:
    '''When output and temp resolve to the same directory, the first
    candidate ("output") wins.

    This is a degenerate config but must not raise or produce duplicate
    entries.
    '''
    shared = tmp_path / "shared"
    shared.mkdir()
    file_path = shared / "img.png"
    file_path.write_bytes(b"data")

    fake_fp = _make_fake_fp(shared, shared)

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([file_path])

    assert len(result) == 1
    assert result[0]["type"] == "output"


# ======================================================================
# Error resilience: bad path in batch                                    #
# ======================================================================


def test_bad_path_in_batch_skipped(tmp_path: Path) -> None:
    '''A malformed path inside the saved list does not kill the loop.

    The ``try/except (ValueError, OSError)`` guard per file must hold.
    '''
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    good_file = output_dir / "good.png"
    good_file.write_bytes(b"data")

    # A Path that resolves outside any real directory tree.
    bad_file = Path(
        "/nonexistent/deeply/nested/path/that/does/not/exist.png"
    )

    fake_fp = _make_fake_fp(output_dir, tmp_path / "temp")

    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([good_file, bad_file])

    assert len(result) == 1
    assert result[0]["filename"] == "good.png"
    assert result[0]["type"] == "output"


# ======================================================================
# Empty input list                                                       #
# ======================================================================


def test_empty_list_returns_empty() -> None:
    '''Passing an empty list returns [].'''
    fake_fp = MagicMock()
    with (
        patch.object(_si_mod, "folder_paths", fake_fp),
        patch.object(_si_mod, "HAS_COMFY", True),
    ):
        result = _si_mod.build_ui_entries([])

    assert result == []
