'''
File:   test_seed_history.py
Brief:  Unit tests for the Seed History node contract (torch-free).
Author: Mistress-Lukutar
Date:   2026-10-17
Version: v0.1.0
'''

from __future__ import annotations

from comfyui_lukutar_nodes.nodes.seed_history import SeedHistoryNode


class TestSeedHistoryContract:
    def test_no_control_after_generate_input(self) -> None:
        '''The frontend owns control_after_generate; declaring it breaks
        validation with "Required input is missing" because the frontend
        auto-widget is created with serialize:false and never sent.'''
        required = SeedHistoryNode.INPUT_TYPES()["required"]
        assert set(required) == {"seed", "history_size"}

    def test_seed_bounds(self) -> None:
        config = SeedHistoryNode.INPUT_TYPES()["required"]["seed"][1]
        assert config["default"] == 0
        assert config["min"] == 0
        assert config["max"] == 2**64 - 1

    def test_history_size_bounds(self) -> None:
        config = SeedHistoryNode.INPUT_TYPES()["required"]["history_size"][
            1
        ]
        assert config["default"] == 5
        assert config["min"] == 1

    def test_output_is_plain_int(self) -> None:
        '''SEED would be a custom output type no sampler input accepts.'''
        assert SeedHistoryNode.RETURN_TYPES == ("INT",)
        assert SeedHistoryNode.RETURN_NAMES == ("seed",)

    def test_category(self) -> None:
        assert SeedHistoryNode.CATEGORY == "Lukutar/Seeds"


class TestSeedHistoryGenerate:
    def test_generate_passes_seed_through(self) -> None:
        node = SeedHistoryNode()
        assert node.generate(seed=42, history_size=5) == (42,)
        assert node.generate(seed=2**64 - 1, history_size=1) == (
            2**64 - 1,
        )

    def test_generate_always_in_valid_range(self) -> None:
        node = SeedHistoryNode()
        for seed in (0, 1, 12345):
            (value,) = node.generate(seed=seed, history_size=3)
            assert 0 <= value <= 2**64 - 1
