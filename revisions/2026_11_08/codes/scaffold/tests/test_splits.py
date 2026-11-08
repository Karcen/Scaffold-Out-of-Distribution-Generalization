from __future__ import annotations

from src.splits import assert_split_integrity, frontier_split, scaffold_split


SMILES = [
    "c1ccccc1O",
    "c1ccccc1N",
    "C1CCCCC1O",
    "C1CCCCC1N",
    "CCO",
    "CCN",
    "c1ccncc1",
    "c1ccoc1",
    "CC(=O)O",
    "CCS",
    "C1CCOC1",
    "C1CCNC1",
]


def test_scaffold_split_has_no_scaffold_overlap() -> None:
    result = scaffold_split(SMILES, seed=7, fractions=(0.5, 0.25, 0.25))
    assert_split_integrity(result)


def test_frontier_split_is_label_free_and_leakage_controlled() -> None:
    result = frontier_split(SMILES, seed=7, fractions=(0.5, 0.25, 0.25), neighbors=2)
    assert result.metadata["label_free"] is True
    assert_split_integrity(result)

