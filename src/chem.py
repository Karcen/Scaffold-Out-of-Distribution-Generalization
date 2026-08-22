"""Cheminformatics primitives used by every phase of the project.

The functions in this module intentionally avoid learned molecular encoders.  The
same fixed molecular views are used to define splits, risk environments, and the
ordinary MLP input, keeping the contribution at the objective/evaluation level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


METALS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
}


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string without raising on an invalid record."""

    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def canonical_smiles(smiles: str, isomeric: bool = True) -> str:
    """Return canonical SMILES, or a stable invalid sentinel."""

    mol = mol_from_smiles(smiles)
    if mol is None:
        return "__INVALID__"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def connectivity_key(smiles: str) -> str:
    """Connectivity-only key used to keep stereochemical analogues in one fold."""

    mol = mol_from_smiles(smiles)
    if mol is None:
        return "__INVALID__"
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def murcko_scaffold(smiles: str) -> str:
    """Robust stereochemistry-free Bemis-Murcko scaffold.

    RDKit's ``MurckoScaffoldSmiles`` convenience function fails on a small set
    of valid CMNPD structures with stale double-bond stereo flags.  Removing
    stereochemistry from the extracted scaffold before canonicalization avoids
    that precondition violation.
    """

    mol = mol_from_smiles(smiles)
    if mol is None:
        return "__INVALID__"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "__ACYCLIC__"
    Chem.RemoveStereochemistry(scaffold)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def bit_vector_to_numpy(fp: DataStructs.ExplicitBitVect) -> np.ndarray:
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fingerprint(smiles: str, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return bit_vector_to_numpy(generator.GetFingerprint(mol))


def maccs_fingerprint(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    return bit_vector_to_numpy(MACCSkeys.GenMACCSKeys(mol))


def atom_pair_fingerprint(smiles: str, n_bits: int = 1024) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    return bit_vector_to_numpy(generator.GetFingerprint(mol))


def descriptor_vector(smiles: str) -> np.ndarray:
    """A fixed, label-free physicochemical view for split construction.

    Values are recomputed from structure rather than copied from CMNPD so the
    exact same view is available for every external experimental benchmark.
    """

    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(12, np.nan, dtype=np.float32)
    atoms = list(mol.GetAtoms())
    formal_charge = sum(atom.GetFormalCharge() for atom in atoms)
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
            formal_charge,
            float(any(atom.GetAtomicNum() in METALS for atom in atoms)),
        ],
        dtype=np.float32,
    )


def featurize_many(
    smiles: Iterable[str],
    view: str,
    n_bits: int = 1024,
) -> np.ndarray:
    """Featurize SMILES using one pre-registered structural view."""

    functions = {
        "ecfp": lambda s: morgan_fingerprint(s, n_bits=n_bits),
        "maccs": maccs_fingerprint,
        "atompair": lambda s: atom_pair_fingerprint(s, n_bits=n_bits),
        "physchem": descriptor_vector,
    }
    if view not in functions:
        raise ValueError(f"Unknown molecular view: {view}")
    return np.stack([functions[view](s) for s in smiles])


@dataclass(frozen=True)
class MoleculeAudit:
    valid: bool
    canonical_isomeric: str
    connectivity: str
    scaffold: str
    fragments: int
    formal_charge: int
    contains_metal: bool


def audit_molecule(smiles: str) -> MoleculeAudit:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return MoleculeAudit(False, "__INVALID__", "__INVALID__", "__INVALID__", 0, 0, False)
    atoms = list(mol.GetAtoms())
    return MoleculeAudit(
        valid=True,
        canonical_isomeric=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        connectivity=connectivity_key(smiles),
        scaffold=murcko_scaffold(smiles),
        fragments=len(Chem.GetMolFrags(mol)),
        formal_charge=sum(atom.GetFormalCharge() for atom in atoms),
        contains_metal=any(atom.GetAtomicNum() in METALS for atom in atoms),
    )

