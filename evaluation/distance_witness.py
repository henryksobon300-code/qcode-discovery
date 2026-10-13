"""Independent checks for binary stabilizer distance upper-bound witnesses.

No decoder or search implementation is used here. A nonzero Pauli in
``ker(S Lambda)`` outside ``row(S)`` proves only ``d <= weight``; it proves no lower
bound or optimality. Coordinates are ``[x_0, ..., x_(n-1), z_0, ..., z_(n-1)]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def binary_stabilizer(matrix: np.ndarray) -> np.ndarray:
    """Validate a binary, commuting stabilizer and return an owned uint8 array."""
    mat = np.asarray(matrix)
    if mat.ndim != 2 or mat.shape[1] == 0 or mat.shape[1] % 2:
        raise ValueError("stabilizer must have shape (checks, 2*n), with n > 0")
    if not np.all((mat == 0) | (mat == 1)):
        raise ValueError("stabilizer entries must be binary")
    mat = np.array(mat, dtype=np.uint8, copy=True)
    n = mat.shape[1] // 2
    # uint8 overflow preserves parity.
    sx, sz = mat[:, :n], mat[:, n:]
    if np.any((sx @ sz.T + sz @ sx.T) & 1):
        raise ValueError("stabilizer rows must commute")
    return mat


def stabilizer_fingerprint(matrix: np.ndarray) -> str:
    """Hash the binary matrix, including shape and coordinate convention.

    This binds a witness to a particular input presentation; it is not an
    equivalence hash and deliberately changes when check rows are reordered.
    """
    mat = binary_stabilizer(matrix)
    header = f"binary-stabilizer-xz-v1:{mat.shape[0]}:{mat.shape[1]}:".encode()
    return hashlib.sha256(header + mat.tobytes(order="C")).hexdigest()


def _rank_gf2(matrix: np.ndarray) -> int:
    """Dense GF(2) elimination, independent of the search's integer arithmetic."""
    work = matrix.copy()
    rank = 0
    for col in range(work.shape[1]):
        found = np.flatnonzero(work[rank:, col])
        if not found.size:
            continue
        pivot = rank + int(found[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        mask = work[:, col].astype(bool)
        mask[rank] = False
        work[mask] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


@dataclass(frozen=True)
class WitnessCheck:
    valid: bool
    weight: int | None
    reason: str


def check_logical_witness(
    stabilizer: np.ndarray,
    witness: tuple[int, ...] | list[int] | np.ndarray,
    *,
    expected_weight: int | None = None,
) -> WitnessCheck:
    """Check shape, binary entries, syndrome, quotient membership and Pauli weight.

    Invalid stabilizer input raises ``ValueError``. A malformed or trivial
    witness returns a failed check. Y on one qubit counts once, not twice.
    """
    stab = binary_stabilizer(stabilizer)
    vec = np.asarray(witness)
    if vec.ndim != 1 or vec.shape[0] != stab.shape[1]:
        return WitnessCheck(False, None, "witness_shape")
    if not np.all((vec == 0) | (vec == 1)):
        return WitnessCheck(False, None, "witness_nonbinary")
    vec = np.array(vec, dtype=np.uint8)
    n = vec.size // 2
    weight = int(np.count_nonzero(vec[:n] | vec[n:]))
    if expected_weight is not None and weight != expected_weight:
        return WitnessCheck(False, weight, "weight_mismatch")
    if np.any((stab[:, :n] @ vec[n:] + stab[:, n:] @ vec[:n]) & 1):
        return WitnessCheck(False, weight, "nonzero_syndrome")
    if _rank_gf2(np.vstack((stab, vec))) == _rank_gf2(stab):
        return WitnessCheck(False, weight, "stabilizer_or_zero")
    return WitnessCheck(True, weight, "nontrivial_logical")
