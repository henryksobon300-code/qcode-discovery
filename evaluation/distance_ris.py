"""Randomized information-set search for stabilizer distance upper bounds.

RIS samples systematic bases of the normalizer using shuffled pivot columns,
then scores basis vectors and XORs of light pairs by *Pauli* weight. Random
single-qubit symplectic frames preserve this objective while exposing other
information sets. This is a bounded heuristic, orthogonal to BP-OSD and MILP:
no witness means UNKNOWN, and any witness proves only an upper bound.

Binary linear algebra uses Python integers, so the search requires no native
extension, decoder, solver, or qldpc logical-operator routine. Every returned
witness is checked by the separate dense implementation in distance_witness.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import random
import time
from typing import TYPE_CHECKING

import numpy as np

from evaluation.distance_witness import (
    binary_stabilizer,
    check_logical_witness,
    stabilizer_fingerprint,
)

if TYPE_CHECKING:
    from qldpc.codes import QuditCode


@dataclass(frozen=True)
class RISResult:
    upper_bound: int | None
    witness: tuple[int, ...] | None
    seed: int
    trials_completed: int
    candidates_tested: int
    elapsed_seconds: float
    stop_reason: str
    stabilizer_sha256: str
    n: int
    k: int

    @property
    def status(self) -> str:
        return "upper_bound" if self.witness is not None else "unknown"

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(status=self.status, method="ris", exact=False)
        if self.witness is not None:
            data["witness"] = list(self.witness)
        return data


def _pack(row: np.ndarray) -> int:
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def _rref(rows: list[int], columns: list[int]) -> tuple[list[int], list[int]]:
    """Keep original coordinate labels; only pivot visitation order changes."""
    work = rows.copy()
    rank = 0
    pivots = []
    for col in columns:
        bit = 1 << col
        pivot = next((i for i in range(rank, len(work)) if work[i] & bit), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        value = work[rank]
        work = [row ^ value if i != rank and row & bit else row
                for i, row in enumerate(work)]
        pivots.append(col)
        rank += 1
        if rank == len(work):
            break
    return work[:rank], pivots


def _kernel(rows: list[int], width: int) -> list[int]:
    reduced, pivots = _rref(rows, list(range(width)))
    occupied = set(pivots)
    basis = []
    for col in range(width):
        if col in occupied:
            continue
        bit = 1 << col
        value = bit
        for row, pivot in zip(reduced, pivots):
            if row & bit:
                value |= 1 << pivot
        basis.append(value)
    return basis


def _insert(value: int, basis: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = value
            return True
        value ^= basis[pivot]
    return False


def _swap(value: int, n: int, mask: int) -> int:
    return ((value & mask) << n) | (value >> n)


# The six invertible 2x2 matrices over GF(2). They act on [x_i, z_i].
_FRAMES = ((1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 0, 1),
           (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 0))


def _frame_masks(rng: random.Random, n: int) -> tuple[int, int, int, int]:
    masks = [0, 0, 0, 0]
    for qubit in range(n):
        for i, enabled in enumerate(rng.choice(_FRAMES)):
            if enabled:
                masks[i] |= 1 << qubit
    return tuple(masks)


def _transform(value: int, n: int, mask: int, frame: tuple[int, ...]) -> int:
    a, b, c, d = frame
    x, z = value & mask, value >> n
    return ((x & a) ^ (z & b)) | (((x & c) ^ (z & d)) << n)


def search_distance_ris(
    stabilizer: np.ndarray,
    *,
    num_trials: int = 100,
    seed: int = 0,
    time_limit: float | None = None,
    target_weight: int | None = None,
    pair_rows: int = 16,
    random_frames: bool = False,
) -> RISResult:
    """Return a reproducible upper-bound witness for a binary stabilizer matrix.

    ``target_weight`` is an early-stop threshold, never a claimed lower bound.
    ``num_trials=0`` performs no search and returns UNKNOWN. A time limit is
    checked between trials (setup and a single elimination are not preempted);
    use trial budgets for deterministic regression runs. ``k=0`` has no
    nontrivial logicals and returns UNKNOWN with ``no_logical_qubits``.
    """
    for name, value in (("num_trials", num_trials), ("pair_rows", pair_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative")
    if target_weight is not None and (
        isinstance(target_weight, bool) or not isinstance(target_weight, int)
        or target_weight < 1
    ):
        raise ValueError("target_weight must be a positive integer")

    started = time.monotonic()
    stab = binary_stabilizer(stabilizer)
    n = stab.shape[1] // 2
    mask = (1 << n) - 1
    rows = [_pack(row) for row in stab]
    echelon: dict[int, int] = {}
    for row in rows:
        _insert(row, echelon)
    k = n - len(echelon)
    digest = stabilizer_fingerprint(stab)
    best: int | None = None
    best_weight: int | None = None
    completed = tested = 0
    reason = "trial_budget"

    def result() -> RISResult:
        witness = (tuple((best >> i) & 1 for i in range(2 * n))
                   if best is not None else None)
        if witness is not None:
            check = check_logical_witness(stab, witness, expected_weight=best_weight)
            if not check.valid:
                raise RuntimeError(f"RIS produced an invalid witness: {check.reason}")
        return RISResult(best_weight, witness, seed, completed, tested,
                         time.monotonic() - started, reason, digest, n, k)

    if k == 0:
        reason = "no_logical_qubits"
        return result()
    if num_trials == 0:
        return result()
    if time_limit == 0:
        reason = "time_limit"
        return result()

    normalizer = _kernel([_swap(row, n, mask) for row in echelon.values()], 2 * n)
    quotient = echelon.copy()
    logical_checks = []
    for row in normalizer:
        if _insert(row, quotient):
            logical_checks.append(_swap(row, n, mask))
    if len(logical_checks) != 2 * k:
        raise RuntimeError("normalizer/stabilizer quotient has unexpected dimension")

    def consider(value: int) -> None:
        nonlocal best, best_weight, tested
        tested += 1
        weight = ((value & mask) | (value >> n)).bit_count()
        if best_weight is not None and weight >= best_weight:
            return
        # The radical of the normalizer is precisely the stabilizer. Thus a
        # nonzero pairing with this quotient basis detects nontriviality.
        if any((value & logical).bit_count() & 1 for logical in logical_checks):
            best, best_weight = value, weight

    rng = random.Random(seed)
    for trial in range(num_trials):
        if time_limit is not None and time.monotonic() - started >= time_limit:
            reason = "time_limit"
            break
        # The first untransformed pass is a useful deterministic baseline.
        frame = _frame_masks(rng, n) if random_frames and trial else (mask, 0, 0, mask)
        inverse = (frame[3], frame[1], frame[2], frame[0])
        working = [_transform(row, n, mask, frame) for row in normalizer]
        columns = list(range(2 * n))
        if trial:
            rng.shuffle(columns)
        reduced, _ = _rref(working, columns)
        candidates = [_transform(row, n, mask, inverse) for row in reduced]
        for row in candidates:
            consider(row)
        light = sorted(candidates,
                       key=lambda v: ((v & mask) | (v >> n)).bit_count())[:pair_rows]
        for i, left in enumerate(light):
            for right in light[i + 1:]:
                consider(left ^ right)
        completed += 1
        if best_weight is not None and target_weight is not None and best_weight <= target_weight:
            reason = "target_weight"
            break
    return result()


def estimate_distance_ris(code: QuditCode, **kwargs) -> RISResult:
    """Matrix-search adapter for CSS or non-CSS binary qldpc codes."""
    if code.field.order != 2:
        raise ValueError("RIS supports only binary (qubit) codes")
    return search_distance_ris(np.asarray(code.matrix, dtype=np.uint8), **kwargs)
