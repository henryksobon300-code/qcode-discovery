"""RIS results are witnessed upper bounds, never exact-distance certificates."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.distance_ris import estimate_distance_ris, search_distance_ris
from evaluation.distance_witness import check_logical_witness, stabilizer_fingerprint


def _paulis(*words):
    n = len(words[0])
    return np.array([[int(p in "XY") for p in word]
                     + [int(p in "ZY") for p in word] for word in words],
                    dtype=np.uint8).reshape(len(words), 2 * n)


def _exhaustive_distance(stab):
    """Tiny-code oracle: enumerate both row space and every Pauli directly."""
    n = stab.shape[1] // 2
    row_space = set()
    for coefficients in itertools.product((0, 1), repeat=stab.shape[0]):
        row = np.zeros(2 * n, dtype=np.uint8)
        for use, check in zip(coefficients, stab):
            if use:
                row ^= check
        row_space.add(tuple(row))
    minimum = None
    for bits in itertools.product((0, 1), repeat=2 * n):
        if bits in row_space:
            continue
        if any(sum(int(a) * b for a, b in zip(check[:n], bits[n:]))
               + sum(int(a) * b for a, b in zip(check[n:], bits[:n])) & 1
               for check in stab):
            continue
        weight = sum(bool(bits[i] or bits[n + i]) for i in range(n))
        minimum = weight if minimum is None else min(minimum, weight)
    return minimum


@pytest.mark.parametrize("stab, distance", [
    (_paulis("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"), 3),  # [[5,1,3]]
    (_paulis("XXXX", "ZZZZ"), 2),  # [[4,2,2]]
    (_paulis("ZZI", "IZZ"), 1),  # repetition code's Z logical
    (_paulis("YY"), 1),  # a Y witness has binary weight 2, Pauli weight 1
    (np.zeros((0, 6), dtype=np.uint8), 1),  # no checks
])
@pytest.mark.parametrize("frames", [False, True])
def test_small_codes_against_exhaustive_oracle(stab, distance, frames):
    assert _exhaustive_distance(stab) == distance
    result = search_distance_ris(stab, num_trials=30, seed=19, random_frames=frames)
    assert result.upper_bound == distance
    assert result.status == "upper_bound"
    assert result.to_dict()["exact"] is False
    assert check_logical_witness(stab, result.witness,
                                 expected_weight=distance).valid


def test_random_small_commuting_presentations():
    rng = np.random.default_rng(2301)
    for _ in range(20):
        # Construct an isotropic set without using RIS elimination helpers.
        rows = []
        for _ in range(80):
            row = rng.integers(0, 2, 6, dtype=np.uint8)
            if all(not ((row[:3] @ other[3:] + row[3:] @ other[:3]) & 1)
                   for other in rows):
                rows.append(row)
                if len(rows) == 2:
                    break
        stab = np.array(rows)
        distance = _exhaustive_distance(stab)
        result = search_distance_ris(stab, num_trials=12, seed=7)
        assert result.upper_bound >= distance
        assert check_logical_witness(stab, result.witness,
                                     expected_weight=result.upper_bound).valid


def test_dependent_checks_and_zero_rows_do_not_change_dimension():
    stab = _paulis("XXXX", "ZZZZ")
    dependent = np.vstack((stab, stab[0] ^ stab[1], stab[0], np.zeros(8)))
    result = search_distance_ris(dependent, num_trials=30, seed=2)
    assert result.k == 2
    assert result.upper_bound == 2
    assert check_logical_witness(dependent, result.witness).valid


def test_no_logical_qubits_is_not_distance_zero_or_exact():
    result = search_distance_ris(_paulis("ZI", "IZ"), num_trials=3)
    assert result.k == 0
    assert result.upper_bound is None and result.witness is None
    assert result.status == "unknown"
    assert result.stop_reason == "no_logical_qubits"
    assert result.to_dict()["exact"] is False


@pytest.mark.parametrize("budget", [{"num_trials": 0}, {"time_limit": 0}])
def test_exhausted_budget_does_not_raise_a_lower_bound(budget):
    result = search_distance_ris(_paulis("YY"), **budget)
    assert result.status == "unknown"
    assert result.upper_bound is None and result.witness is None
    assert result.trials_completed == result.candidates_tested == 0
    assert result.to_dict()["exact"] is False


def test_seed_reproducibility_and_more_trials_cannot_worsen_bound():
    stab = _paulis("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
    first = search_distance_ris(stab, seed=91, num_trials=5)
    repeat = search_distance_ris(stab, seed=91, num_trials=5)
    assert first.witness == repeat.witness
    assert first.upper_bound == repeat.upper_bound
    assert first.candidates_tested == repeat.candidates_tested
    assert search_distance_ris(stab, seed=91, num_trials=20).upper_bound <= first.upper_bound


def test_target_is_only_an_early_stop_not_an_exactness_claim():
    result = search_distance_ris(_paulis("XXXX", "ZZZZ"), target_weight=2)
    assert result.stop_reason == "target_weight"
    assert result.trials_completed == 1
    assert result.upper_bound == 2
    assert result.to_dict()["exact"] is False


@pytest.mark.parametrize("kwargs", [
    {"num_trials": -1}, {"num_trials": 1.5}, {"num_trials": True},
    {"pair_rows": -1}, {"seed": 1.5}, {"time_limit": -1},
    {"time_limit": float("nan")}, {"time_limit": float("inf")},
    {"target_weight": 0}, {"target_weight": 1.5},
])
def test_invalid_search_parameters(kwargs):
    with pytest.raises(ValueError):
        search_distance_ris(_paulis("YY"), **kwargs)


@pytest.mark.parametrize("matrix", [
    np.zeros((2, 3)), np.zeros((2, 0)), np.zeros(4),
    np.array([[2, 0]]), _paulis("X", "Z"),
])
def test_invalid_stabilizers_are_rejected(matrix):
    with pytest.raises(ValueError):
        search_distance_ris(matrix)


def test_independent_checker_rejects_corrupt_and_trivial_witnesses():
    stab = _paulis("ZZI", "IZZ")
    assert check_logical_witness(stab, [0] * 6).reason == "stabilizer_or_zero"
    assert check_logical_witness(stab, stab[0]).reason == "stabilizer_or_zero"
    assert check_logical_witness(stab, _paulis("XII")[0]).reason == "nonzero_syndrome"
    assert check_logical_witness(stab, [1, 0]).reason == "witness_shape"
    assert check_logical_witness(stab, [0, 0, 0, 2, 0, 0]).reason == "witness_nonbinary"
    witness = _paulis("ZII")[0]
    assert check_logical_witness(stab, witness).valid
    assert check_logical_witness(stab, witness, expected_weight=2).reason == "weight_mismatch"
    assert check_logical_witness(_paulis("YY"), _paulis("YI")[0], expected_weight=1).valid


def test_fingerprint_binds_shape_and_matrix_without_mutating_input():
    stab = _paulis("XXXX", "ZZZZ")
    original = stab.copy()
    digest = stabilizer_fingerprint(stab)
    assert digest == stabilizer_fingerprint(stab.astype(bool))
    assert digest != stabilizer_fingerprint(stab[::-1])
    search_distance_ris(stab, num_trials=2)
    assert np.array_equal(stab, original)


def test_qldpc_adapter_rejects_nonbinary_field():
    from qldpc.codes import QuditCode
    code = QuditCode(np.array([[1, 0, 0, 0]]), field=3)
    with pytest.raises(ValueError, match="binary"):
        estimate_distance_ris(code)


@pytest.mark.slow
@pytest.mark.parametrize("code_id, target, trials", [
    ("68d93d677268a3fd", 16, 120),
    ("481101301eadfc29", 20, 1),
])
def test_known_campaign7_overestimates(code_id, target, trials):
    from evaluation.pbb_code import build_pbb_code
    path = Path(__file__).resolve().parents[1] / "results/campaign7_publication_merged.jsonl"
    record = next(json.loads(line) for line in path.read_text().splitlines()
                  if json.loads(line)["bliss_hash"] == code_id)
    code = build_pbb_code(record["ell"], record["m"],
                          *[record[name + "_terms"] for name in "ABCD"])
    result = estimate_distance_ris(code, num_trials=trials, seed=0,
                                   target_weight=target, random_frames=False)
    assert result.upper_bound <= target < record["d"]
    assert result.n == 360 and result.k == 10
    assert result.to_dict()["exact"] is False
    assert check_logical_witness(np.asarray(code.matrix), result.witness,
                                 expected_weight=result.upper_bound).valid
