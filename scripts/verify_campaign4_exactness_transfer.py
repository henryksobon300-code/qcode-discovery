#!/usr/bin/env python3
"""Verify Campaign 4 exact-distance transfer by explicit qubit permutations.

The five corrected records inherit d=12 from one exact representative.
Records are identified by (ell, m, A_terms, B_terms), not by JSONL position.
The frozen pre-correction source blob is recorded only as provenance; the
current JSONL necessarily has a different blob after this correction.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.bb_code import build_bb_code
from evaluation.tanner_equivalence import _extract_check_matrices, canonical_digest

SOURCE = ROOT / "results" / "campaign4_reverified.jsonl"
FROZEN_SOURCE_BLOB_SHA1 = "3fcddce65c1711041cc82eddebee0242a201b61e"

REFERENCE = {"ell":12,"m":12,"A_terms":[[0,0],[2,3],[1,0]],"B_terms":[[0,0],[1,2],[2,1]]}

TARGETS = [
    {
        "id": {
            "ell": 12,
            "m": 12,
            "A_terms": [
                [
                    0,
                    0
                ],
                [
                    1,
                    2
                ],
                [
                    2,
                    1
                ]
            ],
            "B_terms": [
                [
                    0,
                    0
                ],
                [
                    10,
                    1
                ],
                [
                    11,
                    2
                ]
            ]
        },
        "map": {
            "L": {
                "target_sector": "L",
                "matrix": [
                    [
                        1,
                        11
                    ],
                    [
                        11,
                        0
                    ]
                ],
                "translation": [
                    0,
                    0
                ]
            },
            "R": {
                "target_sector": "R",
                "matrix": [
                    [
                        1,
                        11
                    ],
                    [
                        11,
                        0
                    ]
                ],
                "translation": [
                    10,
                    0
                ]
            }
        }
    },
    {
        "id": {
            "ell": 12,
            "m": 12,
            "A_terms": [
                [
                    0,
                    0
                ],
                [
                    3,
                    1
                ],
                [
                    3,
                    2
                ]
            ],
            "B_terms": [
                [
                    0,
                    0
                ],
                [
                    1,
                    2
                ],
                [
                    2,
                    1
                ]
            ]
        },
        "map": {
            "L": {
                "target_sector": "L",
                "matrix": [
                    [
                        0,
                        11
                    ],
                    [
                        1,
                        11
                    ]
                ],
                "translation": [
                    0,
                    0
                ]
            },
            "R": {
                "target_sector": "R",
                "matrix": [
                    [
                        0,
                        11
                    ],
                    [
                        1,
                        11
                    ]
                ],
                "translation": [
                    11,
                    0
                ]
            }
        }
    },
    {
        "id": {
            "ell": 12,
            "m": 12,
            "A_terms": [
                [
                    0,
                    0
                ],
                [
                    3,
                    2
                ],
                [
                    0,
                    1
                ]
            ],
            "B_terms": [
                [
                    0,
                    0
                ],
                [
                    1,
                    2
                ],
                [
                    2,
                    1
                ]
            ]
        },
        "map": {
            "L": {
                "target_sector": "L",
                "matrix": [
                    [
                        0,
                        1
                    ],
                    [
                        1,
                        0
                    ]
                ],
                "translation": [
                    0,
                    0
                ]
            },
            "R": {
                "target_sector": "R",
                "matrix": [
                    [
                        0,
                        1
                    ],
                    [
                        1,
                        0
                    ]
                ],
                "translation": [
                    0,
                    0
                ]
            }
        }
    },
    {
        "id": {
            "ell": 12,
            "m": 12,
            "A_terms": [
                [
                    0,
                    0
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    3
                ]
            ],
            "B_terms": [
                [
                    0,
                    0
                ],
                [
                    1,
                    2
                ],
                [
                    2,
                    1
                ]
            ]
        },
        "map": {
            "L": {
                "target_sector": "L",
                "matrix": [
                    [
                        1,
                        11
                    ],
                    [
                        0,
                        11
                    ]
                ],
                "translation": [
                    0,
                    0
                ]
            },
            "R": {
                "target_sector": "R",
                "matrix": [
                    [
                        1,
                        11
                    ],
                    [
                        0,
                        11
                    ]
                ],
                "translation": [
                    0,
                    11
                ]
            }
        }
    },
    {
        "id": {
            "ell": 12,
            "m": 12,
            "A_terms": [
                [
                    0,
                    0
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    3
                ]
            ],
            "B_terms": [
                [
                    0,
                    0
                ],
                [
                    3,
                    1
                ],
                [
                    9,
                    2
                ]
            ]
        },
        "map": {
            "L": {
                "target_sector": "L",
                "matrix": [
                    [
                        11,
                        8
                    ],
                    [
                        0,
                        11
                    ]
                ],
                "translation": [
                    0,
                    0
                ]
            },
            "R": {
                "target_sector": "R",
                "matrix": [
                    [
                        11,
                        8
                    ],
                    [
                        0,
                        11
                    ]
                ],
                "translation": [
                    7,
                    11
                ]
            }
        }
    }
]

def identity(row):
    return {
        "ell": int(row["ell"]),
        "m": int(row["m"]),
        "A_terms": row["A_terms"],
        "B_terms": row["B_terms"],
    }

def identity_key(value):
    return (
        int(value["ell"]),
        int(value["m"]),
        tuple(tuple(term) for term in value["A_terms"]),
        tuple(tuple(term) for term in value["B_terms"]),
    )

def gf2_rank(matrix):
    a = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(a[rank:, col])
        if len(pivots) == 0:
            continue
        pivot = rank + int(pivots[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank

def same_rowspace(left, right):
    left = np.asarray(left, dtype=np.uint8) & 1
    right = np.asarray(right, dtype=np.uint8) & 1
    rl = gf2_rank(left)
    rr = gf2_rank(right)
    return rl == rr == gf2_rank(np.vstack((left, right)))

def qubit_permutation(spec, ell, m):
    cells = ell * m
    perm = np.empty(2 * cells, dtype=int)
    sector_base = {"L": 0, "R": cells}
    for source_sector, source_base in sector_base.items():
        rule = spec[source_sector]
        target_base = sector_base[rule["target_sector"]]
        matrix = rule["matrix"]
        translation = rule["translation"]
        for a in range(ell):
            for b in range(m):
                aa = (
                    matrix[0][0] * a
                    + matrix[0][1] * b
                    + translation[0]
                ) % ell
                bb = (
                    matrix[1][0] * a
                    + matrix[1][1] * b
                    + translation[1]
                ) % m
                src = source_base + a * m + b
                perm[src] = target_base + aa * m + bb
    if sorted(perm.tolist()) != list(range(2 * cells)):
        raise AssertionError("explicit qubit map is not bijective")
    return perm

def main():
    rows = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 39

    by_id = {identity_key(identity(row)): row for row in rows}
    assert len(by_id) == len(rows), "duplicate polynomial identities in source"

    reference = by_id[identity_key(REFERENCE)]
    assert reference["n"] == 288
    assert reference["k"] == 12
    assert reference["d"] == 12
    assert reference["d_is_exact"] is True

    ref_code = build_bb_code(
        reference["ell"], reference["m"],
        reference["A_terms"], reference["B_terms"],
    )
    hx_ref, hz_ref = _extract_check_matrices(ref_code)
    ref_digest = canonical_digest(ref_code)

    target_keys = {identity_key(item["id"]) for item in TARGETS}
    expected_class_keys = target_keys | {identity_key(REFERENCE)}

    all_codes = {}
    classes = defaultdict(set)
    for row in rows:
        key = identity_key(identity(row))
        code = build_bb_code(row["ell"], row["m"], row["A_terms"], row["B_terms"])
        all_codes[key] = code
        classes[canonical_digest(code)].add(key)

    # Vincent Russo's all-39 question: the reference class is exactly six.
    assert classes[ref_digest] == expected_class_keys

    checked = []
    for item in TARGETS:
        key = identity_key(item["id"])
        row = by_id[key]

        assert row["n"] == 288 and row["k"] == 12 and row["d"] == 12
        assert row["d_is_exact"] is True

        transfer = row.get("distance_transfer")
        assert transfer == {
            "valid": True,
            "method": "colored_bliss_permutation_equivalence",
            "reference_d_exact": 12,
            "source_path": "results/campaign4_reverified.jsonl",
            "source_git_blob_sha1": FROZEN_SOURCE_BLOB_SHA1,
            "reference": REFERENCE,
            "qubit_map_from_reference": item["map"],
        }

        code = all_codes[key]
        assert canonical_digest(code) == ref_digest
        hx_target, hz_target = _extract_check_matrices(code)

        perm = qubit_permutation(item["map"], row["ell"], row["m"])
        assert same_rowspace(hx_ref, hx_target[:, perm])
        assert same_rowspace(hz_ref, hz_target[:, perm])

        # Keep historical MILP metadata historical: exactness was transferred,
        # not newly proved by those previously incomplete MILP runs.
        assert row["stage"] == "milp_incumbent"
        assert row["milp_details"]["exact"] is False

        checked.append(identity(row))

    # Other duplicate classes found by the all-39 BLISS audit contain no exact
    # representative, so this correction intentionally does not promote them.
    other_multi = [
        members for digest, members in classes.items()
        if digest != ref_digest and len(members) > 1
    ]
    assert len(other_multi) == 4

    print(json.dumps({
        "status": "PASS",
        "rows_checked": len(rows),
        "equivalence_classes": len(classes),
        "reference_class_size": len(classes[ref_digest]),
        "transfers_verified": len(checked),
        "other_multi_member_classes_without_transfer": len(other_multi),
        "frozen_source_blob_sha1": FROZEN_SOURCE_BLOB_SHA1,
    }, indent=2))

if __name__ == "__main__":
    main()
