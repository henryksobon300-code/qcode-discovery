"""Independently search PARTIAL non-CSS catalogue entries for smaller logicals.

Write a witness sidecar, leaving the source catalogue unchanged. Every witness
is checked against a freshly reconstructed matrix and bound to the input
file's SHA-256. ``--check`` validates saved evidence without rerunning RIS.

Example::

    uv run python scripts/refute_partial_catalogue.py --trials 600 --seed 0 \
        --output /tmp/campaign7_partial_ris_replay.jsonl
    uv run python scripts/refute_partial_catalogue.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.distance_ris import estimate_distance_ris
from evaluation.distance_schema import DistanceBoundConflict, merge_bounds
from evaluation.distance_witness import check_logical_witness, stabilizer_fingerprint
from evaluation.pbb_code import build_pbb_code

DEFAULT_INPUT = ROOT / "results/campaign7_publication_merged.jsonl"
DEFAULT_OUTPUT = ROOT / "results/campaign7_partial_ris_witnesses.jsonl"
SCHEMA = "stabilizer_ris_witness_v1"
SPEC_KEYS = ("ell", "m", "A_terms", "B_terms", "C_terms", "D_terms")


def read_catalogue(path: Path) -> tuple[list[dict], str]:
    data = path.read_bytes()
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    ids = [record["bliss_hash"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("catalogue must have unique bliss_hash identifiers")
    return records, hashlib.sha256(data).hexdigest()


def reconstruct(record: dict):
    code = build_pbb_code(record["ell"], record["m"],
                          *[record.get(name + "_terms") for name in "ABCD"])
    if code.num_qudits != record["n"] or code.dimension != record["k"]:
        raise ValueError(f"catalogue n/k mismatch for {record['bliss_hash']}")
    return code


def run_record(record: dict, input_sha256: str, **search) -> dict:
    code = reconstruct(record)
    result = estimate_distance_ris(code, **search)
    evidence = {
        "artifact_schema": SCHEMA,
        "input_sha256": input_sha256,
        "bliss_hash": record["bliss_hash"],
        "code_spec": {key: record.get(key) for key in SPEC_KEYS},
        "prior_reported_d": record["d"],
        "prior_trust_level": record.get("trust_level"),
        "search_config": search,
        **result.to_dict(),
    }
    evidence["refutes_reported_value"] = (
        result.upper_bound is not None and result.upper_bound < record["d"]
    )
    # This sidecar never imports legacy exactness flags as new proof. Only an
    # prior lower bound is retained as a conflict guard; a conflict is logged
    # for quarantine rather than resolved by overwriting existing evidence.
    lower, upper = prior_bounds(record)
    try:
        lower, upper = merge_bounds(lower, upper, 0, result.upper_bound,
                                    new_source_kind="ris_witness")
        evidence["merged_bounds"] = {"d_lower": lower, "d_upper": upper}
        evidence["bound_conflict"] = None
    except DistanceBoundConflict as exc:
        evidence["merged_bounds"] = None
        evidence["bound_conflict"] = str(exc)
    return evidence


def prior_bounds(record: dict) -> tuple[int, int | None]:
    """Preserve explicit lower bounds and guard legacy exact assertions.

    Retaining a legacy exact assertion as a conflict guard does not validate
    its certificate. It prevents triage from silently replacing an existing
    claim of proof when --ids selects an EXACT entry.
    """
    if record.get("d_is_exact") is True:
        return max(record.get("d_lower", 0), record["d"]), record.get("d_upper", record["d"])
    return record.get("d_lower", 0), record.get("d_upper", record["d"])


def check_evidence(records: list[dict], input_sha256: str, evidence: list[dict]) -> None:
    """Verify saved evidence without trusting its exact/status/weight flags."""
    by_id = {record["bliss_hash"]: record for record in records}
    seen = set()
    for saved in evidence:
        code_id = saved["bliss_hash"]
        if code_id in seen:
            raise ValueError(f"duplicate evidence for {code_id}")
        seen.add(code_id)
        if saved["artifact_schema"] != SCHEMA or saved["input_sha256"] != input_sha256:
            raise ValueError(f"evidence source/schema mismatch for {code_id}")
        record = by_id[code_id]
        if saved["code_spec"] != {key: record.get(key) for key in SPEC_KEYS}:
            raise ValueError(f"code spec mismatch for {code_id}")
        code = reconstruct(record)
        stab = np.asarray(code.matrix, dtype=np.uint8)
        if saved["stabilizer_sha256"] != stabilizer_fingerprint(stab):
            raise ValueError(f"matrix digest mismatch for {code_id}")
        if saved["exact"] is not False or saved["method"] != "ris":
            raise ValueError(f"RIS cannot certify exactness for {code_id}")
        if saved["n"] != record["n"] or saved["k"] != record["k"]:
            raise ValueError(f"evidence n/k mismatch for {code_id}")
        if saved["prior_reported_d"] != record["d"]:
            raise ValueError(f"prior distance mismatch for {code_id}")
        if saved["witness"] is None:
            if saved["upper_bound"] is not None or saved["status"] != "unknown":
                raise ValueError(f"missing upper-bound witness for {code_id}")
        else:
            upper = saved["upper_bound"]
            if type(upper) is not int or upper < 1 or saved["status"] != "upper_bound":
                raise ValueError(f"invalid upper bound for {code_id}")
            check = check_logical_witness(stab, saved["witness"], expected_weight=upper)
            if not check.valid:
                raise ValueError(f"invalid witness for {code_id}: {check.reason}")
        refuted = saved["upper_bound"] is not None and saved["upper_bound"] < record["d"]
        if saved["refutes_reported_value"] is not refuted:
            raise ValueError(f"refutation flag mismatch for {code_id}")
        try:
            prior_lower, prior_upper = prior_bounds(record)
            lower, upper = merge_bounds(prior_lower, prior_upper, 0, saved["upper_bound"],
                                        new_source_kind="ris_witness")
        except DistanceBoundConflict:
            if saved["merged_bounds"] is not None or not saved["bound_conflict"]:
                raise ValueError(f"unquarantined bound conflict for {code_id}")
        else:
            if (saved["merged_bounds"] != {"d_lower": lower, "d_upper": upper}
                    or saved["bound_conflict"] is not None):
                raise ValueError(f"merged bound mismatch for {code_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-rows", type=int, default=16)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Optional seconds per code, checked between trials")
    parser.add_argument("--random-frames", action="store_true")
    parser.add_argument("--ids", nargs="+", help="Explicit catalogue bliss_hash IDs")
    parser.add_argument("--check", action="store_true",
                        help="Check saved witnesses without invoking RIS")
    args = parser.parse_args()
    records, digest = read_catalogue(args.input)
    if args.check:
        evidence = [json.loads(line) for line in args.output.read_text().splitlines()
                    if line.strip()]
        if not evidence:
            raise ValueError("empty evidence file")
        check_evidence(records, digest, evidence)
        print(f"independently checked {len(evidence)} evidence records")
        return
    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must be a sidecar, not the source catalogue")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}; choose a new --output")
    if args.ids:
        unknown = set(args.ids) - {record["bliss_hash"] for record in records}
        if unknown:
            raise ValueError(f"IDs absent from catalogue: {sorted(unknown)}")
        selected = [record for record in records if record["bliss_hash"] in args.ids]
    else:
        selected = [record for record in records
                    if record.get("trust_level") == "PARTIAL" and record.get("lc_any") is False]
    if not selected:
        raise ValueError("no catalogue records selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for record in selected:
            saved = run_record(record, digest, num_trials=args.trials, seed=args.seed,
                               pair_rows=args.pair_rows, time_limit=args.time_limit,
                               random_frames=args.random_frames)
            stream.write(json.dumps(saved, sort_keys=True) + "\n")
            stream.flush()
            print(f"{record['bliss_hash']}: reported={record['d']}, "
                  f"RIS upper={saved['upper_bound']}, "
                  f"trials={saved['trials_completed']}, "
                  f"seconds={saved['elapsed_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
