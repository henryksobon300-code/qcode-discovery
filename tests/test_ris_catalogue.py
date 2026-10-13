"""Catalogue triage must preserve sources and independently accept evidence."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.pbb_code import build_pbb_code
from scripts.refute_partial_catalogue import check_evidence, reconstruct, run_record


@pytest.fixture
def record():
    terms = [[0, 0], [1, 0], [2, 0]]
    other = [[0, 0], [0, 1], [0, 2]]
    code = build_pbb_code(3, 3, terms, other)
    return {"ell": 3, "m": 3, "A_terms": terms, "B_terms": other,
            "C_terms": [], "D_terms": [], "n": code.num_qudits,
            "k": code.dimension, "d": 10, "trust_level": "PARTIAL",
            "lc_any": False, "bliss_hash": "test-only-id"}


def test_search_and_separate_acceptance_do_not_modify_source(record):
    original = copy.deepcopy(record)
    saved = run_record(record, "catalogue-digest", num_trials=4, seed=0)
    check_evidence([record], "catalogue-digest", [saved])
    assert record == original
    assert saved["upper_bound"] == 2 < record["d"]
    assert saved["refutes_reported_value"] is True
    assert saved["merged_bounds"] == {"d_lower": 0, "d_upper": 2}
    assert saved["exact"] is False


def test_conflicting_certified_lower_bound_is_quarantined(record):
    record.update(d_lower=5, d_upper=10)
    saved = run_record(record, "digest", num_trials=3, seed=0)
    assert saved["merged_bounds"] is None
    assert saved["bound_conflict"]
    assert record["d_lower"] == 5
    check_evidence([record], "digest", [saved])
    saved["merged_bounds"] = {"d_lower": 0, "d_upper": 2}
    with pytest.raises(ValueError, match="unquarantined"):
        check_evidence([record], "digest", [saved])


def test_witness_only_tightens_upper_bound(record):
    record.update(d_lower=1, d_upper=10)
    saved = run_record(record, "digest", num_trials=3, seed=0)
    assert saved["merged_bounds"] == {"d_lower": 1, "d_upper": 2}
    check_evidence([record], "digest", [saved])


def test_legacy_exact_assertion_is_guarded_not_silently_overwritten(record):
    record["d_is_exact"] = True
    saved = run_record(record, "digest", num_trials=3, seed=0)
    assert saved["merged_bounds"] is None
    assert saved["bound_conflict"]
    assert record["d_is_exact"] is True and record["d"] == 10
    check_evidence([record], "digest", [saved])


def test_zero_budget_remains_unknown(record):
    saved = run_record(record, "digest", num_trials=0, seed=0)
    assert saved["upper_bound"] is saved["witness"] is None
    assert saved["refutes_reported_value"] is False
    assert saved["merged_bounds"] == {"d_lower": 0, "d_upper": 10}
    check_evidence([record], "digest", [saved])


@pytest.mark.parametrize("key,value", [
    ("input_sha256", "changed"), ("stabilizer_sha256", "changed"),
    ("exact", True), ("status", "unknown"), ("upper_bound", 3),
    ("witness", [0] * 36), ("n", 999), ("prior_reported_d", 11),
    ("refutes_reported_value", False),
    ("merged_bounds", {"d_lower": 2, "d_upper": 2}),
])
def test_saved_flags_do_not_substitute_for_verification(record, key, value):
    saved = run_record(record, "digest", num_trials=3, seed=0)
    saved[key] = value
    with pytest.raises(ValueError):
        check_evidence([record], "digest", [saved])


def test_duplicate_evidence_rejected(record):
    saved = run_record(record, "digest", num_trials=0, seed=0)
    with pytest.raises(ValueError, match="duplicate"):
        check_evidence([record], "digest", [saved, saved])


def test_reconstruction_checks_catalogue_dimensions(record):
    record["k"] += 1
    with pytest.raises(ValueError, match="n/k mismatch"):
        reconstruct(record)


def test_cli_writes_sidecar_and_checks_without_search(tmp_path, record):
    source = tmp_path / "catalogue.jsonl"
    output = tmp_path / "evidence.jsonl"
    source.write_text(json.dumps(record) + "\n")
    original = source.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/refute_partial_catalogue.py"
    command = [sys.executable, str(script), "--input", str(source), "--output", str(output)]
    searched = subprocess.run(command + ["--trials", "3"], capture_output=True, text=True)
    assert searched.returncode == 0, searched.stderr
    saved = json.loads(output.read_text())
    assert saved["input_sha256"] == hashlib.sha256(original).hexdigest()
    checked = subprocess.run(command + ["--check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert "independently checked 1" in checked.stdout
    assert source.read_bytes() == original
    repeated = subprocess.run(command + ["--trials", "3"], capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
