# Independent RIS upper-bound witnesses for Campaign 7 PARTIAL entries

Issues #13 and #17 describe a missing independent distance-refutation channel:
BP-OSD and shallow MILP can agree on an overestimate. A normalizer search finds
smaller logical operators for **all seven** entries selected by `trust_level=PARTIAL` and `lc_any=false`
in `results/campaign7_publication_merged.jsonl`. These are upper bounds, not
optimal-distance certificates. The original catalogue is unchanged.

## Reproduction

Base revision: `db051c576f51c43d2bf9923b6a87033e7f88f0fe`.
Python 3.12.14, dependencies installed with `uv sync --frozen --group dev`.
The checked-in sidecar is `results/campaign7_partial_ris_witnesses.jsonl`.

```bash
# Verify the saved operators without running the search:
uv run python scripts/refute_partial_catalogue.py --check

# Re-run the bounded search, preserving the existing sidecar:
uv run python scripts/refute_partial_catalogue.py \
  --trials 600 --seed 0 --output /tmp/campaign7_partial_ris_replay.jsonl
uv run python scripts/refute_partial_catalogue.py \
  --output /tmp/campaign7_partial_ris_replay.jsonl --check

# Tests, including the two previously reported collapses:
uv run python -m pytest tests/test_distance_ris.py tests/test_ris_catalogue.py -q
```

The sidecar records the complete input-file SHA-256, matrix SHA-256 with shape
and coordinate convention, polynomial specification, seed, search budget,
actual trials, elapsed time, and full `[x|z]` witness. `--check` reconstructs
each code, checks its n/k, and verifies both the operator and recorded bounds.
It does not trust an `exact` flag or use the search's elimination routines.

| Catalogue `bliss_hash` | n | k | Prior reported value | Witness proves | Search seconds |
|---|---:|---:|---:|---:|---:|
| `68d93d677268a3fd` | 360 | 10 | 40 | d ≤ 16 | 12.727 |
| `6f76d175b6c3996b` | 360 | 10 | 32 | d ≤ 20 | 12.802 |
| `481101301eadfc29` | 360 | 10 | 32 | d ≤ 20 | 11.951 |
| `ca906ef77eaee0ed` | 360 | 10 | 30 | d ≤ 20 | 13.337 |
| `3d5510c528b2e841` | 360 | 10 | 30 | d ≤ 20 | 11.707 |
| `67c015a84425ea3a` | 360 | 10 | 30 | d ≤ 20 | 13.388 |
| `70c03e15fe2839e4` | 180 | 6 | 21 | d ≤ 14 | 3.341 |

Each run used 600 trials, seed 0, 16 light rows for pair combinations and no
random local frames. Times measure the search plus witness checking, exclude
code construction, and depend on the machine. They are not a general speedup
benchmark against BP-OSD or MILP. Already-heuristic values remain valid loose
upper bounds; the smaller witnesses rule them out as *exact* distances.

## Mechanism and acceptance

For binary stabilizer rows S in `[x|z]` convention, compute the normalizer
`N = ker(S Lambda)` and extend a basis of S to obtain the quotient `N/S`.
A vector in N is nontrivial precisely when its symplectic pairing with some
quotient basis element is nonzero. RIS visits columns in random orders to
obtain systematic normalizer bases, then considers every basis row and XORs
of pairs among the 16 lightest rows. It checks the **combined** candidate's
quotient class, since two logical representatives can XOR into a stabilizer.

The objective counts physical support: `popcount(x | z)`. Y on a single
qubit has weight one even though two binary coordinates are set. Optional
single-qubit symplectic frames preserve this objective and are inverted
before returning the witness. On `68d93d677268a3fd`, seed 0 reached weight
16 at trial 117 without frames (2.397 s) versus trial 508 with frames
(10.191 s). This limited comparison motivates leaving frames opt-in, not
a universal claim about which mode is better.

The independent checker uses dense NumPy GF(2) elimination rather than the
search's bit-packed Python integers. It checks binary shape, commutation,
zero syndrome, increased rank when adjoining the witness to S, and physical
weight. Matrix validation accepts dependent/zero check rows but rejects
nonbinary and noncommuting inputs.

This uses standard randomized information-set search and stabilizer quotient
algebra, not a claim of a new mathematical algorithm. Related RIS work is
already available in `unitaryfoundation/qldpc-challenge/verify/heuristic_distance.py`
and CSS witness searches in this repository. This contribution supplies a
generic non-CSS API, independent acceptance, and a reproducible triage path.

## Validation

The complete final suite on revision `b627c090803a22ab3f49e06fb85360583c02a276`
finished with **633 passed, 0 failed, 1 skipped** in **1690.40 seconds**.
No slow or Webster markers were excluded. The 55 new tests all passed.
The full case inventory, per-run outcomes/timings, environment, fixtures and
original failures are recorded in `results/ris_full_test_audit_20261013.json`.

Three complete runs were retained:

| Run | Revision | Passed | Failed | Skipped |
|---|---|---:|---:|---:|
| Unchanged upstream base | `db051c576f51c43d2bf9923b6a87033e7f88f0fe` | 577 | 1 | 1 |
| Initial RIS change | `0075b030dcb00524c8b0cdcecb45f49df23a19bf` | 631 | 2 | 1 |
| Final repaired change | `b627c090803a22ab3f49e06fb85360583c02a276` | 633 | 0 | 1 |

The shared failure was an existing stale referee artifact. Rebuilding it
changed exactly three source SHA-256 references; all numerical/scientific
contents remained identical. The additional initial failure came from an
untracked temporary file already present before RIS work, which the recursive
control-manifest scan included. Its 18,350,080 bytes were preserved outside
the experiment inputs; they are an exact prefix of tracked `draws.jsonl`.
No tracked experimental records or test assertions were modified. Six focused
checks passed after these repairs, then the entire final suite ran again.

The sole skip is the upstream OSD-CS test marked as unsafe because of a native
crash. Its **original test body was additionally executed in isolated
processes on both the base and initial RIS revisions**: both terminated with
**SIGSEGV (signal 11)** in the existing `ldpc` decoding path. This is a known,
reproduced native-library defect, not a passing test or a fixed decoder.
RIS does not invoke that decoder. The final suite also reported 16 warnings.

Coverage includes exhaustive tiny-code oracles, both optional frame modes,
dependent checks, Y weights, reproducible seeds, corrupted/trivial operators,
budget exhaustion, bound conflicts, legacy exact-claim conflict guards, the
CLI, and both real previously reported collapses. All seven saved witnesses
also pass the separate `--check` command.

Full replay dependencies and command:

```bash
uv sync --frozen --group dev --group evolve --group webster
uv run --no-sync python -X faulthandler -m pytest tests/ -vv --durations=30
```

The two fixture-dependent referee tests require a checkout at
`/tmp/qldpc-challenge-src`, pinned to
`3b50503b058a0c09500c1017b54c3d547e1101cc`, with
`codes/96-4-12.json` and `codes/140-6-14.json`. Their SHA-256 values are in the
case ledger. For a fresh fixture path:

```bash
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/unitaryfoundation/qldpc-challenge.git /tmp/qldpc-challenge-src
git -C /tmp/qldpc-challenge-src sparse-checkout set --no-cone \
  /AGENTS.md /LICENSE /codes/96-4-12.json /codes/140-6-14.json
git -C /tmp/qldpc-challenge-src fetch --depth 1 origin \
  3b50503b058a0c09500c1017b54c3d547e1101cc
git -C /tmp/qldpc-challenge-src checkout --detach \
  3b50503b058a0c09500c1017b54c3d547e1101cc
```

The historical evaluator object must also be available in the qcode-discovery
object database (a shallow checkout may need this fetch):

```bash
git fetch --depth 1 https://github.com/qiskit-community/qcode-discovery.git \
  aa0f18b600e38c4f5459a1fbfaee0836d9cf99ab
```

The original targeted suite was 149 passed; that partial milestone is now
superseded by the completed full runs. The initial two full runs were
concurrent and the final run was alone, so elapsed times are not a speedup
benchmark. The earlier interrupted runs and their unexplained native message
are historical; they are not substituted for the completed results above.

## Limits and integration

- A search result is either a witnessed upper bound or UNKNOWN. Exhausting
  a trial/time budget never raises a lower bound. k=0 returns UNKNOWN with
  `no_logical_qubits`, not an exact-distance claim of zero.
- The sidecar uses existing `distance_schema.merge_bounds` to tighten only
  the upper bound. Explicit prior lower bounds stay unchanged, and legacy
  exact assertions are retained as conflict guards without validating their
  certificates. A conflicting witness is quarantined, not silently merged.
- No publication/Pareto code reads this sidecar automatically. Maintainers
  can inspect and incorporate the independently checked bounds; certified
  lower bounds still require exhaustive or solver evidence.
- The default workflow is manual, inexpensive triage before deeper MILP.
  It does not add a stochastic mandatory stage to existing evaluators.
- Wall-clock limits include setup and final checking but are only checked
  between trials, so they are soft limits. Use trial budgets for regression
  reproducibility. Witness checking remains useful even across search RNG
  changes; matrix hashes are presentation hashes, not equivalence hashes.
