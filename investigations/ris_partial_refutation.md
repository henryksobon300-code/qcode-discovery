# Independent RIS upper-bound witnesses for Campaign 7 PARTIAL entries

Issues #13 and #17 describe a missing independent distance-refutation channel:
BP-OSD and shallow MILP can agree on an overestimate. A normalizer search finds
smaller logical operators for **all seven** genuinely non-CSS PARTIAL entries
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

The targeted suite passed **149 tests** (including **55 new tests**):

```bash
uv run python -m pytest tests/test_distance_ris.py tests/test_ris_catalogue.py \
  tests/test_distance_schema.py tests/test_distance_milp_bounds.py \
  tests/test_pbb_code.py tests/test_noncss_gate.py -q
```

Coverage includes exhaustive tiny-code oracles, both optional frame modes,
dependent checks, Y weights, reproducible seeds, corrupted/trivial operators,
budget exhaustion, lower-bound conflicts, legacy exact-claim conflict guards,
the CLI, and both real previously reported collapses. All seven saved
witnesses also pass the separate `--check` command.

The full `tests/ -m 'not slow and not webster'` run was attempted but **not
completed**. It was interrupted while the existing catalogue-wide Clifford
equivalence audit ran; an unchanged base worktree also reached the same
expensive test (`TestLCCatalog::test_all_368_not_lc_css`) and was interrupted.
The changed-tree run emitted a native-library message (`fatal library error,
lookup self`) before that interruption; its cause was not established. This
is not a claim that the full repository suite passed or that this message was
reproduced on the base revision. No unrelated tests or runtime code were
changed to work around it.

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
