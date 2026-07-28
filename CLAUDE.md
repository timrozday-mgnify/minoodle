# minoodle

Probabilistic sampling of sequences from a metagenome assembly graph (SMC over paths in a
metaSPAdes GFA), as an alternative to rules-based consensus assembly. See
[README.md](README.md) for the one-paragraph summary.

## Authoritative spec

**[docs/minoodle-implementation-plan.md](docs/minoodle-implementation-plan.md)** is the spec.
It is written for an autonomous coding agent and is load-bearing — the statistical formulation
in particular (§2) is to be implemented literally, not "improved." Read it before writing code
in this repo. It covers: state space and target distribution, likelihood terms and why each is
built the way it is, SMC inference design, module layout, milestones with hard gates, and open
decisions (§8).

## Current status

**M0, M1, M2 done.** `minoodle/interfaces.py` (§4.1 ABCs + `CompositeLikelihood`),
`minoodle/simdata.py` (genome-blender adapter + manifest with P1/P2/P3 phase tag;
`datasets/L0.yaml` regenerates bit-identically), `minoodle/exact.py` (toy graphs, §2.3 prior,
brute-force enumerator, golden fixtures in `fixtures/`) and `minoodle/sampler.py` +
`minoodle/diagnostics.py` (SMC, validated against those fixtures). **M3 (metaSPAdes GFA and
index) is next.**

```bash
uv run python -m minoodle.exact verify fixtures/manifest.json
uv run python -m minoodle.sampler validate fixtures/manifest.json   # the M2 gate, ~5 s
```

Things earlier milestones settled that later code depends on:

- `IncrementalLikelihood.init` returns `(state, log increment)`, not just `state`. The §4.1
  sketch left the start unitig's own bases unscored, and SMC needs that number as a particle's
  initial log-weight.
- The §2.3 prior formula as written does not normalise; `exact.py` implements the per-base
  geometric its prose describes (terminal factor `1 - (1-ρ)^{len_T}`, forced STOP at dead
  ends). `Σ p(x) == 1` on an acyclic toy graph is the test that guards it, and the sampler's
  counterpart — prior-only `log Ẑ == 0.0` exactly — is the sharpest test in `test_sampler.py`.
- **STOP is one of the fully-adapted alternatives** in the SMC step, not a separate case; §3.2's
  pseudocode shows out-edges only. Anything that changes the alternative set changes the target.
- Truncation at `max_bases` must match `enumerate_paths` exactly: drop the over-long extension,
  but still score STOP by *graph* out-degree, not "nothing legal left".
- The `TV < 0.01` gate is enforced against an exact-iid reference band, not the flat number —
  242 atoms at N = 1e5 put a perfect sampler above 0.01 on `repeat_twice` (M2 finding 1).
- Adaptive resampling never fires on the toy graphs (ESS stays ≥ 0.84 N), so that code path is
  only covered by tests that force it with `ess_frac=1.0`. Watch for the same blind spot when
  adding anything that only runs under degeneracy.

Follow the plan's milestone order (§5) — M1 before the sampler, M6 (fixture freeze) before any
Rust port, etc. Milestones are hard gates: do not proceed past a failing one.

CI landed at M2 (`.github/workflows/ci.yml` runs ruff, pytest, `exact verify` and the full
sampler gate). Still deferred: mypy (M3/M4, once the type surface stops moving), `bench/` (M6),
committed uniform-stream fixtures (M6's freeze). The §4 module layout is created per milestone,
not scaffolded empty.

### Working with datasets

Generated data lives outside the repo, in `~/Documents/minoodle_run/<dataset>/`:

```bash
uv run python -m minoodle.simdata run datasets/L0.yaml --out ~/Documents/minoodle_run/L0
uv run python -m minoodle.simdata verify ~/Documents/minoodle_run/L0/manifest.json
```

genome-blender is invoked as a shell command from its own conda env (`generate_reads_cmd` in the
dataset YAML), not imported — the two projects share no environment.

Manifests hash file *content*: `.gz` outputs are hashed decompressed, because gzip's header
mtime otherwise makes byte-identical reads look non-reproducible. genome-blender at
`6e1efe0` is verified deterministic under a fixed seed.

## Constraints an agent should not accidentally violate (plan §6)

- Every likelihood term must be incrementally decomposable (§2.4) — computable from a bounded
  recent window. If a term can't be written that way, it doesn't belong in v0.
- No likelihood-threshold stopping (§2.8) — termination is via the geometric prior only.
- No taboo/visited-node exclusion for diversity (§3.3) — it makes the target history-dependent.
- Don't mix "insert size" terminology — `f_insert` is fragment length (outer distance), never
  inner distance (§2.6).
- P1/P2 error-model results (§2.5) are circular by construction; never quote them outside the
  repo. Only P3 (skiver trained on real Zymo reads) numbers are externally quotable.
- No N50 or contig-style contiguity metrics as evidence of correctness (§5.5.3).

## Stack (D1)

Python first (numpy, numba `@njit` for the pair-HMM and SMC weight loop, pytest + hypothesis,
pyarrow), Rust port later using the Python implementation as test oracle via injected-uniforms
golden fixtures (§4.2). Do not start the Rust port before M6.
