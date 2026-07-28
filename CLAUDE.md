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

Pre-M0: only the plan and repo scaffolding exist. No package code yet. Follow the plan's
milestone order (§5) — M1 (exact enumerator) before the sampler, M6 (fixture freeze) before any
Rust port, etc. Milestones are hard gates: do not proceed past a failing one.

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
