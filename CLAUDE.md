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

**M0–M3.5 done.** `minoodle/interfaces.py` (§4.1 ABCs + `CompositeLikelihood`, `Side`, `Seed`),
`minoodle/simdata.py` (genome-blender adapter + manifest with P1/P2/P3 phase tag;
`datasets/L0.yaml` regenerates bit-identically), `minoodle/exact.py` (toy graphs, §2.3 prior,
seed proposals, brute-force enumerator, golden fixtures in `fixtures/`), `minoodle/sampler.py` +
`minoodle/diagnostics.py` (SMC, validated against those fixtures), and `minoodle/graph.py` +
`minoodle/index.py` (GFA loader, CSR bidirected graph, k-mer index, anchors, seed weights).
**M4 (likelihood terms, one at a time) is next.** Plan rev 10 changed the state space and M3.5
redid the M1 and M2 gates on it: a path is a k-mer **seed** with a **left and a right frontier**,
alternated deterministically, each with its own STOP (§2.1–2.3, §3.2).

```bash
uv run python -m minoodle.exact verify fixtures/manifest.json
uv run python -m minoodle.sampler validate fixtures/manifest.json   # the M2 gate, ~5 s
uv run python -m minoodle.graph stats ~/Documents/minoodle_run/L0/asm/assembly_graph_with_scaffolds.gfa
uv run python -m minoodle.index check ~/Documents/minoodle_run/L0/asm/assembly_graph_with_scaffolds.gfa \
    ~/Documents/minoodle_run/L0/sim_reads_R{1,2}.fastq.gz     # the M3 gate, ~1.5 s
```

Things earlier milestones settled that later code depends on:

- `IncrementalLikelihood.init(seed)` returns `(state, log increment)`, not just `state`. The
  §4.1 sketch left the start unitig's own bases unscored, and SMC needs that number as a
  particle's initial log-weight. It is also where §2.6's score-once rule pays out: the seed
  unitig's bases are charged here, once, for both frontiers. `extend` and `stop_logp` both take
  a `Side`.
- **The left frontier is an ordinary forward walk from `seed.node.flipped()`.** Nothing in
  `graph.py` or the prior needs a mirrored code path, and `mirror` is
  `(seed.flipped(), right, left)` with the walks untouched — RC symmetry becomes a one-line
  bijection instead of a case analysis. Don't change the representation.
- **`ρ = 0.04`, because expected total length is `2/ρ`** (two geometrics, one per side). Any `ρ`
  copied from a pre-M3.5 artefact is wrong by 2× in length.
- **`ε` in `weighted_seed_proposal` caps the importance weight at `1/ε`** — an unanchored graph
  k-mer gets `q = ε/n` against `p = 1/n`. That is the number to tune it by, not "small".
- The §2.3 prior formula as written does not normalise; `exact.py` implements the per-base
  geometric its prose describes (terminal factor `1 - (1-ρ)^{len_T}`, forced STOP at dead
  ends). `Σ p(x) == 1` on an acyclic toy graph is the test that guards it, and the sampler's
  counterpart — prior-only `log Ẑ == 0.0` exactly — is the sharpest test in `test_sampler.py`.
  Under rev 10 that test holds only with **uniform seeding** (`q_seed = p_seed`): the coverage
  proposal makes the seed weight vary per particle, so `log Ẑ` is 0 in expectation, not exactly.
- **Seeding is a proposal, not the prior** (D18). `p_seed` uniform over oriented k-mer
  positions; `q_seed` coverage-weighted from the read anchors, **normalised** by total anchored
  placements (`log Ẑ` is not invariant to a constant in `q`, unlike the posterior) and mixed
  with `ε` uniform so no graph k-mer has `q = 0` against `p > 0`. Initial log-weight carries
  `log p_seed - log q_seed`. Swapping `q_seed` must not move the posterior — that invariance is
  the test that catches a mis-normalised or partial-support `q`. That gate sits at **3×** the
  iid p99, not the M2 gate's 1.25×, and the multiplier was calibrated rather than guessed: a
  bad proposal is *meant* to be less efficient than iid, and dropping the `log p - log q`
  correction moves TV from 0.03 to 0.29. `log Ẑ` does not discriminate here (0.007 off).
- **STOP is one of the fully-adapted alternatives** in the SMC step, not a separate case; §3.2's
  pseudocode shows out-edges only. Anything that changes the alternative set changes the target.
  Under rev 10 this is *per side*: the alternative set is one frontier's out-edges plus that
  side's STOP, and the particle is absorbed only when both sides have stopped.
- Truncation at `max_bases` must match `enumerate_paths` exactly: drop the over-long extension,
  but still score STOP by *graph* out-degree, not "nothing legal left". The budget is over total
  length, so the two sides reach it mid-alternation and asymmetrically. `repeat_twice`'s budget
  is deliberately tight (120 bases, 2e-3 of mass truncated) so that a divergence in this rule —
  the one the two implementations have historically disagreed on — is visible rather than
  buried at 1e-8. When both sides are impossible (no bases left to stop in, every extension
  over budget) the enumerator drops the subtree and the sampler kills the particle.
- **Extension direction is never chosen by the data** — deterministic alternation only, else the
  target goes history-dependent (§6 item 11). Same defect as a taboo list.
- The seed is part of the state: `π` is over `(seed, path)` and `Σ p(x) == 1` sums over both.
  Sequence-level output is the marginal, via §3.4's summed-weight dedup.
- The `TV < 0.01` gate is enforced against an exact-iid reference band, not the flat number —
  242 atoms at N = 1e5 put a perfect sampler above 0.01 on `repeat_twice` (M2 finding 1).
  Two-sided states multiply by the seed count, so that graph now has 6432 atoms and even the
  cheap tests in `test_sampler.py` have to scale their threshold with the band (`_tv_budget`).
- Adaptive resampling never fires on the toy graphs (ESS stays ≥ 0.84 N), so that code path is
  only covered by tests that force it with `ess_frac=1.0`. Watch for the same blind spot when
  adding anything that only runs under degeneracy. Related: the "resamples" count inferred from
  `parents != arange` undercounts, because systematic resampling on near-uniform weights can
  return the identity permutation.
- **`metaspades.py -k 21` writes `21M` overlaps, so `k` is 22 in this codebase** (M3 finding 1).
  `UnitigGraph.from_gfa` infers k from the `L` lines and asserts against any k you pass; don't
  hard-code it. `KC/DP` on a metaSPAdes segment equals `L-k+1`, which confirms the same k.
- The `cov_base = cov_kmer · L/(L-k+1)` conversion is implemented as §5 M3 specifies, but it is
  a ~2.9× multiplier at L0's median unitig length and unbounded as `L → k` (M3 finding 2). M4
  item 1 decides what §2.7 does about short unitigs; don't silently change the formula.
- The anchor table caches the rival placement's *identity and votes*, not §2.5's competing
  **score** — that needs the §2.5 read scorer and lands with M4.
- skiver's error model is **not** an HMM: it is a customisable composable context model whose
  component string is per-dataset, evaluated to a per-position `[L, 10]` emission table used
  for both scoring and read simulation. A latent-state layer is an optional component and is
  usually off. Read scoring starts at fixed alignment (edlib) + flat gap penalty; the pair-HMM
  is §2.5 level 2, taken only on ablation evidence. `~/Documents/superresolution-amplicon`
  (`bin/subspecies_infer.py`, `vendor/skiver`) is the worked example of both directions.

Follow the plan's milestone order (§5) — M1 before the sampler, M6 (fixture freeze) before any
Rust port, etc. Milestones are hard gates: do not proceed past a failing one. The remaining
order is `M4 → M5a (synthetic ladder) → M6 → M7 (Rust) → M5b (real data, HPC) → M9`: the
real-data rungs run on the Rust build, under the §5.5.4 HPC discipline.

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

Every *other* external tool runs in a container, never a local install (D17): biocontainers
under Docker locally, singularity via the nextflow pipeline on HPC. Record the image **digest**,
not the tag. M3's assembly, and the pattern to copy:

```bash
docker run --rm --platform linux/amd64 -v "$PWD:/data" \
  quay.io/biocontainers/spades:4.3.0--hde4eca7_0 \
  metaspades.py --only-assembler -k 21 \
  -1 /data/sim_reads_R1.fastq.gz -2 /data/sim_reads_R2.fastq.gz -o /data/asm
```

Provenance (image digest, command, output hashes) goes next to the outputs — see
`~/Documents/minoodle_run/L0/asm/provenance.json`.

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

Python first (numpy, numba `@njit` for read scoring and the SMC weight loop, pytest + hypothesis,
pyarrow), Rust port later using the Python implementation as test oracle via injected-uniforms
golden fixtures (§4.2). Do not start the Rust port before M6.
