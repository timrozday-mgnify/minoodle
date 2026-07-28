# minoodle — implementation plan (rev 12)

Probabilistic sampling of sequences from a metagenome assembly graph, as an alternative to
rules-based consensus assembly.

**Audience:** an autonomous coding agent (Claude Code or equivalent) with repo write access.
**Progress:** M0–M3.5 complete; M4 item 1 (coverage) is implemented and validated against the
exact enumerator on the toy graphs, but **its L1 gate fails for a diagnosed structural reason**
(item 1 finding 6 — at a repeat the term pays for looping instead of preventing it, so any
cycle runs away and §2.8's termination is defeated). Each milestone records what shipped, what
was deferred, and the findings from the run. M4 item 2 — single-end read congruence — is next;
M4's milestone gate is not met until §2.4's open multiplicity question is answered.
Note M1's finding 1: the §2.3 prior is implemented as the normalised per-base geometric.
Note M2's finding 1: the `TV < 0.01` gate is restated against an exact-iid reference band.
Note M3's finding 1: `metaspades.py -k 21` gives `21M` overlaps, so k = 22 in this codebase.
Note M3.5's finding 1: `ρ` is 0.04, because expected total length is `2/ρ`.
Note M4 item 1's finding 2: **every** term is scored as a log-odds, not just §2.5's.
**Changes in rev 12:** M4 item 1 built and recorded with seven findings and a **failing L1
gate** — chief among them that the coverage term *pays* for traversing a repeat again rather
than preventing it (finding 6), which §2.4 now carries as an open question. Also: the
k-mer→base coverage conversion corrected to the constant `R/(R−k+1)` (closing M3 finding 2),
§2.5's log-odds rule generalised to every term, coverage starving the rare organism, and
`branch_marginals` ceasing to be a probability once paths are long. `datasets/L1.yaml` added.
**Changes in rev 11:** M3.5 recorded as done, with its five findings — the `mirror` map that
makes RC symmetry a one-liner, the `ε` importance-weight cap, and the calibration of the
proposal-invariance gate against the failure it exists to catch.
**Changes in rev 10:** paths are **two-sided** — a k-mer seed with a left and a right frontier,
grown by deterministic alternation (§2.1, §2.3, §3.2). The seed is sampled from the reads as a
**proposal** against a uniform prior (D18, resolved), which supplies §3.3's stratified starts for
free and keeps the target data-independent. This re-opens M1 and M2, so **M3.5** (§5, new) redoes
those gates on the two-sided prior before M4 starts.
Endpoint-conditioned bridging — seeding both mates of a pair and scoring closure — was
considered and **rejected** for v0; §7 records why.
**Changes in rev 9:** §2.5 rewritten — the skiver error model is a customisable composable
context model producing a per-position `[L, 10]` emission table, used for both scoring and
read simulation; the latent-state HMM layer is an optional component that is usually off; and
the pair-HMM is demoted to the second of two alignment-marginalisation levels, behind a
fixed-alignment default (D3 amended). Worked example: `superresolution-amplicon`.
**Changes in rev 8:** M3 recorded as done, with its five findings — chief among them that
`-k 21` means k = 22 here; D17 added (external tools run in containers); §5.5.4's environment
paragraph now names the mechanism.
**Changes in rev 7:** M5 split into M5a/M5b with optimisation between them (D16); §5.5.4 HPC
execution discipline added.
**Changes in rev 6:** candidate R2 accessions recorded; §5.5.2c verification protocol added.
**Changes in rev 5:** D11-D13 resolved; §5.5.2a (R2 in-scope set) and §5.5.2b
(reference verification) added; D14-D15 opened.
**Changes in rev 4:** name settled on `minoodle`.
**Changes in rev 3:** D6-D10 resolved; §5.5 benchmark data ladder added; insert-size
terminology pinned down in §2.6; skiver training strategy in §2.5.

---

## 0. Naming — settled

`minoodle` is clear. No PyPI package, no crates.io crate, no bioinformatics tool. The only
things using the string are an Amazon storefront, a small Minecraft mod ("Minoodles"), a
dormant WordPress blog, and a now-closed New York noodle bar — none of which will ever
compete for the search term in a scientific context.

Two residual notes, neither blocking:
- It reads as a diminutive of `noodle`, so if a reader knows the project history the name
  looks like a joke. That is fine for a methods tool; it is worth a one-line note in the
  README explaining it, because reviewers do ask.
- Register `minoodle` on PyPI early, even as an empty placeholder, so it stays clear.

Use `minoodle` throughout. Rename now, before the first commit.

---

## 1. Scope

### v0 in scope
- Single sample, paired-end short reads.
- Input: metaSPAdes GFA (single-k, `--only-assembler`) + FASTQ.
- Output: a **weighted** multiset of sequences with proper importance weights, `log Ẑ`, and
  per-branch marginal probabilities.
- Development target: 16S-derived synthetic shotgun data at **k=21**.

### v0 out of scope
Taxonomic assignment, binning, MAGs, joint strain deconvolution, long reads, multi-sample
differential coverage, base-space reconstruction from minimizer space (§7).

---

## 2. Statistical formulation

Load-bearing. Implement literally; do not "improve" it.

### 2.1 State space

A path is **two-sided**: a seed position with a left and a right frontier, each extended
unitig-by-unitig.

```
seed  = (v_0, o_0, offset_0)   # a k-mer position: oriented unitig + base offset
x_t   = (seed, F_L, F_R, s_L, s_R)
F_L, F_R                       # left/right frontiers, each (v, o, offset)
s_L, s_R ∈ {open, stopped}     # per-side termination flags
```

Absorbing **STOP** is `s_L = s_R = stopped`. Likelihood is evaluated at base resolution
internally, as before. Four consequences, each of which something later depends on:

- **Direction alternation is deterministic**, not chosen by the data: step `t` extends the
  right frontier when `t` is even and the left when odd, skipping a side that is `stopped`.
  Choosing the direction by likelihood would make the target history-dependent for the same
  reason §3.3 rules out taboo lists.
- **The seed is part of the state.** `π` is a distribution over `(seed, path)` pairs; the
  distribution over *sequences* is its marginal, obtained by the summed-weight deduplication
  §3.4 already specifies. `Σ p(x) == 1` is therefore checked over `(seed, path)`.
- **The prior becomes RC-symmetric**, up to the seed's own orientation — M1's finding 3 (prior
  directional, likelihood RC-symmetric) is the wart this removes. A seed and its reverse
  complement generate mirror-image path sets with equal prior mass.
- **Seeds sit mid-unitig.** `offset_0` is a base offset, so a path's endpoints are not forced
  to unitig boundaries. This is what makes per-branch marginals (§3.4) supportable by any seed
  near the branch rather than only by seeds upstream of it.

### 2.2 Target

```
π(x) ∝ p(x) · L(x)
```

`L` is a **pseudo-likelihood** — a product of local congruence factors — not a true generative
likelihood of the read set, because reads come from the whole community rather than one path.
The sampler is exact with respect to `π`; whether `π` is a posterior is a separate, weaker
claim. State this in the README from day one. "Exact samples from a defined pseudo-posterior"
is defensible; "posterior samples" is not, and the difference costs nothing to say.

### 2.3 Path prior

Two sides, each an independent chain with its own termination:

```
p(x) = p_seed(seed) · C(left extensions) · C(right extensions)
C(side) = Π_t [ (1 - ρ)^{len_t} · p_edge(x_{t+1} | x_t) ] · ρ
```

- `p_seed` is **uniform over oriented k-mer positions in the graph** (D18). Not
  coverage-weighted: coverage-proportional seeding is a *proposal*, below. Uniform over
  *oriented* positions, so a seed and its reverse complement carry equal mass and the
  RC symmetry of §2.1 holds exactly. The old `p_start ∝ (unitig length × mean coverage)` is
  retired — its coverage factor moves into `q_seed`, its length factor is what uniform-over-
  positions already gives you.
- `p_edge` uniform over out-edges — branch information lives in `L`, not the prior, so
  ablations mean something.
- `ρ` = per-base stop probability, applied **per side**. Total length is a sum of two
  geometrics rather than one, so the length prior is negative-binomial-shaped with mode away
  from zero. Expected total length is `2/ρ`; keep that in mind before reusing an old `ρ`.
- Forced STOP at a dead end applies to that side only; the other side keeps going.
- As in M1's finding 1, this normalises only when the per-base geometric of the prose is
  implemented (terminal factor `1 - (1-ρ)^{len}` on each side), not the formula read literally.

**Seeding is a proposal, not the prior (D18).** Seeds are drawn from `q_seed` — sample a read
uniformly, then a k-mer position uniformly from its anchor (§4.1 `index.py` already produces
the anchors) — and the particle's initial log-weight carries `log p_seed(s) - log q_seed(s)`.
That keeps the target a fixed, data-independent measure over `(seed, path)` while still
spending the particle budget where the reads are. It also *is* §3.3 item 5's stratified starts,
so that deferred item lands here rather than separately.

Four things this costs, all of which have to be handled at M3.5:

- **`q_seed` must be normalised, not just proportional.** Self-normalised weights make the
  posterior invariant to a constant factor in `q`, but `log Ẑ` is not — and `log Ẑ` is the
  quantity §3.1 justifies the whole SMC choice with. The normaliser is exactly computable:
  the total number of anchored read-k-mer placements, which `index.py` already counts.
- **`q_seed` needs full support.** A graph k-mer with no read anchored on it has `q_seed = 0`
  and `p_seed > 0`, which biases the estimator rather than merely inflating its variance. Use
  a mixture `q = (1-ε)·anchored + ε·uniform`, `ε` small but not zero, and assert support before
  sampling rather than discovering a zero divisor mid-run.
- **Prior-only `log Ẑ == 0.0` is no longer exact under the coverage proposal.** With `L ≡ 1`
  the weights are `p_seed/q_seed`, which vary per particle, so `log Ẑ` is 0 only in
  expectation. Keep the exact test by running it with `q_seed = p_seed` (the uniform-seeding
  flag) — it stays the sharpest test in the suite, just under a stated configuration.
- **Proposal invariance becomes testable, and is the better test.** The target does not depend
  on `q_seed`, so uniform seeding and coverage seeding must give the *same* posterior within
  Monte Carlo error. That check catches a mis-normalised or partial-support `q` in a way
  nothing in the current suite does. Add it at M3.5.

**The seeding read is used twice** — once to place the seed, once again when term A scores it.
Proposal-only seeding makes this a variance/efficiency question rather than a
double-counting-in-the-target one, since `q` is not part of `π`. Still record it with the
ablation rather than discovering it in a calibration plot.

### 2.4 Incremental decomposability — hard constraint

```
log π(x_{1:t+1}) - log π(x_{1:t}) = γ_t(x_{1:t+1})
```
computable in bounded time from the last `W = max_fragment + read_len` bases. Any term that
cannot be written this way does not go in v0. Enforced by the ABC in §4.1.

**Open: unitig multiplicity vs the bounded window.** M4 item 1 finding 6 measured the first
real collision with this constraint. §2.7 as written asks "is this unitig's depth consistent
with `λ`", which is bounded-window and wrong: the running posterior absorbs the unitig's own
count, so re-traversing it scores *better* each time (+1.566, +1.672, +1.720, … nats on
`repeat_twice`) and the term pays for looping instead of preventing it. Preventing looping is
the other question — "how many traversals would make this unitig's depth consistent with `λ`",
i.e. score `y ~ NegBin(c · λ · m)` for the multiplicity `c` the path assigns and never update
`λ` from a unitig already counted — and **`c` is a property of the whole path**, so an exact
version needs per-unitig counts over unbounded history. That is precisely what this section
forbids.

Three ways out, to be decided before M4's gate can be met:

1. **Window-bounded multiplicity.** Keep counts only for unitigs seen within the last `W`
   bases. Catches short cycles, which are the ones that run away; a repeat re-entered from
   further off is scored as new. Stays inside §2.4 by construction, and is the default unless
   the ablation says otherwise.
2. **Multiplicity as part of the state, not the history.** Carry `c` per unitig in the particle
   and accept state that grows with the number of *distinct* unitigs visited. Bounded per step
   but not per path — a weakening of §2.4 that must be written down as such, not slipped in.
3. **Leave coverage local and let another term stop the looping.** §2.6's censored pairs (M4
   item 4) already have the job of suppressing spurious long paths. Cheapest, and it keeps
   §2.7 honest about being local self-consistency only — but it leaves copy number unmodelled,
   and finding 6 shows the term cannot tell 2× from 4× at all.

Do not resolve this by bounding the per-unitig increment with a constant: that changes the
target to fit a gate (§6 anti-goal 4).

### 2.5 Term A — read congruence under the skiver error model

**The error model is a customisable per-position emission model, not an HMM.** A fitted skiver
model is a *composable context model*, specified by a component string —
`AdditiveContext(N)` or `BaseContext(N)` (required), optionally `+Homopolymer`, `+Strand`,
`+Position(n)`, `+PhredContext(n)`, `+FragmentOverdispersion`. Which components are in play is
a per-dataset choice, usually made by fitting a candidate list and keeping the min-AIC winner
(`superresolution-amplicon`: `bin/build_model_config.py` writes the model-config JSON,
`skiver/scripts/train_context_error_models.py` fits it, `bin/pick_best_model.py` picks). Treat
the component set as data, not as something this plan fixes.

Evaluated against a reference sequence the model yields, per reference position, a categorical
over 10 error types — match, 4 substitutions, 4 insertions, deletion:

```python
from lib.error_application import ErrorModel, _masked_probs, _CHAR_TO_IDX
model = ErrorModel.load(model_pt)                                  # trained .pt
probs = _masked_probs(model._logits_for_reference(ref_idx, is_forward), ref_idx)   # [L, 10]
```

That `[L, 10]` table is the entire interface, and it runs in both directions:

- **Score** — "were these two sequences produced from the same source, differing only by
  sequencing error?" Drop the indel mass, renormalise to an `[L, 4]` emission distribution,
  align the two sequences, and sum the per-column emission log-prob (see the fixed-alignment
  scorer below). `emission_distribution` / `read_score` in
  `superresolution-amplicon/bin/subspecies_infer.py`.
- **Sample** — `lib.error_application.apply_read(model, name, reference, rng, ...)` draws one
  error type per reference position from the same table and assembles the read, CIGAR and
  Phred string (quality from a separately fitted `P(Q | error_type)` calibration). This is how
  synthetic reads are produced; there is no forward/backward pass anywhere in it.

**A latent state is an optional component, and usually absent.** skiver *can* carry a
read-level latent HMM layer (`_HMMLayer`, `lib.hmm_latent_state`) whose states multiply the
local error probability by a per-state scale — a quality-regime model, sampled as a state
trajectory when generating. It is off in the bundled platform presets and in the amplicon
pipeline. So: code against the `[L, 10]` table, let the layer modulate it when a model has one,
and do not design anything that assumes latent states exist.

Two constraints that fall out of this and are easy to trip over:

- Not every fitted model is usable generatively. `_reject_unsupported` refuses PhredContext and
  Position covariates when simulating, because quality and read position are outputs, not
  inputs. A model trained for scoring may therefore be unusable for P1/P2 read simulation;
  fit the generative and scoring models from the same data rather than assuming one artefact
  serves both.
- The model is conditioned on preceding *consensus* bases, so scoring needs an alignment to
  supply that context — which is the next decision, and it is separate from the error model.

**Alignment marginalisation — two levels; start at the cheap one.**

1. **Fixed alignment (default).** One edlib alignment (`task="path"`, infix mode so read
   overhang past the reference is free), sum the per-column emission log-prob, charge a flat
   `gap_penalty` per indel column, take the better of the two orientations, length-normalise.
   No HMM, no forward pass, and it is what the amplicon pipeline uses in production
   (`_oriented_score` / `read_score`).
2. **Pair-HMM (only if 1 measurably loses information).** States Match / Insert / Delete;
   match and substitution emissions read off the same `[L, 10]` table; gap-open from the
   aggregate indel mass; gap-extend from `P(error at i+1 | error at i)` — skiver's
   next-error-depends-on-previous structure *is* gap-extend, and that conditional is the one
   number skiver does not emit (D3): P1/P2 take it from the simulator's known parameters, P3
   measures or fits it from Zymo reads against the reference genomes. Scored by a banded
   forward around the anchor diagonal.

M4 builds (1) and reports the ablation; (2) is justified by that ablation, not assumed. Level 1
is an edlib call and a gather-plus-sum; level 2 is where numba would earn its keep.

**Training strategy (D6, D10).** Three phases, each with a known and bounded weakness:

| Phase | Error model source | What it tests | Weakness |
|---|---|---|---|
| P1 | Same model used by genome-blender to simulate | Sampler mechanics | Fully circular. Results are an upper bound, not a performance estimate. Never quote them externally. |
| P2 | Simpler skiver model trained on the synthetic data | Robustness to model misspecification | Still generated by the same simulator, so it **bounds** the circularity rather than removing it. This is its value: a controlled misspecification test with a known direction and magnitude. Label it as such. |
| P3 | skiver trained on real Zymo shotgun reads | Actual performance | The only phase whose numbers can leave the repo. |

Ship each fitted model as its artefact set — the `.pt`, the Phred-calibration JSON, and the
**component string it was fitted with** — versioned and tagged with its phase; the component
string is part of the model's identity, not a build detail. Every results table
must state which phase produced it — this is the sort of provenance that gets lost between
notebook and manuscript and then costs a revision round.

**Use a log-odds, not a raw likelihood:**

```
γ_A(r) = log P(r | path) - log P(r | best competing placement)
```
with the denominator cached at index time. Without it, term A degenerates into counting reads
and every particle is rewarded for walking into high-coverage regions regardless of
correctness.

Under fixed-alignment scoring the denominator has a closed form worth knowing: for a read drawn
from A, `LLR(A,B)` is a sum of independent per-column terms, so `E[LLR]` is the summed KL
between the two emission distributions and `Var[LLR]` the summed second moment — no read
simulation needed. `_llr_moments` / `build_mismapping_matrix` in
`superresolution-amplicon/bin/subspecies_infer.py` use exactly this to get a mis-mapping matrix
analytically. Same trick applies here to precompute a rival placement's expected score at index
time.

Only reads anchored in the newly-added segment contribute at step *t*.

### 2.6 Term B — paired-end congruence

**Terminology — pin this down before writing any code.** D10 specifies ~100 bp between the
mates and a fragment of ~400 ± 100 bp with 2×150 reads. Those are consistent, but the field
uses "insert size" for the **fragment length** (400), not the inner distance (100). Mixed
usage in a codebase silently produces a model that is wrong by 2×read_len.

**Convention for this project:** `f_insert` is a distribution over **fragment length = outer
distance**, i.e. from the 5′ end of mate 1 to the 5′ end of mate 2 along the path. Inner
distance is derived, never stored. Reason: inner distance goes negative when mates overlap,
which breaks the survival function; fragment length is the physically generated quantity and
is always positive.

Two contributions:

- **Resolved pairs:** when mate 2 becomes reachable at fragment distance `d`,
  add `log f_insert(d)`.
- **Censored pairs:** for pairs with mate 1 placed and mate 2 unreached after `d` bases,
  add `log S_insert(d)` (survival function). Without this the model only rewards pair closure
  and never penalises a path that walks away from its obligations.
- Retire pairs from the open set once `d` exceeds the fragment support, or particle state
  grows without bound.

**Two consequences of 400 ± 100 with 2×150 that the agent should expect and not treat as
bugs:**

1. Roughly 15% of fragments are shorter than 300 bp, so those mates overlap. Overlapping pairs
   contribute nothing to bridging but *do* contribute to the error model (overlap consensus).
   Handle them as a separate case rather than letting them produce negative gaps.
2. The effective bridging reach beyond the reads is only ~100–250 bp. For 16S that is
   adequate — conserved stretches between hypervariable regions are shorter than that — but
   it means term B cannot span longer repeats, and any claim about long-range phasing rests on
   chained coverage consistency, not on paired ends. Say so rather than implying otherwise.

`f_insert` estimated empirically from uniquely-placed pairs during indexing; the D10 values
are the prior/fallback, not a substitute for measurement.

**Two-sided paths (§2.1) change when this term switches on, and add one bookkeeping rule.**

- A pair straddling the seed resolves within roughly one fragment of step 0 *in either
  direction*, instead of only forward. That is the whole reason the cold start is short, and it
  is why seeding both mates of a pair is unnecessary (§7).
- Fragment distance `d` is measured along the path, across the seed where required: a pair with
  mate 1 left of the seed and mate 2 right of it is a single `d`, not two.
- **Score each read exactly once.** For the first few steps the left and right windows both
  cover the seed unitig. Rule: the seed's own bases are scored once, in `init`'s log increment
  (which exists for exactly this shape of reason — M1 finding 2); every read thereafter is
  charged to whichever frontier first extends past it, and a read already charged is never
  rescored when the other frontier reaches it.

### 2.7 Term C — coverage (local only in v0)

Observed unitig depth is a *sum* over every organism traversing it. "Constant coverage depth"
conflates a local self-consistency term (incremental, fine) with a global deconvolution
(couples all paths, not incrementally decomposable — what BayesPaths does with VB). Both at
once double-counts abundant taxa and starves low-abundance ones.

v0: carry a latent abundance `λ` in particle state. Per-base depth `y_i ~ NegBin(λ, φ)`, `λ`
under a Gamma prior, integrated out or maintained as a running posterior. Score each
newly-added unitig's depth under the predictive distribution. At a branch this is the
discriminative signal. Include a coverage-change hazard so genuine changes (repeats, chimeric
junctions) are possible but penalised — otherwise one bad estimate kills a path permanently.

v1 (deferred, §7): outer loop — sample a path set, fit abundances by NNLS/EM against observed
unitig depths, subtract explained coverage, resample against the residual.

**As built at M4 item 1 — read this before implementing the paragraph above literally.** Three
corrections, each measured; the full write-ups are under §5 M4 item 1.

- **Units.** Not per-base depth: k-mer *counts* `y = cov_kmer · (L−k+1)` against a k-mer-span
  exposure. The `L/(L−k+1)` conversion §5 M3 specified is wrong (finding 1), and counts avoid
  needing any conversion.
- **Log-odds, not raw likelihood.** `γ` is the Bayes factor against the same count under the
  prior alone. As a raw `log p(y)` the term is a flat toll on extending and collapses the
  posterior onto single-unitig paths (finding 2). §2.5 states this rule for reads; it is not
  specific to reads.
- **It does not prevent looping — as specified it rewards it** (finding 6), and the fix is
  §2.4's open multiplicity question. Nothing in this section as written models copy number,
  and the built term cannot tell a 2× repeat from a 4× one.

The prediction in this section's first paragraph — that conflating the local and global terms
"starves low-abundance ones" — held even with only the local term: on L1 the 1× organism got no
posterior mass at its branch points at all (finding 5). Coverage is an abundance signal, so it
needs a per-read term beside it.

### 2.8 Termination

Per-base continuation probability `(1-ρ)` in the prior, plus forced STOP at dead ends. Makes
variable-length paths comparable under one normalised model; without it longer paths
mechanically score lower and the sampler collapses to fragments.

**Not** a likelihood-drop threshold — data-dependent stopping breaks the target distribution
and forfeits the rigour requirement.

`--target-length L` supported as *conditioning* (reweight/reject paths not reaching `L`) for
benchmark comparability only.

---

## 3. Inference

### 3.1 SMC primary

Path construction is sequential, so SMC fits without reversible-jump proposals over a
variable-dimension space, and it yields `log Ẑ` (unbiased for `Z`) — the quantity that makes
downstream statistics defensible. MCMC returns as rejuvenation *inside* SMC (§3.3); that is
the useful reading of "MCMC particles".

### 3.2 Fully-adapted SMC step

```
side = alternate(t)                     # deterministic; skip a stopped side
alts = out_edges(frontier[side]) + [STOP_side]
for each a in alts:  compute γ(a) exactly
w_total = logsumexp_a γ(a)
sample a ~ softmax(γ)
log_weight += w_total
```

Rao-Blackwellises the branch choice; large variance reduction versus blind branching, at
`outdegree` (typically ≤ 4) likelihood evaluations per step. Implement from the start.

**The alternative set is per-side and includes that side's STOP** — M2's finding 3 (STOP inside
the `logsumexp`, not a separate case) applies once per side, not once per step. `STOP_side`
stops only that frontier; the particle is absorbed when both are stopped. Two rules the
enumerator and the sampler must agree on *exactly*, since the last divergence between them was
of this kind:

- **Dead ends.** Zero out-edges ⇒ that side's alternative set is `[STOP_side]` alone, so its
  γ is forced with weight 0 — not a special case outside the `logsumexp`.
- **Truncation.** At `max_bases`, drop the over-long extension but still score `STOP_side` by
  *graph* out-degree, as the one-sided version already requires. The budget is over total
  path length, so it is reached mid-alternation and the two sides do not hit it symmetrically.

Optional deterministic expansion at outdegree ≤ 2 with resampling doing the pruning
(beam-search regime, correct weights), gated on a particle-budget cap.

### 3.3 Resampling, degeneracy, exploration

Resampling causes **path coalescence**: after enough steps all surviving particles share an
ancestor, so effective sample size *at early loci* collapses to 1 while current-step ESS looks
healthy. That is exactly the failure mode that destroys long-range phasing.

1. **Adaptive resampling** at `ESS < N/2`. Systematic or stratified, never multinomial.
2. **Track full ancestry; report per-locus ESS** as a first-class output. If early-locus ESS
   is 1 by the end of a run, the long-range claims are worthless and you need to see it.
3. **Rejuvenation moves.** After resampling, MH moves re-routing the last `L` bases (bubble
   swap, branch flip). Symmetric proposal for a simple bubble swap ⇒ acceptance ratio is the
   likelihood ratio. The practical fix for coalescence.
4. **Island model.** `M` islands × `N` particles, no cross-island resampling. Near-linear
   parallel scaling plus between-island `log Ẑ` spread as a convergence diagnostic.
5. **Stratified starts** weighted by coverage mass, so low-abundance organisms get budget.
   Subsumed by read-sampled seeding (§2.3) — sampling the seed from the reads *is* coverage-mass
   weighting. What remains here is stratification proper: over-sample low-coverage seeds and
   correct with an importance weight, so a 10²-fold abundance range (R2) does not spend the
   whole budget on the dominant organism.

**No taboo/visited-node lists.** They make the target history-dependent, so it stops being a
probability measure. Diversity comes from stratification, islands, and tempering. If genuinely
repulsive sampling is wanted later, the principled formulation is a determinantal point
process over path sets — future work, not improvisation.

### 3.4 Output

- Weighted sequences, deduplicated with **summed** normalised weights.
- `log Ẑ` with per-island spread.
- Per-locus ESS trace.
- Per-branch marginal probabilities — likely the most immediately useful output for downstream
  statistics, more than the sequences themselves.

Document that self-normalised importance weights are **consistent but biased**, `O(1/N)`.
Report ESS with every estimate; use inter-island variance rather than asserting convergence.

---

## 4. Architecture — Python first (D1)

```
minoodle/
  minoodle/
    graph.py          # metaSPAdes GFA parse, bidirected graph, CSR arrays (numpy)
    index.py          # kmer -> (unitig, offset, strand); read anchors; q_seed weights
    model/
      errors.py       # skiver emission table + read scoring (§2.5)
      insert.py
      coverage.py
      composite.py
    sampler.py        # SMC engine
    exact.py          # toy graphs, §2.3 prior, seed proposals, brute-force enumerator
    diagnostics.py    # ESS, ancestry, calibration
    cli.py
  tests/
  fixtures/           # golden fixtures — the contract for the Rust port
  bench/
```

Stack: numpy for anything array-shaped, **numba** `@njit` for the read-scoring inner loop (and a pair-HMM forward, if §2.5
level 2 is ever needed) and the
inner SMC weight loop, pytest + hypothesis, pyarrow for output. No pandas in hot paths.

Chopin's `particles` is a poor fit for a custom discrete state space with fully-adapted steps
and ancestry tracking — write the ~200-line loop directly, but lift its resampling routines
(systematic/stratified), which are well-tested.

### 4.1 ABCs — define in M0, mirror the eventual Rust traits exactly

```python
class TokenSpace(ABC):
    """Base-space k-mers and minimizer tuples both implement this."""
    @abstractmethod
    def tokenize(self, seq: bytes) -> list[tuple[int, int]]: ...   # (token, offset)
    @property
    @abstractmethod
    def k(self) -> int: ...

class PathGraph(ABC):
    @abstractmethod
    def nodes(self) -> list[OrientedNode]: ...
    @abstractmethod
    def out_edges(self, n: OrientedNode) -> np.ndarray: ...
    @abstractmethod
    def unitig_seq(self, n: OrientedNode) -> bytes: ...
    @abstractmethod
    def unitig_depth(self, n: OrientedNode) -> np.ndarray: ...
    def seeds(self) -> list[Seed]: ...          # every oriented k-mer position (p_seed support)

class IncrementalLikelihood(ABC):
    """The bounded-window contract from §2.4. If a term can't fit this, it's not in v0."""
    @abstractmethod
    def init(self, seed: Seed) -> tuple[State, float]: ...   # seed unitig's bases, scored once
    @abstractmethod
    def extend(self, st: State, e: Edge, side: Side) -> tuple[State, float]: ...
    @abstractmethod
    def stop_logp(self, st: State, side: Side) -> float: ...
```

Amended at M1 (`init` returns its increment — the seed unitig's own bases are not free) and at
M3.5 (`Seed` and per-`Side` extension, §2.1). `Side.LEFT`'s frontier is an ordinary *forward*
walk from `seed.node.flipped()`, so `out_edges`/`new_bases` need no mirrored variants.

Composite likelihood = a list of `IncrementalLikelihood` summing increments. Free ablations by
composition.

### 4.2 Port contract (Python as oracle for Rust)

Cross-language RNG streams will not match, so:

- **Do not** attempt to match sampled trajectories via seeds. numpy PCG64 and Rust `rand_pcg`
  differ from the same seed.
- **Inject the RNG.** The SMC engine in both languages takes a `uniforms()` source. Tests feed
  a *recorded stream of uniforms* generated once in Python and stored in `fixtures/`. Both
  implementations then produce byte-identical trajectories, ancestries, and weights.
- **Golden fixtures** for everything deterministic: read congruence scores, incremental
  log-weights for a fixed (graph, particle) set, resampling indices from a fixed weight vector,
  `log Ẑ` on the M1 toy graphs. Tolerance **1e-9** in log space.
- Fixtures as `.npz` + JSON manifest with a schema version. Frozen at M6; thereafter the
  Python implementation is the spec and changes require a fixture bump.
- Do not port before M6.

---

## 5. Milestones

Hard gates. Do not proceed past a failure.

**Order note (rev 7).** M5 is split. The synthetic ladder (**M5a**, §5.5.1) runs on the
Python implementation; the real-data ladder (**M5b**, §5.5.2) does **not**. Optimisation —
M6 (Python ceiling + fixture freeze) and M7 (Rust port) — sits between them:

```
M4 → M5a (synthetic ladder) → M6 → M7 → M5b (real-data ladder) → M8? → M9
```

Reason: R1–R4 are HPC-sized and each iteration costs a queue wait, so entering them on the
slow implementation buys nothing but wall clock. M8 (minimizer space) stays after M5b and is
still to be avoided — reach for it only if M7's measured throughput leaves M5b infeasible,
and record the measurement that forced it. §5.5.4 is the HPC discipline that keeps the
round-trips down; read it before writing any M5b code.

### M0 — Skeleton and data (0.5–1 day) — **DONE**
- Package, CI, ruff/mypy, pytest, bench harness.
- ABCs from §4.1 with stubs.
- Simulation adapters: reproducible data from genome-blender + skiver with a manifest
  recording seeds, error parameters, fragment distribution (mean 400, sd 100, 2×150), and
  ground-truth sequences.
- Error-model phase tagging (P1/P2/P3 from §2.5) wired into the config from the start.

**Gate:** tests green; data regenerable from the manifest. **Passed.**

**What shipped.** `pyproject.toml` (uv, ruff, pytest); `minoodle/interfaces.py` — the §4.1 ABCs
plus `OrientedNode` and `CompositeLikelihood`, the latter being the "free ablations by
composition" mechanism and the only non-stub logic; `minoodle/simdata.py` — `ErrorModelPhase`,
`Manifest`, `run`, `verify`; `datasets/L0.yaml` with its reference FASTA committed under
`datasets/refs/`; tests for composite summation and manifest tamper-detection.

**Deferred, deliberately.** CI workflow and mypy → M2, when there is a numerical gate worth
protecting. `bench/` → M6, per that milestone's own instruction to profile before optimising.
The §4 module layout is created per milestone rather than scaffolded empty now.

**Conventions established, which later milestones depend on:**

- Generated data lives outside the repo at `~/Documents/minoodle_run/<dataset>/`, matching the
  existing `genome-blender_run/` convention. The repo holds configs and adapter code only.
- genome-blender is invoked as a configured shell command (`generate_reads_cmd` in the dataset
  YAML, pointing at the `genome_blender_dev` conda env), not imported. The two projects share
  no environment.
- The manifest records the genome-blender config **verbatim** rather than by path — the config
  is the reproduction recipe — alongside both repos' git SHAs and sha256 of every reference and
  output.

**Three findings from the M0 run, recorded because they bit once already:**

1. **Hash content, not container.** The first reproducibility check showed differing FASTQ
   hashes between two identically-seeded runs. The cause was gzip's embedded header mtime; the
   decompressed reads were byte-identical. `FileRecord` now hashes `.gz` files decompressed.
   Any future output format that embeds a timestamp needs the same treatment, or the M2
   reproducibility claims will fail spuriously.
2. **genome-blender is deterministic under a fixed seed** (verified at `6e1efe0`, two runs, all
   three outputs identical once (1) was handled). The contingency noted for a
   non-deterministic simulator was not needed.
3. **`num_reads` is total reads, not pairs.** `num_reads: 20000` with `paired_end: true` yields
   10000 fragments / 10000 pairs. Coverage arithmetic for L1–L3 and the §5.5.2a expected-coverage
   partition must use this convention. Current-version genome-blender also gzips its FASTQ where
   older run directories hold plain `.fastq`; the adapter accepts either.

### M1 — Exact enumerator (1 day) — *before the sampler* — **DONE**
Enumerate all paths up to a length bound in tiny hand-built graphs (≤ 20 nodes, ≤ 1e5 paths);
compute exact `π` by brute force. Graphs: (a) linear chain, (b) one bubble, (c) nested bubbles,
(d) a repeat visited twice.

**Gate:** exact posteriors stored as fixtures. **Passed.**

Do not skip. Without a ground-truth posterior you cannot distinguish "sampler works" from
"sampler produces plausible-looking sequences", and every rigour claim rests on that.

**What shipped.** `minoodle/exact.py`: `ToyGraph` (a `PathGraph` whose every forward edge
installs its bidirected twin on construction, so orientation cannot be got wrong in a fixture),
a `build()` helper that makes edge sequences satisfy the k−1 overlap by construction (union-find
over unitig ports, one shared joint string per equivalence class), the four graphs at k=5, the
§2.3 prior, `GCBias` (a fixture-only likelihood term — the real terms are M4), `enumerate_paths`,
and `.npz` + `manifest.json` fixture I/O with a 1e-9 re-enumeration check (§4.2). Fixtures are
committed under `fixtures/` but **not frozen**; freezing is M6.

**Three findings, each of which changes something later:**

1. **The §2.3 formula as written is not a probability measure.**
   `p_start · Π_t [(1-ρ)^{len_t} · p_edge] · ρ` survives the last unitig's bases *and then* stops
   with probability ρ, which does not sum to 1 over paths. Implemented instead is the per-base
   geometric that §2.3's own prose specifies: the path ends when the stop lands *within* the last
   unitig's new bases, terminal factor `1 - (1-ρ)^{len_T}`, and 1 at a dead end where STOP is
   forced (§2.8). That sums to exactly 1 on an acyclic graph, and asserting it is the test that
   catches essentially every length-accounting bug. M2's `log Ẑ` comparison depends on both sides
   using this convention.
2. **`IncrementalLikelihood.init` needed a log increment.** The §4.1 sketch returns only a
   `State`, which leaves the start unitig's own bases unscored — free likelihood for the first
   node. Amended to `init(start) -> (state, float)`, mirroring `extend`; the SMC engine needs the
   same number as a particle's initial log-weight. The Rust traits at M7 follow the amended form.
3. **The prior is directional; only the likelihood is RC-symmetric.** A path and its reverse
   complement are different states with different start nodes, out-degrees and terminal unitigs,
   so their *priors* legitimately differ. §5 M3's property test must therefore be stated over `L`,
   not over `π`. Tested here on all four graphs.

**Deliberately not done:** per-branch marginals (§3.4) — a few lines off the enumeration, added
at M2 when there is a consumer.

### M2 — SMC validated against exact (2–3 days) — **DONE**
Fully-adapted step, systematic resampling, ancestry tracking, island model.

**Gate:** TV distance < 0.01 at N = 1e5 on all four toy graphs; `log Ẑ` within Monte Carlo
error of exact `log Z`; SBC rank statistics uniform. If not met the bug is in weight
bookkeeping — fix it, don't add features. **Passed, with the TV criterion restated — see
finding 1.**

**What shipped.** `minoodle/sampler.py`: the fully-adapted step of §3.2 with STOP as one of the
alternatives, an equally fully-adapted start step, adaptive systematic resampling at `ESS < N/2`,
an ancestry arena (`(T, N)` node codes plus per-step parent permutations, no copied path lists),
the island model, deduplicated weighted output with summed weights, per-branch marginals, and a
`validate` subcommand that reruns the gate against `fixtures/`. `minoodle/diagnostics.py`: ESS,
systematic resampling, lineage reconstruction, per-locus ESS, TV, the iid TV reference, and the
calibration rank machinery. `Enumeration.posterior()` and `branch_marginals()` close M1's
deferred item. CI added (`.github/workflows/ci.yml`: ruff, pytest, `exact verify`, and the full
N = 1e5 gate, which runs in ~5 s). mypy still deferred — the type surface moves again at M3/M4.

**Findings:**

1. **`TV < 0.01` at N = 1e5 is unattainable on `repeat_twice` by any sampler.** With 242 atoms,
   an *exact iid* sampler of 1e5 draws averages TV 0.0104 against π (0.0123 at the 99th
   percentile); the threshold is a property of the atom count, not of the sampler. The gate is
   therefore stated as *TV within the band an exact iid sampler of the same achieved ESS would
   incur* (`multinomial_tv_reference`), which is the criterion the flat number was reaching for.
   Measured: TV/iid-p99 = 0.48, 0.60, 0.88, 0.98 on the four graphs, i.e. the sampler is
   statistically indistinguishable from exact iid sampling at its own ESS. The absolute number is
   still printed, and still met on the other three graphs.
2. **Adaptive resampling never triggers on the toy graphs.** The fully-adapted step is effective
   enough that ESS stays at 0.84–0.98 N to the end of every run, so `ESS < N/2` never fires and
   the resampling, state-permutation and ancestry-permutation paths are dead in a default run —
   they would have shipped untested. They are exercised deliberately with `ess_frac = 1.0`, which
   must leave `log Ẑ` and TV unchanged. Expect the same blind spot at M3–M4: any new code that
   only runs under degeneracy needs forcing, not waiting for real data.
3. **STOP has to be inside the `logsumexp`, not a separate case.** §3.2's pseudocode enumerates
   out-edges only. A path may end at any node — `enumerate_paths` emits a record at every node
   visited — so omitting STOP from the fully-adapted alternative set targets a different measure.
   The test that catches it instantly is the sampler's analogue of M1's `Σ p(x) == 1`: with no
   likelihood, every step's `logsumexp` is the prior's total continuation mass, so `log Ẑ` is
   *exactly* 0.0 with no Monte Carlo error at all. That, plus truncation matching the enumerator's
   (drop the over-long extension, but still score STOP by *graph* out-degree), is the whole of the
   weight bookkeeping the gate warns about.
4. **SBC is degenerate until there is a generative likelihood.** `GCBias` is a bare potential:
   there is no data to draw, so the prior → data → posterior loop does not exist yet. Implemented
   is what SBC reduces to with the data integrated out — an exact draw from π is exchangeable with
   the sampler's draws, so its randomised rank among them is uniform (KS p = 0.56 over 300
   replicates). Ranks must be randomised: every statistic on a 20-path graph is heavily tied and
   plain `#{v < x}` is not uniform under the null. Proper SBC arrives with M4's error model.

**Deliberately not done:** rejuvenation MH moves (§3.3 item 3) and stratified starts (item 5) —
neither is in this milestone's deliverable line, and per finding 2 there is no degeneracy on
these graphs for them to fix. Add when a real graph shows early-locus ESS collapsing. Recorded
uniform streams are consumed through the injected seam and the draw count is asserted, but no
stream is committed as a fixture yet; that is M6's freeze (§4.2).

### M3 — metaSPAdes GFA and index (2–3 days) — **DONE**

Per D2/D7/D8, run metaSPAdes as: `metaspades.py --only-assembler -k 21 -1 R1.fq -2 R2.fq`,
in a container rather than from a local install (D17):

```bash
docker run --rm --platform linux/amd64 -v "$PWD:/data" \
  quay.io/biocontainers/spades:4.3.0--hde4eca7_0 \
  metaspades.py --only-assembler -k 21 \
  -1 /data/sim_reads_R1.fastq.gz -2 /data/sim_reads_R2.fastq.gz -o /data/asm
```

Consequences to handle explicitly:

- **`--only-assembler` skips BayesHammer**, which is what we want: the §2.5 likelihood scores
  raw reads, and pre-corrected reads would have had some of the errors the skiver model exists
  to describe already removed. Record in the run config that correction was skipped.
- **Single-k with `-k 21`** was expected to give GFA overlaps at k−1 = 20. It does not — see
  finding 1: the overlaps are 21M, so this project's k is 22. metaSPAdes accepted single-k in
  metagenomic mode, so the Bifrost/Cuttlefish fallback was not needed.
- **metaSPAdes still simplifies the graph** — tip clipping, bubble removal, chimera filtering —
  even with `--only-assembler`, which only skips read correction. That removes some of the
  uncertainty this project exists to sample. Document as a known limitation; if results look
  suspiciously clean, this is the first thing to check.
- **Coverage units:** metaSPAdes' `cov` in the segment name and the `KC:i:` tag are k-mer
  based, not base coverage. Convert before use — but with `R/(R-k+1)` in the *read* length, not
  the `L/(L-k+1)` written here originally; see M4 item 1 finding 1, which measures it.
- **Bidirected orientation handling** (`+`/`-`, reverse complement) is the most common source of
  silent bugs. Property test: every path and its reverse complement must receive identical
  likelihood.

Then: k-mer index (dict → later minimal perfect hash), read anchor table with cached
best-competing-placement scores, fragment-length estimation, per-unitig depth arrays.

**Gate:** graph loads; RC round-trip passes; anchor recall ≥ 99% on error-free reads.
**Passed:** 156 unitigs / 222 links load from L0, RC round-trip OK, **recall 100.00%** on all
20 000 L0 reads (99.77% of them uniquely), in 1.5 s.

**What shipped.** `minoodle/graph.py`: `UnitigGraph` (CSR `indptr`/`indices` over oriented-node
codes, same method surface as `ToyGraph` so the sampler and enumerator run against a real graph
unchanged), `from_gfa`/`to_gfa`, the k-mer→base coverage conversion, `rc_roundtrip_ok`, and a
`stats` subcommand. `revcomp`/`code`/`decode` moved here from `exact.py`, and `new_bases`/
`path_seq`/`k` moved *up* into the `PathGraph` ABC, where both implementations shared them
verbatim. `minoodle/index.py`: `KmerIndex` (canonical k-mer → oriented placements, occurrence
cap), diagonal-voting `anchor`, `fragment_length`, and the `check` subcommand that is the gate.
Assembly provenance (image digest, command, GFA hashes) is written next to the GFA as
`asm/provenance.json`.

```bash
uv run python -m minoodle.graph stats <asm>/assembly_graph_with_scaffolds.gfa
uv run python -m minoodle.index check <asm>/assembly_graph_with_scaffolds.gfa R1.fq.gz R2.fq.gz
```

**Findings:**

1. **`metaspades.py -k 21` writes `21M` overlaps, so k = 22 here.** SPAdes' unitigs are paths
   of (k+1)-mers; consecutive segments share 21 bases, not 20. Every downstream use of "k−1
   overlap" therefore means 22−1. `from_gfa` reads k off the `L` lines and *asserts* against
   any k passed in rather than trusting it — passing `--k 21` fails loudly, which is how this
   was found. The same arithmetic sets the coverage conversion: `KC/DP` is exactly `L−k+1`
   with k = 22, which independently confirms it.
2. **The coverage conversion blows up on short unitigs.** `cov_base = cov_kmer · L/(L−k+1)` is
   a factor of ~2.9 at the median unitig length (32 bp, k = 22) and unbounded as `L → k`.
   Implemented as specified, but §2.7's NegBin will see wildly over-dispersed depths on short
   unitigs unless M4 either weights by k-mer span or works in k-mer coverage directly. Decide
   at M4 item 1, with the number in front of you; do not quietly change the formula here.
   **Resolved at M4 item 1** (finding 1): the formula was wrong, not merely awkward — the
   conversion is the constant `R/(R−k+1)`, and §2.7 works in k-mer counts and never applies it.
3. **The L0 graph is not a single path** — 156 unitigs, 222 links, 122 of 312 oriented nodes
   with out-degree > 1, from one 100 kb genome with error-free reads. k = 22 collapses every
   repeat longer than 21 bp, so even the "trivial" rung has branching for M4's terms to
   discriminate. Total segment length 101 330 vs a 100 260 bp genome, as expected once shared
   overlaps are counted twice.
4. **Fragment length measures 397.5 (var 9 658) against a planted 400 (var 10 000)** on 8 907
   FR pairs — the estimator is right, with the mild downward bias you would predict from
   measuring only pairs that fit inside one unitig. §2.6's "estimate empirically, the D10
   numbers are a prior" is therefore doing real work rather than confirming an assumption.
5. **No twin `L` lines.** metaSPAdes writes each bidirected edge once; the loader installs the
   twin `(v,¬ov) → (u,¬ou)` itself and deduplicates, so a GFA that *does* write both parses
   identically. `rc_roundtrip_ok` over every oriented node is the check.

**Deliberately not done:** the *score* of the best competing placement (§2.5) — that needs the
§2.5 scorer, so M3 caches the rival placement's identity and vote count and M4 fills in the
score.
Minimal perfect hash (the plan already defers it), per-base depth from a pileup (GFA carries one
number per segment; revisit only if §2.7 wants it), and numba (M6).

### M3.5 — Two-sided paths: redo the M1 and M2 gates ✅ *(done, rev 11)*

Added at rev 10. §2.1 changes the state space, so the exact enumerator and the sampler are both
invalidated. Doing this before M4 is the point: building six likelihood terms against a state
space that is about to change means writing each of them twice.

1. `exact.py`: seed enumeration, two-sided `enumerate_paths`, the §2.3 two-geometric prior.
   `Σ p(x) == 1` now sums over `(seed, path)` — that test is the whole gate for the prior.
2. Regenerate the golden fixtures. Atom counts rise by roughly the seed multiplicity, so the
   toy graphs may need shrinking to stay enumerable; prefer shrinking the graphs to loosening
   the gate.
3. `sampler.py`: alternation, per-side STOP inside the `logsumexp`, per-side dead ends and the
   shared `max_bases` budget (§3.2). Prior-only `log Ẑ == 0.0` exactly under uniform seeding
   (`q_seed = p_seed`) — still the sharpest test, now with a stated configuration.
4. Seeding as a proposal (D18, resolved): normalised `q_seed` from the anchor counts, the
   `ε`-mixture for full support, and `log p_seed - log q_seed` in the initial log-weight.
5. Two new tests that the one-sided formulation could not express:
   - **RC symmetry.** A seed and its RC give mirror path sets with equal prior mass. This is
     the check that catches an asymmetric alternation rule.
   - **Proposal invariance.** Uniform seeding and coverage seeding agree on the posterior
     within Monte Carlo error, and on `log Ẑ` within its band. This is what catches a
     mis-normalised or partial-support `q_seed`, and neither of the inherited gates would.

**Gate:** M1's `Σ p(x) == 1` and M2's TV gate (against the exact-iid reference band, per M2
finding 1) both pass on the two-sided formulation, plus RC symmetry and proposal invariance.
Do not start M4 on a failing M3.5.

**Watch for:** the M2 blind spot repeating — adaptive resampling never fired on the toy graphs,
so anything that only runs under degeneracy is covered only by tests that force it. Two-sided
paths add their own such path (one side stopped, one open); test it explicitly rather than
hoping a toy graph produces it.

**Shipped:** `Side`/`Seed` and the two-sided `IncrementalLikelihood` in `interfaces.py`;
`seeds()` on `PathGraph` and `seed_path_seq` in `graph.py`; `seeds`/`next_side`/`mirror`/
`SeedProposal` and the two-sided enumerator in `exact.py`; seed-proposal sampling, per-side
STOP and `(nodes, sides)` ancestry in `sampler.py`; `index.seed_weights`. Fixtures
regenerated at schema version 2. All four gate lines pass at N = 1e5 (`repeat_twice` TV
0.0709 against an iid p99 of 0.0680), plus RC symmetry exactly and proposal invariance.

**Findings.**

1. **`ρ` is 0.04 now, not 0.02.** Total length is a sum of two geometrics, so mean length is
   `2/ρ`. 0.04 keeps expected total length where the one-sided fixtures had it. Any `ρ` copied
   from an M1/M2 artefact is wrong by 2× in length.
2. **The mirror map is `(seed.flipped(), right, left)` — the walks are untouched.** That falls
   out of representing the left frontier as an ordinary *forward* walk from
   `seed.node.flipped()`, which also means no per-side variants of `out_edges` or `new_bases`
   are needed anywhere. Any other representation makes RC symmetry a case analysis instead of
   a one-liner; do not change it.
3. **`ε` caps the importance weight at `1/ε`.** A seed with no anchored read gets `q = ε/n`
   against `p = 1/n`. That, not "small but not zero", is the number to tune `ε` by: at
   `ε = 1e-3` a single particle on an unread k-mer carries 1000× weight and dominates `Ẑ`.
   Default is 0.01; the validation run uses 0.05.
4. **Proposal invariance needs a looser TV band than the M2 gate, and it was calibrated
   rather than guessed.** A bad proposal is *supposed* to be less efficient than iid, and the
   ESS the band is built from is measured after resampling, so it overstates surviving seed
   diversity. Measured against the failure it exists to catch — dropping `log p - log q` — the
   correct run gives TV 0.03 and the broken one 0.29, so the gate sits at 3× the iid p99.
   Note `log Ẑ` does *not* discriminate here: the broken run's was off by only 0.007.
5. **`repeat_twice`'s budget is deliberately tight (120 bases, 2e-3 of mass truncated).** Two
   competing pressures resolve the same way: fewer atoms, and truncation loss large enough to
   *see*. At `max_bases = 250` the lost mass is 8.5e-8, which would hide any enumerator/sampler
   divergence in the truncation rule — the exact rule those two have historically disagreed on.

**Deferred:** a `prior.py` split (the seed/prior machinery still lives in `exact.py`, which
`sampler.py` already imported the prior from — revisit if M4 makes that module unwieldy) and
the `max_bases` story for real graphs, where a unitig can be longer than the budget and
`run_island` currently raises rather than choosing a policy. *(The budget became optional at
M4 item 1; the seed guard still raises, deliberately — see that item's finding 3 for why a
real run wants a budget anyway, and why it has to exceed the longest unitig.)*

### M4 — Likelihood terms, one at a time (4–5 days) — *item 1 built, gate failing, rev 12*
Order, with an ablation after each:
1. Coverage (§2.7) — should recover the correct branch in an unbalanced bubble.
2. Single-end congruence (§2.5), phase P1: fixed-alignment scoring against the skiver emission
   table, plus a `gap_penalty` sensitivity sweep. Escalate to the pair-HMM (§2.5 level 2) only
   if the ablation says fixed alignment is losing information.
3. Paired-end, resolved (§2.6). Include the seed-straddling case (mate 1 left of the seed,
   mate 2 right of it, one `d` across the seed) and the score-once rule — both are new at
   rev 10 and neither appears on a one-sided path.
4. Paired-end, censored — check it *reduces* spurious long paths.
5. Stop model (§2.8) — output length distribution must match the prior when data are
   uninformative.
6. Repeat 2 under phase P2 and record the degradation. That delta is the misspecification
   sensitivity, and it is a reportable result in its own right.

**Gate:** each term individually raises the posterior probability of the ground-truth sequence
on a two-species synthetic set. If a term doesn't, report it — a term that doesn't help is a
finding, not a tuning target.

#### M4 item 1 — coverage ⚠️ *(implemented and toy-validated; the L1 gate FAILS, rev 12)*

**Gate status.** The term does exactly what §2.7 asks on a toy graph where the exact posterior
is available: on the unbalanced bubble it prefers the depth-8 arm over the depth-2 arm by 157:1
against a prior that is 1:1, and the sampler reproduces the enumerator's target with the term
on. On **L1 it does not produce a usable posterior at any hazard tested**, and finding 6 says
why: at a repeat the term *pays* for looping rather than preventing it, so any cycle is a
runaway. Do not read item 1 as passed, and do not tune the hazard to make it pass — the defect
is structural, and findings 6 and 3 say where. Item 2 can proceed (it is a per-read term, and
finding 5 says that is exactly what coverage needs beside it), but **M4's milestone gate is not
met until the multiplicity question in finding 6 is answered against §2.4.**

**What shipped.** `minoodle/model/coverage.py`: `CoverageTerm`, a per-side Gamma–Poisson with
§2.7's change hazard; `NoLikelihood` (the prior-only ablation arm); `truth_edges` /
`true_edge_mass` and an `ablate` subcommand. `index.read_fasta` and `index.reference_walk`
supply ground truth on a real graph by running a reference's k-mers back through the k-mer
index — no aligner. `PathGraph.unitig_kmer_cov` was added (finding 1) and `SMCConfig.max_bases`
became optional, closing M3.5's deferred "what does the budget mean on a real graph" item.
`datasets/L1.yaml` is the two-species rung the gate needs: Prevotella and a Clostridia contig
at 10:1, 60 000 reads, assembled with the same container and command as L0.

```bash
uv run python -m minoodle.simdata run datasets/L1.yaml --out ~/Documents/minoodle_run/L1
uv run python -m minoodle.model.coverage ablate ~/Documents/minoodle_run/L1 --hazard 0.05 0.2
```

**Findings.**

1. **The k-mer→base coverage conversion is a constant in the read length, not `L/(L−k+1)`**
   — this closes M3 finding 2, which deferred the decision to here. On L0 the span-weighted
   k-mer coverage of the unitigs over 1 kb is **25.73**, and the planted base depth
   (20 000 × 150 / 100 260 = 29.92) times `(R−k+1)/R` with `R` = 150 is **25.73**, to both
   decimals. A read of length `R` carries `R−k+1` k-mers over `R` bases; the unitig's own
   length never enters. The old formula was a factor of 22 out on the shortest unitigs, since
   L0's median unitig is 32 bp against k = 22. `graph.py` now applies `R/(R−k+1)` with
   `read_len` as an explicit knob (no assembler records it), and §2.7's term sidesteps the
   conversion entirely by working in **k-mer counts against a k-mer-span exposure**: a
   short unitig then carries *little* information rather than wildly overdispersed information,
   which is the truth about it.
2. **§2.5's log-odds rule is not specific to term A — term C needs it too.** Scored as a raw
   `log p(y)`, every unitig charges a large negative number, so the term is a flat toll on
   extending against a STOP that costs nothing: it stops arguing about *which* branch and
   starts arguing with the length prior. On L1 the posterior collapsed onto **single-unitig
   paths** (mean 1.0 unitigs/path) and the ablation had no branch left to score. Scoring the
   log-odds against the same count under the prior alone — the Bayes factor for "this unitig
   continues the abundance I have been tracking" — fixes it, and is exactly the correction
   §2.5 prescribes for reads. Expect to apply it to every remaining term.
3. **Coverage alone does not terminate, and that defeats the L1 gate.** With the log-odds in
   place the increment is unbounded *above*: measured on L1 it reaches **+37 nats** for one
   unitig, because Bayesian evidence scales with exposure and a long unitig matching the
   running abundance is overwhelming evidence. The geometric stop is `O(ρ)` per base and cannot
   compete, so on a self-consistent stretch the walk does not stop — §2.8's guarantee holds for
   the prior alone, not for the prior times this likelihood. Measured on L1 at `max_bases`
   40 000, `ρ = 1e-3`:

   | arm | `log Ẑ` | island spread | states | unitigs/path |
   |---|---|---|---|---|
   | prior only | −0.007 | 0.073 | 2474 | 6.45 |
   | hazard 0.05 | +46 567 | 21 952 | 694 | 50.7 |
   | hazard 0.2 | +75 237 | 26 813 | 271 | 8 595 |

   An island spread of 2e4 is not an estimate. **Particle count decides whether you see this:**
   the same `h = 0.05` arm at 400 particles gave 1.95 unitigs/path and a plausible-looking
   0.950 → 0.982 improvement in true-branch mass; at 4 000 particles a particle finds the
   runaway and dominates. Any small-N run of this term will look fine and be wrong — check
   `unitigs/path` and the island spread, not the branch metric.
   Not a bug to patch inside the term: item 4 (censored pairs) is the term the plan already
   nominates to *reduce* spurious long paths, and it is the right place to fix this. The
   alternative — bounding the per-unitig increment — is a change to the target and needs its
   own justification, not a constant chosen to make a gate pass.
   Real-graph runs also need an explicit `max_bases` exceeding the longest unitig (34 860 bp on
   L1) or the seed guard fires. Both arms run under the same budget, so the comparison is
   like-for-like as far as it goes.
4. **The hazard is the load-bearing knob on a real graph, not a nuisance parameter.** It floors
   the per-unitig penalty at `log h`, so it sets how much local depth inconsistency a path may
   absorb — and adjacent metaSPAdes unitigs differ in depth by 10× routinely, because a short
   repeat unitig carries the summed depth of every traversal (§2.7's own caveat, and the
   reason v0 is local self-consistency only). `hazard = 0.01` is far too aggressive; the
   ablation sweeps it. But it only moves *where* finding 3 bites, not whether: at 4 000
   particles `h = 0.05` and `h = 0.2` both run away. There is no hazard that rescues the term
   on L1, which is why finding 3 is a structural problem and not a calibration one.
5. **Coverage starves the rare organism.** In both coverage arms the 1× organism gets **no
   posterior mass at its branch points at all** — nothing to be accurate about. (The prior-only
   arm does reach it, with 0.026 of mass and all of that on true edges.) Evidence scales with
   counts, so a high-coverage stretch simply out-scores a low-coverage one; this is §2.5's
   warning about term A, arriving early and for the same reason, and it is the §3.3 failure
   mode L1 exists to expose. The honest reading is that coverage on its own is an *abundance*
   signal, so it needs a per-read term beside it rather than a tuning pass.
6. **The term does the opposite of preventing looping — it pays for it, and this is the root
   cause of finding 3.** §2.7's implicit job at a repeat is copy number: a unitig at 2× the
   flanking depth should be traversed twice, one at 1× once. Measured on `repeat_twice`
   (unitig 1 is the repeat; posterior over how many times a path traverses it):

   | repeat depth | prior | coverage |
   |---|---|---|
   | 1× flanks | mode 1, 0.58 | mode **6**, and 0.53 on ≥ 6 |
   | 2× flanks | mode 1, 0.58 | **0.89 on zero traversals** |
   | 4× flanks | mode 1, 0.58 | identical to 2× — no copy-number discrimination at all |

   The mechanism is self-confirmation: the running posterior absorbs the unitig's *own* count
   on first traversal, so re-scoring the same unitig is a match against evidence it supplied.
   Successive traversals of one unitig score +1.566, +1.672, +1.720, +1.748, +1.766 nats —
   each loop is cheaper than the last, without bound. A path that finds any cycle is paid to
   stay in it, which is exactly the L1 runaway.

   The fix is a multiplicity model, and it collides with the bounded window, so **it is parked
   in §2.4 as that section's open question** with the three candidate resolutions written out.
   Decide it there, not here, and not with a constant that bounds the increment.
   `test_repeat_traversal_is_self_confirming` pins the broken behaviour deliberately: when the
   fix lands, that test must be rewritten to assert the opposite, and its failure is the signal
   that it worked.
7. **`branch_marginals` is not a probability once paths are long.** It accumulates per
   traversal, so the runaway arms report total branch mass of 2 317 and 9 785 against the
   prior arm's 1.73. The per-genome fraction is still meaningful as a ratio, but any absolute
   reading of it is not, and a metric built on it cannot detect the degeneracy that produced
   it. Read `unitigs/path` and the island spread first — the M5a metric suite (§5.5.3) needs
   a degeneracy guard of its own, not just accuracy numbers.

**Deliberately not done:** per-base depth from a pileup (the term wants counts, not rates), a
global abundance deconvolution (v1, §7), and the full run-length change-point posterior — the
hazard collapse is BOCPD truncated to two atoms, which keeps §2.4's bounded state.

### M5a — Synthetic ladder and calibration (3–4 days)

Everything in §5.5.1, on the Python implementation, local. This is where the calibration
machinery of §5.5.3 is *built and debugged*, because a synthetic rung is cheap to rerun and a
real one is not. M5b reuses the same code and changes only the inputs.

**Gate:** calibration within tolerance on L0–L2; documented failure modes on L3; the metric
suite of §5.5.3 runs end-to-end from a manifest with no interactive steps (the §5.5.4
requirement). Then go to M6 — do not start any R-rung.

#### 5.5.1 Synthetic ladder (development)

16S at k=21 is a good stress test and a bad smoke test: nine hypervariable regions between
conserved stretches means near-identical paths everywhere. Debug on the lower rungs.

| Level | Data | Tests |
|---|---|---|
| L0 | one sequence, no errors | recovers it; `log Ẑ` correct |
| L1 | two divergent genomes, 10:1 abundance | low-abundance organism still sampled |
| L2 | 3–5 species' 16S, shared conserved regions | the real ambiguity test |
| L3 | 3 small genomes (~5 Mb) | performance, memory |

### M5b — Real-data ladder (4–5 days, mostly HPC wall clock) — *after M7*

Runs on the Rust implementation from M7, against the frozen M6 fixtures. Prerequisites:
M6 gate passed, M7 gate passed, §5.5.4 harness in place. Gate is at the end of §5.5.3.

The cheap, local, non-HPC parts of this milestone — accession resolution (§5.5.2c step 1),
reference-bundle diff (step 2), sylph triage (step 3), the expected-coverage partition
(step 4) and the §5.5.2b pre-flight — are **pull-forward work**: none depends on the sampler,
so do them during M6/M7 while compute is idle. They resolve D14/D15, which block M5b.

#### 5.5.2 Real-data ladder (D9)

The honest structure of what exists: **physical mock standards top out around 20–21
organisms.** There is no real 100-genome mock community. Above ~21, everything is simulated.
Plan around that split rather than looking for a real ladder that doesn't exist.

| Rung | Material | Organisms | Why it's on the ladder |
|---|---|---|---|
| R1 | ZymoBIOMICS D6300 / D6305 / D6311 | 10 (8 bacteria + 2 yeast), even | Baseline. Wide GC range (15–85%) and varied genome sizes. Zymo's own characterisation used Illumina HiSeq and MiSeq 2×150. |
| R2 | ZymoBIOMICS D6310 / D6311 log-distribution | same 10, spanning ~10²–10⁸ cells | **The important one for this project.** Directly tests whether stratified starts and island budget let low-abundance organisms get sampled at all — the §3.3 failure mode, on real data. |
| R3 | ZymoBIOMICS D6331 gut standard | 21 strains, bacteria + archaea + fungi | Contains closely related strains, so it is a strain-resolution test rather than a species test. Reference genomes published at the Zymo S3 `D6331.refseq.zip` bundle (the one hifiasm-meta used). |
| R3′ | ATCC MSA-1003 | 20 strains, staggered abundance | Independent alternative to R3; different vendor, different genome set — useful as a check that results aren't Zymo-specific. |
| R4 | CAMISIM / CAMI simulated | 60 → 1500 species | The only route above ~21. CAMI generated 150 bp PE reads with an Illumina HiSeq error profile at low/medium/high complexity; more recent work has used CAMISIM at 60, 150, 600, 1000 and 1500 species. |

**Why R4 matters more than complexity alone:** CAMI ships a **gold-standard assembly**, which
is precisely what calibration metrics need — per-base ground truth rather than just a set of
reference genomes. For a project whose main claim is about uncertainty quantification, that is
worth more than the extra species.

**Caveat to carry:** CAMI-style simulated data is generated with an ART/HiSeq error profile,
so validating a skiver-derived error model on it is a fourth flavour of the circularity in
§2.5 — different simulator, but still a simulator. R1–R3′ are the phases that carry real
sequencing error.

**Accession sourcing.** I did not find a single canonical HiSeq 2×150 *shotgun* accession for
D6300 that I'd want to hard-code here. Concrete leads:
- Zymo publishes per-lot shotgun characterisation data retrievable by the lot number printed
  on the tube — the most direct match to whatever material is used.
- BioProject PRJNA587452 (SRR10391187, Illumina; SRR10391201, ONT) is a ZymoBIOMICS standard
  dataset from the Emu paper. Emu is a 16S profiler, so check whether the Illumina run is
  amplicon or shotgun before using it — if amplicon, it is directly useful for the 16S
  development phase rather than the shotgun phase.
- Reference genomes: the Zymo S3 refseq bundles (D6331 confirmed; check for a D6300
  equivalent).
- For a canonical shotgun run, query SRA/ENA for the standard's catalogue number restricted to
  ILLUMINA + WGS, and pick a run with 2×150 and adequate depth. Record the accession in the
  data manifest so runs are reproducible.

**Task for M5b, pulled forward:** run skiver on the selected real dataset to produce the P3
error model (D10), and re-estimate the fragment distribution from it — the 400 ± 100 assumption
is a prior, and the real library will differ. This needs reads but not the sampler, so do it
during M6/M7; M4 item 6's P2 → P3 comparison then has its P3 side ready the moment M7 lands.

#### 5.5.2a R2 as primary (D12) — two things that must be handled first

**Define the in-scope set by coverage, before running anything.** The log-distribution standard
spans roughly 10² to 10⁸ cells. At any realistic sequencing depth the lowest members sit below
1× coverage, which means they are not in the metaSPAdes graph at all. Benchmarking "did the
sampler recover the rare organism" against an organism that never entered the graph measures
graph construction, not sampling, and will read as a minoodle failure when it isn't one.

Procedure: from the chosen run's depth and the standard's nominal composition, compute expected
per-organism coverage; partition members into `in-graph` (comfortably above the assembler's
retention threshold), `marginal`, and `absent`. Report recovery **conditional on the in-graph
set**, and report the marginal set separately as a graph-construction observation. Record the
partition in the data manifest — it is dataset-specific and will change with depth.

**Keep R1 as the calibration baseline.** R2 is the right primary *stress* test, but calibration
curves need a dataset where every member is well represented; otherwise low coverage confounds
the calibration signal and you cannot tell a miscalibrated sampler from a starved one. Two
distinct roles: R1 answers "are the probabilities right", R2 answers "does the sampler reach
the rare members at all". Run both; don't let R2 displace R1.

#### 5.5.2b Reference genome exactness (D13) — verify rather than assume

The decision records that the Zymo reference genomes should be exact. Treat that as a
hypothesis to check in one cheap pre-flight step, not as a premise, for three reasons:

1. The standard is a batch of cultured cells and cultures drift; Zymo's per-lot characterisation
   exists because lots differ.
2. Some deposited assemblies in mock-community bundles are drafts rather than closed genomes.
   **rRNA operons are the usual casualty** — short-read-derived drafts routinely collapse
   multi-copy rRNA loci. Since 16S is the development target, a collapsed operon in the
   reference makes "the sampler got the copy structure wrong" and "the reference is wrong"
   indistinguishable, which is the one confusion this project can least afford.
3. Eukaryotic members (*S. cerevisiae*, *C. neoformans*) are more prone to collapsed repeat
   regions still.

**Pre-flight check (half a day, do it before any R-rung run):** map reads to the references and
measure (a) the fraction of positions carrying a *fixed* non-reference allele — fixed
differences are strain divergence, not sequencing error, and (b) whether rRNA operons appear at
expected copy number or collapsed, via depth ratio at those loci versus genome median.

If fixed differences exist, build a batch-consensus reference before using per-base ground
truth. **Circularity warning:** correcting the reference using the same reads you then score
against inflates apparent accuracy at exactly the positions where the model does work. Build the
corrected reference from an *independent* dataset — a different run of the same standard, or
long reads for the same catalogue number — or else exclude the corrected positions from
calibration. Note which was done in the results.

If the check comes back clean, the cost was half a day and D13 is now evidence rather than
assumption.

#### 5.5.2c Candidate R2 dataset — accessions and pre-flight (D14)

18 BioSamples supplied, all from one study, all reported as Illumina, all based on
ZymoBIOMICS Microbial Community Standard II (log distribution):

```
SAMN41057710  SAMN41057711  SAMN41057712  SAMN41057713  SAMN41057714  SAMN41057715
SAMN41057716  SAMN41057717  SAMN41057718  SAMN41057719  SAMN41066578  SAMN41066667
SAMN41066762  SAMN41066850  SAMN41066940  SAMN41067035  SAMN41067131  SAMN41067227
```

Reference bundles already downloaded locally:
`~/Downloads/ZymoBIOMICS.STD.refseq.v2.zip`, `~/Downloads/ZymoBIOMICS.STD.refseq.v3.zip`.

**Note on "synthetic metagenome".** That string is the NCBI taxon label conventionally applied
to mock communities, not a statement that the reads were simulated. It does not by itself
indicate in-silico data — but it does not rule it out either, and an in-silico dataset would
be useless for the P3 error model, which is the entire reason for using real data. Confirm
from the run metadata (instrument model, run date, real quality-score distribution) before
committing. A simulated dataset will usually give itself away in the quality profile.

**Step 1 — resolve runs and metadata.** The ENA portal API returns everything needed in one
call per sample (or one call for the whole list), including run accession, instrument,
library layout, read count, base count and read length:

```
https://www.ebi.ac.uk/ena/portal/api/filereport?accession=<SAMN>&result=read_run&fields=run_accession,sample_accession,instrument_platform,instrument_model,library_layout,library_strategy,library_source,nominal_length,read_count,base_count,fastq_ftp&format=tsv
```

NCBI edirect equivalent, which returns spots, bases, avgLength, LibraryLayout, Platform,
Model, BioSample and BioProject in one CSV:

```bash
# samples.tsv is the supplied accession list
for s in $(tail -n +2 samples.tsv | cut -f1 | tr -d '"'); do
    esearch -db sra -query "$s" </dev/null | efetch -format runinfo
done | awk 'NR==1 || $0 !~ /^Run,/' > runinfo.csv
```

Once any one sample yields the BioProject, the whole study comes back in a single call:

```bash
esearch -db sra -query PRJNA<id> | efetch -format runinfo > runinfo.csv
```

Record the full table in the data manifest — run selection has to be reproducible and the
in-scope partition (§5.5.2a) depends on depth.

**Step 2 — reference bundle choice.** Diff v2 against v3 before picking. Check specifically:
which strains are present, whether any assembly is a draft rather than closed, and whether
rRNA operons are present at expected copy number. Note that Standard II datasheets name the
yeast as *Cryptococcus deneoformans* where older material says *C. neoformans* — a renaming,
but confirm the bundle's genome matches the strain actually in the standard. Pin one bundle
version in the manifest and do not silently switch later; every calibration number depends on
which one was used.

**Step 3 — verify with sylph.** Sketch the reference genomes into a sylph database, profile
each candidate run against it, and check that all ten members are detected. Lower the
subsampling rate from the default when sketching: the point of this standard is that its
lowest members sit near the detection limit, and default parameters are tuned for ordinary
metagenomes, so a low-abundance member reported as absent may be a parameter artefact rather
than a real absence.

sylph's per-genome **effective coverage** output is the useful by-product here: it is close
enough to what §5.5.2a needs to build the in-graph / marginal / absent partition directly, so
this step does double duty as verification and triage. Run it across all 18 samples and pick
the run whose partition puts the most members in `in-graph` — that is the run that maximises
what R2 can actually test.

**Step 4 — expected-coverage calculation, with two corrections that are easy to miss.**

1. *Cell counts are not DNA mass.* The log distribution is specified in cells, and two members
   are yeasts with genomes an order of magnitude larger than the bacteria. Convert cell
   fraction → genome copies → bases using each genome's size before predicting coverage, or
   the predicted ordering will not match what sylph reports and the disagreement will look
   like a bug.
2. *Whole-cell vs DNA standard.* D6310 is whole cells and D6311 is extracted DNA. If these
   samples used the whole-cell standard, lysis efficiency shifts observed abundance away from
   theoretical, and the theoretical composition is the wrong baseline. Establish which from
   the sample metadata.

The sylph-measured coverage is the operative number in either case; the theoretical
calculation is a cross-check that should agree to within a factor of a few. If it doesn't,
find out why before running any benchmark.

#### 5.5.3 Metrics

N50 is not a metric here, and no assembly-contiguity number should appear in any results table.

- **Calibration:** are level-α credible sets covered α of the time across replicates?
- Rank of ground truth in the weighted output.
- Edit distance MAP → truth.
- Per-locus ESS at run end.
- `log Ẑ` spread across islands and seeds.
- Recovery rate as a function of organism abundance (specifically for R2).

**M5b gate:** §5.5.2b pre-flight complete and its outcome recorded; calibration within
tolerance on R1; recovery reported conditional on the R2 in-scope set; documented failure modes
on the R2 marginal set. (L0–L2 calibration and L3 failure modes are M5a's gate, already passed.)

#### 5.5.4 HPC execution discipline — design for few round-trips

M5b's iteration cost is a queue wait, not a runtime. The engineering goal is to make each
HPC visit answer as many questions as possible and to make local reproduction of a remote
failure possible without a second visit.

**One artefact, one command.** A run is `minoodle sample` (§4/M9) plus a config file and a
manifest — no interactive steps, no notebook, no "then I ran this bit by hand". Whatever is
needed to produce a results table must be reachable from the manifest alone. M5a's gate
enforces this while it is still cheap to fix.

**Batch the ladder, don't walk it.** Submit R1, R2 and the R4 rungs as one job array over the
(dataset × seed × island-count) grid, with independent outputs, rather than running a rung and
deciding what to do next. A failed cell in an array is one cell; a failed serial rung is a
whole round-trip.

**Fail fast, locally, first.** Every config gets a `--smoke` run — the same code path, tiny
particle count, one locus — that must pass on the dev machine before submission. Most remote
failures are path, environment and config errors, and those cost a full queue wait each if the
first thing the cluster does is a real run. Job scripts run the smoke config as step 0 and
abort the array on failure.

**Ship diagnostics, not just results.** Each run writes, unconditionally: per-locus ESS
traces, `log Ẑ` per island, resampling event counts, wall clock and peak RSS per phase, plus
the exact config and both git SHAs. The question "why did that run behave oddly" must be
answerable from downloaded output. Re-running remotely to add a diagnostic is the single most
expensive mistake available in this milestone.

**Deterministic replay of any remote particle.** The §4.2 injected-uniforms seam already
allows this: a run records its uniform stream (or the seed plus stream position for a chosen
lineage), so a suspicious remote trajectory replays bit-identically in the local Python
oracle. Keep the recorded stream small — one lineage, not the whole run — and write it on
demand via a config flag, not by default.

**Environment pinned, not rebuilt.** Per D17: every external tool is a container, run locally
under Docker (biocontainers) and on HPC under singularity via the nextflow pipeline, with the
**image digest** — not the tag — recorded next to the outputs. Do not debug a compiler or a
dependency resolver at the front of a queue.

**Estimate before submitting.** M7's scaling curve gives time and memory per (particles ×
bases × unitigs). Use it to set walltime and memory requests; a job killed at the limit at
hour 11 is the worst possible outcome and is entirely avoidable arithmetic.

**Small-data proxy kept green.** L2 stays runnable locally in minutes throughout M5b. Any
change made in response to a real-data finding is validated against L2 and the frozen M6
fixtures locally *before* it goes back to the cluster.

### M6 — Python performance ceiling and fixture freeze (2 days) — *after M5a*
- numba the read-scoring loop and the weight loop; multiprocessing across islands.
- Store paths as ancestry pointers into a shared arena, not copied lists — the difference
  between 1e4 and 1e6 particles.
- Profile before optimising. Expect read scoring (alignment + emission lookup) to dominate.
- **Freeze golden fixtures** per §4.2.

**Gate:** L2 and L3 run end-to-end in Python within tolerable wall clock; fixtures frozen.
R1 is no longer part of this gate — it has moved to M5b, which is after the port. Substitute
an L3-scale extrapolation to R1's graph size instead of running R1 itself.

### M7 — Rust port (4–5 days) — *before the real-data ladder*
Same module structure, traits mirroring the ABCs. Validated exclusively against M6 fixtures
with the injected-uniforms scheme.

**Gate:** every fixture matches to 1e-9; scaling curve over cores and particle count
documented. Rough target, to revise with M5a's numbers: 1e5 particles × 1e4 bases on a
1e6-unitig graph, under an hour on 16 cores, under 32 GB. The scaling curve is not optional
paperwork — §5.5.4 sizes every HPC job request from it. If the curve says an R-rung will not
fit, that measurement is what justifies pulling M8 forward; nothing else does.

### M8 — Minimizer space (3 days) — *avoid unless M7's numbers force it*
Instantiate `TokenSpace` with `(w, k)` minimizer tuples à la mdbg; reuse
`~/Documents/rust-mdbg` for graph construction. Nothing in the sampler changes.

**Per D5, output is minimizer sequences only.** Base-space reconstruction deferred (§7). Two
caveats not to paper over:

1. **The error model changes.** A single base error usually destroys a minimizer, appearing as
   a *deletion* in minimizer space, not a substitution. The error model must be re-parameterised
   on minimizer-space error statistics — derivable from skiver's base-space model by
   simulation, but not reusable as-is.
2. **Coverage semantics change.** Minimizer-space depth is not base depth. Re-derive the
   NegBin parameters.

Minimizer-space output limits downstream use to things that work on minimizer sequences
(profiling, clustering, sketch comparison). Annotation and CDS extraction — part of the
original motivation — need §7.

**Gate:** M2's toy-graph validation repeated in minimizer space, same TV criterion.

### M9 — Interfaces and packaging (1–2 days)
- `minoodle sample --graph g.gfa --reads r1.fq r2.fq --particles N --islands M --out ...`
- Weighted FASTA + Parquet sidecar: sequence id, log weight, normalised weight, path node
  list, per-term likelihood decomposition, error-model phase tag.
- Seed everything; log full config.
- `--smoke` (tiny run, same code path) and the unconditional diagnostic bundle of §5.5.4.
  These are needed *at M5b*, so build them there and let M9 only tidy the surface.

---

## 6. Anti-goals

1. No likelihood term that isn't incrementally decomposable (§2.4).
2. No likelihood-threshold stopping (§2.8).
3. No taboo/visited-node exclusion (§3.3).
4. No parameter tuning against the benchmark before calibration gates pass.
5. Do not skip M1.
6. Do not report N50 or contig-style metrics as evidence the method works.
7. Do not start the Rust port before M6, and do not start the real-data ladder (M5b) before
   M7 — a queue wait per iteration on the slow implementation is the most expensive way
   available to find a bug the synthetic ladder would have caught.
8. Do not quote P1 or P2 numbers outside the repo (§2.5).
9. Do not mix inner distance and fragment length (§2.6).
10. Do not run anything on HPC that has not passed its `--smoke` config locally (§5.5.4), and
    do not go back to the cluster for a diagnostic that should have been written by default.
11. Do not pick the extension direction using the data (§2.1). Alternation is deterministic;
    "extend whichever side looks more promising" makes the target history-dependent, which is
    the same defect as a taboo list (§3.3) wearing different clothes.

---

## 7. Deferred work

- **Base-space reconstruction from minimizer-space paths** (D5). Requires storing spanning read
  sequences per minimizer-space node, as metaMDBG does. Needed before annotation, CDS
  extraction, or variant calling on minimizer-space output.
- **Global coverage deconvolution** as an outer loop (§2.7 v1): conditional SMC + abundance
  fitting + residual coverage. The route to strain-level resolution and the strongest
  differentiator against BayesPaths.
- **Endpoint-conditioned bridging SMC** — seed both mates of a pair and grow toward closure.
  Considered at rev 10 and rejected for v0, recorded because it is an idea that recurs. It does
  not do what it looks like it does: two disjoint frontiers have no fragment distance `d`
  between them, so no paired-end term exists until they meet, and steering extension toward a
  known target is data-dependent construction of the kind §3.3 rules out. It is a genuinely
  different algorithm — conditioning on both endpoints — not a change of start point. The cold
  start it was meant to fix is short anyway: two-sided paths (§2.1) activate term B within
  ~one fragment either side of the seed, and §2.6 caps paired-end reach at ~100–250 bp beyond
  the reads, so pairs were never the long-range discriminator. Revisit only with a proper
  bridging formulation and an importance weight for the conditioning, not as a seeding tweak.
- Particle Gibbs / conditional SMC with ancestor sampling for long-range phasing.
- Annealed SMC (tempering) for tangled regions.
- Emitting indel continuation probabilities from skiver directly.
- gLM-derived path prior.
- Multi-sample differential coverage.

---

## 8. Decisions

**Resolved:**
- **D1** Python first, Rust port for performance, Python as test oracle (§4.2 for the contract).
- **D2** metaSPAdes GFA canonical.
- **D3** (amended rev 9) The error model is skiver's composable context model, not an HMM —
  a customisable component string giving a per-position emission table; a latent-state layer
  is optional and normally unused. Read scoring defaults to fixed alignment (edlib) plus a
  flat gap penalty; the Match/Insert/Delete pair-HMM is level 2, taken only on ablation
  evidence. skiver emits no indel length distribution, so if level 2 is reached, gap-extend
  comes from `P(error at i+1 | error at i)` per §2.5.
- **D4** k=21.
- **D5** Minimizer sequences as output; base-space reconstruction deferred.
- **D6** Phased error-model strategy P1 → P2 → P3 (§2.5). P2 bounds the circularity rather
  than removing it; only P3 numbers are externally quotable.
- **D7** Single odd `-k 21`.
- **D8** `--only-assembler`.
- **D9** Zymo standards, with the R1–R4 ladder in §5.5.2. Physical standards cap at ~21
  organisms; above that, CAMISIM.
- **D10** Fragment length ~400 ± 100 with 2×150 as the prior; re-estimated from real data at
  M5. skiver run on the real dataset produces the P3 model. Terminology fixed per §2.6.

**Open:**
- **D11** metaSPAdes accepts single-k under `--only-assembler`, which is the mode in use.
  Still run a five-minute smoke assembly as the first task of M3 rather than discovering a
  version-specific objection four days in.
- **D12** R2 (log-distribution) is the primary real-data rung. R1 stays as the calibration
  baseline — see §5.5.2a for why both are needed and for the in-scope-set procedure.
- **D13** Reference genomes taken as exact, subject to the §5.5.2b pre-flight verification.
- **D16** Optimisation goes between the ladders: M5a → M6 → M7 → M5b. The real-data rungs are
  HPC-bound, so they run on the Rust implementation; §5.5.4 governs how. M8 stays deferred and
  is only pulled forward on evidence from M7's scaling curve.
- **D17** External tools run in containers, never local installs: biocontainers under Docker
  locally, singularity via the nextflow pipeline on HPC (§5.5.4). Record the image **digest**
  alongside the outputs — a tag is not a version. Settled at M3, where metaSPAdes 4.3.0 ran as
  `quay.io/biocontainers/spades@sha256:2af76c9b…` and its provenance was written to
  `asm/provenance.json` next to the GFA.
- **D18** Read-sampled seeding is a **proposal**, not the prior. `p_seed` is uniform over
  oriented k-mer positions; `q_seed` is coverage-weighted via the read anchors, normalised by
  the total anchored placement count, mixed with `ε` uniform for full support, and corrected by
  `log p_seed - log q_seed` in the initial log-weight (§2.3). The target therefore stays
  data-independent, seeding strategy becomes a pure efficiency knob, and swapping `q_seed`
  gives a testable invariance rather than a different posterior. Cost: prior-only
  `log Ẑ == 0.0` is exact only under uniform seeding, so that test now names its configuration.

**Open:**
- **D14 (blocks M5):** Which run, from the 18 candidate BioSamples in §5.5.2c. Selection
  procedure is specified there; the decision is which run comes out of it. Also open: whether
  reference bundle v2 or v3 is canonical.
- **D15 (blocks M5, conditional):** If §5.5.2b finds fixed reference differences, which
  independent dataset is used to build the batch-consensus reference.
