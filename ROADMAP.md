# fastMRI E2E VarNet Reproduction + Deep Parameter Interpolation — Roadmap

**Project (two phases):**
- **Phase A (this week):** reproduce End-to-End Variational Network (VarNet) results on the fastMRI knee dataset on CHTC.
- **Phase B (the actual research contribution, later):** extend VarNet with **Deep Parameter Interpolation (DPI)** — from Park et al., "Deep Parameter Interpolation for Scalar Conditioning" (CVPR 2026) — to condition the network on MRI acceleration rate, instead of training one model per rate or training jointly with no explicit conditioning (VarNet's current default behavior).

Phase A exists to get a working, correctly-instrumented baseline before touching the DPI idea — you need a VarNet you understand and trust end-to-end before swapping parameters inside its cascades.

**Lab:** Kamilov Computational Imaging Group (continuing from the diffusion model collapse / ECE 399 work on CHTC).

---

## Where this actually stands (2026-10-05)

The noise / dc email went to Chicago. The reply reordered Phase D: specialists
without added noise first, a test-time noise sweep, and training-slice
evaluation. See "Chicago's reply" below; it takes precedence over the order in
Phase D.

### When to contact Chicago (user's instruction, 2026-10-07)

No update emails mid-stream. Tell the user when one of these is reached:

1. **Interim (optional):** R2/R6/R10 specialists and the mixed 2/6/10 run
   trained and scored (val, training slices, noise sweep). Gives the upper
   bound against mixed at 2/6/10. Fixed-λ DPI goes in as partial W&B curves.
2. **Full stopping point:** all of block D1 done, meaning specialists, mixed
   and fixed-λ DPI at 2/6/10, each scored on val, training slices and the
   noise sweep (with the 2026-10-07 results for blind / full / dc / io /
   R8). This answers every item of the 10-01 meeting and the 10-05 reply.
   Then ask which noise levels to train with (D2).

Already done and waiting for that update: the noise sweep and the
overfitting check on the existing models, the visual noise check, the
`dc` / `io` / R8-specialist per-rate scores (`RESULTS.md`).

## The 2026-10-01 meeting

The 2026-10-01 meeting replaced the 2026-09-24 asks with seven new items.
The main one is a new experiment grid (Phase D below): rates **2, 6, 10**,
three model families, with the original noise and then with added noise.
Phase C blocks B-D are still useful but now rank below Phase D; item 3 is
where to confirm that.

| # | Meeting item (2026-10-01) | Status |
|---|---|---|
| 1 | Make sure the input PSNR/SSIM (zero-filled) are byte-identical to the baseline's, and add this rule to `CLAUDE.md` | **Rule added to `CLAUDE.md` 2026-10-01. Checked by hand for dc 2026-10-01: inputs identical** (byte-identical `zf_ssim`/`zf_mse` against both baseline and `full`; `zf_nmse` differs across node CPUs, see `RESULTS.md`). Still to automate: `paired_from_wandb.py` reads `zf_*` from the baseline run only and never compares it with the treatment's. To do: assert per-volume `zf_ssim`/`zf_psnr`/`zf_nmse` are exactly equal across every run being compared, and fail loudly if not. Run it on the existing pairs (model2 vs DPI full, dc, io, R8 specialist) |
| 2 | Fresh clone of the E2E VarNet code; ask the agent what the noise level is for training and testing | **Answered 2026-10-01 from the paper itself** (Sriram et al., arXiv:2004.06688v2, the `e2evarnetpaper` PDF, all 13 pages read) and its official code (fastMRI at `91f2df4`). **E2E VarNet adds no measurement noise, in training or testing.** The paper writes the forward model with noise, k_i = F(S_i x) + ε_i (eq. 1-2), but never specifies, measures or simulates ε. It trains on the raw fastMRI multicoil k-space with masks applied (section 4.1: Adam 3e-4, 50 epochs, "without any regularization or data augmentation"). So the noise level is whatever the scanner recorded, the same in train and test, and also in the fully-sampled target. The **only noise the paper adds** is dithering in supplement 6.1: a post-processing step on the *output image* for visual sharpness. The reconstruction is divided by its max; Gaussian noise is added with per-pixel std = σ·sqrt(local median over 11×11), **σ = 0.02 for brain** (and non-fat-suppressed knee), 0.03 for fat-suppressed knee; σ was tuned by eye. It is shown only in figs. 4-7, is not in the reported metric tables as far as the text says, is not measurement noise, and is not in the released fastMRI code. Our setup matches the paper here: we add nothing either. Measuring the inherent σ is item 6a |
| 3 | Ask Chicago what to do next | Open. Questions to bring: (a) σ for the added-noise run (item 6); (b) whether Phase C blocks B (remaining scopes), C (SDUM) and D (seeds) continue alongside Phase D or wait; (c) the open "selective DPI" question from 2026-09-30; (d) DPI scope for Phase D (assumed `full`) |
| 4 | Organize the results so far for the midterm presentation | Open. Material: `RESULTS.md` (per-rate table, λ saturation, `dc` ties `full`, R8 specialist headroom), the parameter breakdown (item 7), cost (26.4 h vs 101.7 h vs 33.0 h). **`dc` scored 2026-10-01: beats blind at every rate** (+0.01 to +0.08 dB, all significant) and beats `full` at R6/R8 with 1,012 extra parameters. That is the strongest slide so far (`RESULTS.md` top). Still missing: `io` (training at epoch ~45) and the R8 specialist (cluster 11926921 scored the released knee checkpoint by mistake, no `ckpt=`) |
| 5 | With the original measurement noise: specialist E2E VarNet, mixed E2E VarNet, and DPI with **fixed** λ (linear, λ(6)=0.5) at rates 2, 6, 10; report validation and training performance | Open. Phase D, block D1 |
| 6 | The same experiment with added measurement noise | Open. Phase D, block D2. Needs σ (item 3a) and noise-injection code |
| 7 | Check which modules are learnable apart from DC | **Answered 2026-10-01** (from `plan.md` section 2's measured counts and `dpi/dpi_varnet.py`). Blind VarNet has 29,936,966 learnable parameters in three groups: **the 12 cascade regularizer U-Nets** (`NormUnet`, 18 chans, 4 pools, 2,454,338 each, 29,452,056 total = 98.4%); **the coil-sensitivity estimation U-Net** (`SensitivityModel`, 8 chans, 4 pools, 484,898 = 1.6%); and the **DC step sizes** (`dc_weight`, 1 scalar per cascade, 12). Nothing else learns: the FFTs, masking, coil combination and RSS are fixed. DPI adds a copy of each duplicated tensor plus `phi` (1,000 entries) for λ. `full` duplicates all three groups; `--no_dpi_sens` leaves out the sensitivity net |

## Chicago's reply to the noise / dc email (2026-10-05)

The email (noise table plus dc per-rate result) went out. The reply sets the
order for Phase D:

| # | Ask | What it means here | Status |
|---|---|---|---|
| C1 | **First, train our own specialists without added noise, as the upper bound** | Block D1's three specialists at R2, R6, R10 on the original data. No code needed: `training/` model1 preset with `accelerations=R center_fractions=0.32/R`, as the R8 specialist was run. "Noise-free" here means *no added noise*; the data still carries its inherent σ ≈ 0.5% of max (`RESULTS.md`). Say so when reporting | **Submitted 2026-10-06** (12006400-02), running |
| C2 | Put the zero-noise row in the table plus several added noise levels; run "whatever evaluation workflow you used" across them; 0.05 as a test value for data in [0,1] or [-1,1]; check visually that the images look noisy; add noise **in measured k-space**: y = Ax + noise_level · noise | A test-time noise sweep in `verification/verify_varnet.py`: new `--noise_level` (one pass per level). Noise = standard normal on each real/imaginary channel of the full k-space, scaled by noise_level × the volume's target max (our data is not normalised; the RSS target max plays the role of the data range, so in image units per coil σ = noise_level, the same unit as the measured inherent 0.005), added **before** masking so y = M(FSx) + M·n. Seeded per volume/slice so every model gets byte-identical noisy inputs (the input identity rule). Target stays the original image. Save example PNGs (fully-sampled RSS and zero-filled, with and without noise) per level for the visual check. Proposed levels: 0, 0.005, 0.01, 0.02, 0.05 (0.05 = 10x the inherent noise). Then score every checkpoint (blind, full, dc, the specialists) at each level | **Code done 2026-10-06; sweep submitted** for 5 existing models (state table below). Specialists get swept when they finish |
| C3 | Q: "Is the second table the same results as last week? I recall the baseline was the strongest" | **Different model, same baseline run.** Last week's table was DPI `full` vs blind: `full` won R2/R4 and lost R6/R8, so the baseline was stronger at high rates. This week's is the DC-only variant (`dc`), a different checkpoint, which beats blind at every rate. The blind numbers are the identical `verify-model2-brain` scores in both. At R8 the strongest model is the R8 specialist (+0.22 dB over blind, unpaired) | Answered in the user's reply (draft 2026-10-05) |
| C4 | Evaluate PSNR/SSIM on **training slices** for every method to check overfitting | The meeting note "report performance validation + training" means scored metrics on training data, not just train loss. Needs `run_verify.sh` / `verify.sub` to accept `brain_multicoil_train_batch_0.tar.xz` (they handle `multicoil_val` only today), then a train-split pass per checkpoint and a train−val gap per method and rate. Already suggestive: DPI `full` reached 34% lower train loss than blind with no val gain | **Code done 2026-10-06; submitted** for 5 existing models |

Order: C1 (submit now, ~27 h per specialist), C2 and C4 code in parallel,
then score everything at all noise levels on val and train, then the rest of
block D1 (mixed and fixed-λ DPI at 2/6/10), then D2 (training with added noise).

**Update 2026-10-07 (read from W&B; CHTC session had expired):**
- Done: all 8 scoring jobs for blind / full / dc / io (noise sweep + training
  slices), plus `verify-dpi-io-brain` and `verify-blind-r8-brain`; the
  `condor_rm` of those two never ran because the connection had dropped.
  Results in `RESULTS.md`, "Noise sweep and training-slice scores": the
  clean-trained models break down under added noise (below zero-filled at
  0.02), DPI `full` degrades least, and no method overfits (train−val
  ≤ 0.18 dB, the same for all).
- **R10 specialist (12006402) failed at epoch 19 after 10.6 h**:
  `DataLoader worker ... killed by signal: Killed`, most likely the 48 GB
  memory cap (check `condor_history 12006402 -af MemoryUsage`). Resume from
  its `last.ckpt` with `mem=64GB`, taking args from
  `condor_history 12006402 -af Args`.
- Running at last check: R2 specialist (epoch 32), R6 (epoch 26).
- Never started (idle since 10-06 17:10): mixed 2/6/10 (12015718), fixed-λ
  DPI (12015819), R8-specialist noise/train scoring (12015614/15). If they are
  still idle, check `condor_q -better-analyze`.
- Not yet looked at: the noise PNG panels (`~/eval_job/.../images/`).

**Update 2026-10-07 10:30 (on CHTC):**
- R10's hold was the memory cap ("Docker job has gone over memory limit of
  49152 Mb"; the image jumped 32 → 46 GB around 03:09). **Its checkpoint was
  lost**: a memory hold does not transfer `output/` back (an eviction would),
  and nothing was spooled. Removed. Resubmitted from scratch as
  **`blind r10 brain v2`, cluster 12078790, `mem=64GB`**.
- Idle 2/6/10 runs 12015718 / 12015819 raised to 64 GB (`condor_qedit`
  RequestMemory). They match 43 GPU slots and are waiting on cluster demand,
  not misconfigured. R2 / R6 run at ~30 GB on 48 GB; if either gets
  memory-held, resubmit with `mem=64GB` (its checkpoint will be lost the same
  way).
- **Retry bug found and fixed:** `verify-blind-r8-brain` (12012758) ran 4
  times. A specialist's expected `fail` verdict exited 1, and `max_retries = 5`
  reran it. Now a verdict failure exits 2 and `verify.sub` has
  `retry_until = 2` (dry-run checked on CHTC). 12012758 was removed;
  12015614/15 were set to `JobMaxRetries 0`. Staged copies updated.
- Visual check done (`RESULTS.md`): 0.05 is clearly noisy; most of the
  reconstruction error under noise sits in the background.

**State at end of 2026-10-06 (all submitted by Claude over `ssh chtc`):**

| what | cluster(s) | where | status |
|---|---|---|---|
| specialists R2 / R6 / R10, no added noise (C1) | 12006400 / 12006401 / 12006402 | `training/runs/brain/model1/` | running |
| mixed blind at 2/6/10 (D1) | 12015718 | `training/runs/brain/model2/` | queued |
| fixed-λ DPI `full`, 2/6/10, λ = 0 / 0.5 / 1 (D1) | 12015819 | **`~/eval_job/dpi/runs/brain/dpi-fixed/`** | queued |
| `io` per rate (`verify-dpi-io-brain`) | 12012757 | `verification/runs/brain/model2/` | queued |
| R8 specialist per rate (`verify-blind-r8-brain`) | 12012758 | `verification/runs/brain/model2/` | queued |
| noise sweep, val, levels 0 / 0.005 / 0.01 / 0.02 / 0.05 (C2): model2, dpi, dpi-dc, dpi-io, blind-r8 | 12015606 / 08 / 10 / 12 / 14 | **`~/eval_job/verification/runs/brain/model2/`** | queued |
| training-slice scores (C4), same five models | 12015607 / 09 / 11 / 13 / 15 | **`~/eval_job/verification/runs/brain/model2/`** | queued |

W&B run names: `verify-<tag>-brain-noise` and `verify-<tag>-brain-train`,
with tag ∈ {model2, dpi, dpi-dc, dpi-io, blind-r8}. The baseline checkpoint is
`training/runs/brain/model2/10833090` (50 epochs; 10826573 is the 50-step
test). Check: the noise job's level-0 pass must reproduce `verify-model2-brain`
exactly.

**`~/eval_job` on ap2001 is a staged copy of uncommitted code**, made so that
CHTC's git checkout stays clean (Claude does not change git state). It
holds the new `verify_varnet.py` / `run_verify.sh` / `verify.sub` (noise
sweep, train split) and the new `dpi/` files (`--lambda_mode fixed`), and
`.env` is a symlink. After the user commits, pushes and pulls these changes
on CHTC, move `~/eval_job/*/runs` into the checkout's `runs/` trees and delete
`~/eval_job`. To resume the fixed-λ run after an eviction, use
`~/eval_job/dpi/submit.sh resume=...` (or the checkout once it is pulled).

Code added 2026-10-06 (uncommitted; tests green locally):
- `verification/verify_varnet.py --noise_levels`: k-space noise, seeded per
  (file, slice), scaled by target max, level 0 bit-identical to before. PNG
  panels in `output/images/`.
- `run_verify.sh` / `verify.sub split=train`; `request_disk` raised to 360 GB.
- `paired_from_wandb.py`: any rate list, `--noise`, an automatic input
  identity check (exit 2), and the macOS SSL fix.
- `dpi/ --lambda_mode fixed` (+5 tests, 54 pass), wired through `train.sub`.
- `dpi/submit.sh` no longer prints the W&B key. It did print it on
  2026-10-06, so it is in that session's transcript: **rotate the key
  once the running jobs finish** (rotating now would break their W&B logging).

Retroactive check: `paired_from_wandb.py` confirms input identity for last
week's `full`-vs-blind and this week's `dc`-vs-blind comparisons (all 5
passes byte-identical in `zf_ssim`/`zf_mse`).

## Phase D (2026-10-01): rates 2/6/10, specialist vs mixed vs fixed-λ DPI

Every run trains and validates on the same brain batch 0 data, with
`equispaced_fraction` masks, 12 cascades, Adam 3e-4, batch 1, 50 epochs,
seed 42. Center fractions follow the 0.32/R convention already in use
(0.16, 0.0533, and **0.032 for R10**, which is new). Inputs must be
byte-identical across models (item 1). "Report training + validation" means
the train-loss curve plus Lightning's validation per epoch from W&B, and
afterwards the per-rate 460-volume scoring with `paired_from_wandb.py`.

### Block D1: original measurement noise (item 5)

- [ ] **Code: a fixed-λ DPI option.** Not in `dpi/` yet. `LambdaTable` with
  `phi` as a buffer is not enough: `phi` is `randn`, so a frozen λ would be
  noisy and only roughly linear. Needs an analytic λ(R) = (R − 2) / 8,
  clamped to [0, 1] (0, 0.5, 1 at R2, R6, R10), with nothing learnable. Use
  `spacing="linear"`, `accel_min=2`, `accel_max=10`. Proposed flag:
  `--lambda_mode {learned,linear}`. Tests: λ values exactly 0 / 0.5 / 1, no
  `phi` gradient or optimizer group, checkpoint rebuilds in
  `verification/verify_varnet.py`, parameter count = `full` minus 1,000.
  Run the local gate (`make local-run` in `dpi/`, `make smoke-ckpt` in
  `verification/`)
- [ ] **Presets:** a rate-list preset for 2/6/10 in `training/submit.sh`,
  `dpi/submit.sh` and `verification/submit.sh`, which hard-code 2/4/6/8 today.
  Pass `accel_max=10` in `dpi/train.sub`
- [ ] **Submit 5 training jobs:** specialist R2, specialist R6, specialist
  R10 (model1-style, one rate each); mixed blind at 2/6/10; DPI `full`,
  fixed linear λ, 2/6/10. The existing R8 specialist and the 2/4/6/8 models
  do not cover these rates and are not reused
- [ ] Score all five at R2/R6/R10 plus mixed; pair the mixed and DPI runs
  against each specialist at its own rate. Training report: train loss and
  val loss/PSNR/SSIM per epoch, matched epochs

### Block D2: added measurement noise (item 6)

- [x] **6a, measure the existing noise. Done 2026-10-02**, cluster 11927049
  (run from `~/noise_job` on ap2001, a copy outside the git checkout; the
  repo's `noise/` is the same code). Result in `RESULTS.md`, "Measurement
  noise already in the data": σ ≈ 0.5-0.6% of the target max, full-k-space
  SNR ≈ 12-14 dB, identical in train and val, coils not prewhitened, ground
  truth floor ~2.9% of max. Original plan: One CPU job per split
  (train batch 0 and val batch 0, the data every model uses). Each streams
  its tarball through `xz` without extracting it and measures, per slice:
  coil σ from the air in the readout-oversampled margin, in image, k-space
  and target-max-normalised units; measurement SNR in dB, full and for each
  mask at R2/4/6/8/10; coil noise correlation; the target's background noise
  floor; and whether any noise data or noise header field exists. Local gate
  green: `make test` (σ recovered within 3% on synthetic k-space with known
  noise, zero padding included) and `make job-smoke` (the job script in the
  verify image). Next: on ap2001, `cd noise && make submit-test`, then
  `make submit`, then `make summary` into `RESULTS.md`. This is the base σ
  that any added noise is measured against
- [ ] **σ from Chicago** (item 3a): an absolute σ, or a multiple of the
  measured σ, or several levels. Also confirm: is noise added in training
  and validation both, and is the target still the clean (original) RSS?
- [ ] **Code: noise injection** in the data transform: complex Gaussian on
  the full k-space before masking, seeded per volume/slice (as the mask is),
  so every model sees byte-identical noisy inputs (item 1 extends to this).
  Same hook for the baseline, specialists and DPI
- [ ] Rerun block D1's five jobs with noise; score and report the same way

## Previous status (2026-09-24, updated 2026-09-28; superseded by the 2026-10-01 meeting above)

Supervisor feedback on the first reported result (message of 2026-09-24,
signed "Chicago") set four items. Two are answers, two are experiment
blocks; the meeting happens after (3) and (4) are done.

| # | Ask | Status / where the plan lives |
|---|---|---|
| 1 | Is the result on the official evaluation data the benchmark relies on? | **Answered by email (draft in Gmail, 2026-09-24).** Short version: yes for the split (NYU's `brain_multicoil_val`, batch 0, scored with `fastmri.evaluate`), no for the leaderboard test split, which has no public leaderboard since 2023. Full wording below, "Answer to item 1" |
| 2 | Performance per acceleration rate, reported at next week's meeting | **Done 2026-09-29 — `RESULTS.md`.** DPI trades accuracy from high rates to low rates: wins R2/R4, loses R6/R8, all significant except R8 NMSE. Paired Wilcoxon over 460 volumes per `VERIFICATION.md` 6.3. Provisional until block D measures sigma |
| 3 | Several DPI variants | Variant grid and priority order in `plan.md`, "DPI variant grid (2026-09-24)". `--dpi_scope` re-added 2026-09-28 (`scope=io` etc. in `dpi/submit.sh`); the variants are now submissions, not code. Plan: Phase C, block B |
| 4 | Add NV-Raw2insights-MRI (SDUM) as a baseline at a matched parameter count, identical input setup, implementation checked against their code | Architecture chosen and parameter-counted: their 12-cascade Restormer narrowed to widths 64/128 lands at **29.93M vs VarNet's 29.94M**. Table, parity checklist and correctness test in `plan.md`, "Matched-parameter SDUM baseline". Plan: Phase C, block C |

**Update 2026-09-28.** Both baseline training runs are finished (model1 and
model2, brain, 50 epochs). Block A's harness work landed 2026-09-25 and block
B's `--dpi_scope` landed 2026-09-28, so the remaining items in both blocks are
submissions and a report, not code. The repo holds no run outputs; W&B and the
checkpoints under `training/runs/` and `dpi/runs/` on the access point are the
record, so the exact cluster state is whatever `condor_q` and W&B say.

**What W&B says (read via the API, 2026-09-28).** Both 50-epoch brain runs
finished, same seed and setup:

| | blind `mixed acceleration brain` | DPI `full`, from scratch |
|---|---|---|
| val SSIM (mixed, epoch 49) | 0.95355 | 0.95363 (+0.00009) |
| val PSNR | 39.734 | 39.784 (+0.05 dB) |
| val loss | 0.04637 | 0.04629 |
| train loss | 0.0342 | 0.0224 |
| lambda(R4), lambda(R6) | — | 0.626 → 0.656, 0.825 → 0.847 (all of it in epoch 0→1, flat after) |

DPI led in 39/50 epochs, but the lead peaked at +0.002 SSIM around epoch 10
and shrank to +0.0001 after the LR drop at 40. Read plainly: doubling the
parameters bought a 34% lower **training** loss and no validation gain, and
lambda barely moved after epoch 1. The first guess, that the two parameter
sets never separated, is **wrong**: `dpi/divergence.py` on the epoch-49
checkpoint (cluster 10976827) gives an overall ||W − Wc|| / ||W|| of 0.79,
with W/Wc cosine 0.2–0.5 in the most-moved tensors. Both sets start
identical, but λ(2)=0 and λ(8)=1 route R2 gradients only to Wc and R8
gradients only to W, so they split from the first step. The spread is even:
0.7–0.84 through the deep U-Net levels and bottleneck, lower only at the
edges (down L0 0.55, final 1x1 0.32, dc_weight 0.16). `io` covers 0.2% of the
squared movement, `shallow` 6.5%. So the endpoints are close to two different
networks that reach the same validation score, and weight-space distance
alone does not show the network *uses* the rate. Per-rate scores exist
for the baseline only (`verify-model2-brain`, 460 volumes: SSIM 0.9735 /
0.9572 / 0.9467 / 0.9377 at R2/4/6/8, mixed 0.9534); the DPI checkpoint has
not been scored per rate.

**Consequence for block B.** The narrow scopes exist to make a gain cheaper.
There is no gain yet, so `io`/`dc` queued now would most likely tie the
baseline and show nothing. Sequence instead (details in the 2026-09-28
conversation notes, `dpi/README.md`):

1. [x] *No GPU:* `dpi/divergence.py` on the finished `full` checkpoint (done
   2026-09-28, result above: the sets separated heavily and evenly, so no
   narrow scope reproduces `full` in weight space). Still open: a
   **mismatched-rate test**, where the R=8 volumes are scored with λ forced to
   λ(2), and so on. If quality drops, the network depends on the rate
   functionally; if it does not, the separation is redundant
   reparameterisation. Needs a λ-override option in `verify_varnet.py`.
2. [x] *One verification job:* per-rate scores for the DPI checkpoint (block A,
   supervisor item 2). Done 2026-09-28, `verify-dpi-brain`, cluster 11380823.
   Paired per-volume against `verify-model2-brain` (460 identical volumes and
   masks), DPI − blind:

   | rate | SSIM | PSNR | DPI better on |
   |---|---|---|---|
   | R2 | +0.00059 | +0.253 dB | 460/460 (PSNR) |
   | R4 | +0.00015 | +0.039 dB | 353/460 |
   | R6 | −0.00019 | −0.047 dB | 146/460 |
   | R8 | −0.00018 | −0.023 dB | 210/460 |
   | mixed | +0.00008 | +0.051 dB | 289/460 |

   DPI scores 0.9741 / 0.9573 / 0.9465 / 0.9375 SSIM at R2/4/6/8. So the tie
   does hide a redistribution: DPI is better at low rates and slightly worse
   at high rates. The paired t-statistics are large (R2 SSIM t=43), but they
   only measure volume-to-volume noise. With one training seed per model,
   seed-to-seed variation is not measured and could be as large (block D).
3. [x] *(Done 2026-09-30, `blind r8 brain`, 50/50: it leads the joint blind
   model by ~+0.22 dB at R8, unpaired; see `RESULTS.md`.)*
   *One training job, the ceiling:* a blind specialist at R=8 only
   (`training/make submit-model1 ARGS='accelerations=8 center_fractions=0.04 ...'`).
   If it does not beat the joint blind model at R=8, no conditioning method
   can, at this data scale — that is a result, and it reframes block C too.
4. *One training job:* warm-started `full` (`INIT=`, plan.md priority 1)
   with `lambda_lr=1e-2`. Its original motive (frozen λ from unseparated
   sets) is gone per item 1; what remains is that from scratch each endpoint
   effectively trains on only part of the data, and warm start gives both
   endpoints the full-data solution first. Lower priority than 2 and 3.
5. Scopes only after 1 and 3 say there is something to make cheaper.

### 2026-09-30: the repo was split across two clones (resolved by hand)

- Two clones existed: `C:\Users\aryan\fall26research\MRIParameterInterpolation`
  held the 2026-09-28 `--dpi_scope` work (block B, 13 files) **uncommitted**, and
  `C:\Users\aryan\OneDrive\Desktop\fall26research\MRIParameterInterpolation`
  held the 2026-09-29 results commits (`4e58b4c`, `3d5c689`, `f13c0a1`)
  **unpushed**. Each lacked the other's work; the 2026-09-29 session searched
  the OneDrive clone and wrongly concluded `--dpi_scope` did not exist. This
  file is the three-way merge of both. Use the non-OneDrive clone from here on.
- The CHTC checkout is `~/MRIParameterInterpolation` (not `~/Fall26Research`).
  It only gets `--dpi_scope` after the scope commit is pushed and pulled there.
- `--dpi_scope` tests were not re-run on 2026-09-30 (no pytest/Docker on the
  laptop that day); last green run is the 2026-09-28 local gate. UNVERIFIED on a
  GPU slot.
- Given the lambda-saturation finding, a scope variant (`io`, `dc`) is a cost
  ablation, not a fix for the R6/R8 loss; expect it to tie `full`. It is what
  supervisor item 3 asked for, so it is being submitted anyway. 50 epochs of a
  DPI run took 101.7 h for `full`, so nothing submitted on 2026-09-30 finishes
  before the meeting: report partial W&B curves at matched epochs.
- State at end of the 2026-09-30 session (UNVERIFIED unless stated): the
  commands to push the OneDrive commits, commit the scope work, rebase and land
  this file were handed to the user, not run by Claude (the permission system
  blocked the commit/push; CHTC rejects Claude's own ssh, since each login needs
  password + Duo and ControlMaster is broken on Windows). Next on CHTC, after
  the push: `git stash -u && git pull` in `~/MRIParameterInterpolation`, then
  `make submit SCOPE=io` and `make submit SCOPE=dc` in `dpi/`, and the
  `blind r8 brain` resume if it was not resubmitted on 2026-09-29. Whether
  any of these ran is UNVERIFIED; check `condor_q` and W&B.
- Parameter framing, confirmed from `plan.md` section 2 on 2026-09-30: selective
  scopes train fewer parameters than `full` DPI (59.88M), not fewer than the
  blind baseline (29.94M): `shallow` 30.88M, `io` 29.98M, `dc` 29.94M + 1,012.
  "+0.14%" means relative to blind VarNet. Open question to the supervisor:
  does "selective DPI" mean DPI on a subset of layers (the scopes, built), or
  freezing the finished blind VarNet and training only the DPI copies (fewer
  trainable parameters than the baseline)? No freeze option exists in `dpi/`
  as of 2026-09-30; it would be a small addition plus tests.

### Since 2026-09-24

- **Item 2 is done** — per-rate results in `RESULTS.md`, regenerable with
  `verification/paired_from_wandb.py`. Headline: DPI makes a rate-dependent
  trade rather than a uniform gain, and the learned lambda's saturation
  explains it, which reorders block B toward the lambda reparameterisation.
- `blind r8 brain` (cluster 11380824, the per-rate ceiling baseline,
  accelerations [8]) **stopped at epoch 20/50 after 11.80 h, killed by SIGTERM**
  (`ExitBySignal=true`, `ExitSignal=15`, `NumJobStarts=1`). Not a crash and not
  an eviction: val loss fell monotonically to 0.06671 by epoch 18, the `.err`
  ends clean, and memory (39.2/48 GB) and disk (405/520 GB) were both inside
  their requests. `last.ckpt` is intact, so it is resumable.
  **Repo bug this exposed:** `train.sub`'s `on_exit_hold = (ExitBySignal ==
  False) && (ExitCode == 4)` is false for a signal kill, so a runtime-cap
  SIGTERM neither holds nor requeues the job - it just leaves the queue as
  "Completed" with no resubmission. The eviction-resume contract does not cover
  this case. Needs a policy for signal exits; candidate
  `on_exit_remove = (ExitBySignal == False)` plus a `periodic_hold` on
  `NumJobStarts`, tested on a short run first. **Cause resolved: the slot was
  preempted.** `GPUJobLength="long"` and `WantGPULab=true` both reached the ad
  and it ran on `gpu5000.chtc.wisc.edu`, so no runtime class was exceeded -
  "long" is 7 days of eligibility, not immunity. HTCondor vacates by SIGTERM
  (`run_train.sh:138`); the trap fired, output transferred, and the exit-by-
  signal made HTCondor log 005 rather than 004, masking the vacate. At
  ~34 min/epoch the remaining 30 epochs need ~17 h, so expect 2-3 resubmissions
  (~30 min input transfer each) until the requeue rule lands.
  When resuming, take the argument list from `condor_history 11380824 -af Args`:
  `model1` defaults to `accelerations=4 center_fractions=0.08`, and the rate
  config comes from arguments, not from the checkpoint. Detail in `RESULTS.md`.
- Claude cannot log in to CHTC by itself (password + Duo). **On the Mac
  (since 2026-10-01)** `~/.ssh/config` has a `chtc` host with ControlMaster
  (8 h persist). The user runs `ssh chtc` once in a terminal, and Claude then
  runs `ssh -o BatchMode=yes chtc '...'` over that connection
  (`ssh -O check chtc` tests it). On Windows, ControlMaster must not be used:
  mux is broken in Git Bash OpenSSH and silently breaks logins.

### Answer to item 1 (as drafted for the email)

- The reported number is on the **official fastMRI brain multicoil
  validation split**: NYU's `brain_multicoil_val_batch_0.tar.xz`, 460 of the
  1,378 validation volumes, unmodified files, scored with fastMRI's own
  `fastmri.evaluate` code (SSIM per slice averaged over the volume, PSNR and
  NMSE per volume, on the RSS target with the standard centre crop). That is
  the data and the metric code the fastMRI papers and leaderboard used.
- It is **not** the benchmark's test split. The leaderboard test split has
  no public leaderboard any more (fastmri.org moved to NYU in April 2023 and
  the boards were not rebuilt), so no one can score on it today. The brain
  test split's ground truth was released later (`brain_multicoil_test_full`,
  in our group directory), so a test-split number in the sense of the E2E
  VarNet paper's Table 3 is computable here once we want it.
- Two things make the absolute number not comparable to published tables:
  training used one NYU batch (455 of 4,469 training volumes), and the masks
  are `equispaced_fraction` (realised rate equals the nominal R) rather than
  the leaderboard script's uncorrected `equispaced`. The validation mixture
  Lightning logs is one rate drawn per volume; per-rate numbers are item 2.
- All models in the comparison (blind VarNet, DPI, the SDUM baseline) train
  and validate on the same two batches with the same per-volume mask seed, so
  the comparisons are paired even though the absolute numbers are not the
  paper's.

## Previous status (2026-09-18)

**The dataset changed.** Group-directory access came through and the full
fastMRI **brain** multicoil set (20 NYU tarballs, ~1.37 TB, compressed) is in
`/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/`,
transferred and SHA256-verified by `verification/prepare_brain_staging.sh`.
The 2026-09-18 meeting set the training target: **E2E VarNet on multicoil
brain, acceleration rates 2, 4, 6, 8**, with the training configuration and
the mask generation checked against the paper. `training/` was repointed the
same day; the knee subsets in personal staging are superseded but still
reachable through `dataset=knee` macros.

| Piece | State |
|---|---|
| `training/` (`train.sub`, `run_train.sh`, `train_wandb.py`, `submit.sh`) | repointed to brain batch 0 of each split, group staging (listing confirmed 2026-09-18), `runs/brain/<model>/`; model2 logs to the W&B run **"mixed acceleration brain"** (id `mixed-acceleration-brain`) by convention, `NAME='...'` / `name=` to choose, model1 to `varnet-brain-model1`; the key comes from the repo-root `.env`; a missing or rejected key is caught before extraction and **holds the job** (exit 4, `on_exit_hold`), so nothing trains without W&B unless `OFFLINE=1`; defaults now 12 cascades / Adam 3e-4 (paper + leaderboard script) instead of the demo's 8 / 1e-3; smoke-tested locally on synthetic data. **Not yet submitted in this form.** |
| Configuration audit (meeting item 1) | `training/README.md` "Configuration": every setting against the paper, the leaderboard script, and the demo. Batch size and LR schedule are not in the paper; effective batch 1 vs the leaderboard's 32 is the one real deviation |
| Mask audit (meeting item 3) | `training/README.md` "Masks": equispaced lines with density correction, rate drawn uniformly per training slice, per volume at validation, no rate input to the network |
| Gain hypothesis (meeting item 4) | `plan.md` "Expected gain from explicit rate conditioning" |
| Docker images | **must be rebuilt and pushed as `genjigod/fastmri-train:2026-09-18` and `genjigod/fastmri-verify:2026-09-18`** (both Dockerfiles now pin `wandb==0.26.1`). Found 2026-09-18 by the new preflight: the `2026-09` images' wandb 0.18.7 rejects the 86-character W&B key in `.env` on length; the key itself verifies with wandb 0.26.1. Both `.sub` files already name the new tags |
| `verification/` | wandb pin and image tag bumped with training's; at the time could not score a DPI checkpoint and its job executable only matched knee tarball names. Both fixed 2026-09-25 (Phase C block A) |
| Knee subsets in `/staging/a/apryan3/fastmri/` | built 2026-09-12..14 (`knee_multicoil_{train,val}_subset`), used by short test jobs only, superseded |
| **Phase B (DPI)** | **implemented in `dpi/` (2026-09-19).** VarNet with two parameter sets per learnable tensor and a learnable monotone lambda(R), following arXiv:2511.21028 eq. (2) and section 3.2. 21 unit tests plus four smoke targets green on synthetic data; parameter count exactly 2x the baseline plus the 1,000-entry phi. The training setup is the baseline's by construction: `train_dpi.py` calls `../training/train_wandb.py`'s `cli_main` with the model and transform hooks swapped, and the job reuses `../training/run_train.sh`. **Not yet submitted on CHTC.** Runbook: `dpi/README.md` |

The immediate next actions: on the laptop, push the rebuilt images
(`make push IMAGE=genjigod/fastmri-train:2026-09-18` in `training/`, same
for `verification/`); then a CHTC session: `git pull` in `~/Fall26Research`,
`ls -la` the brain directory to size `request_disk`, the 4.1 short test job
(which now also proves W&B end to end), then `make submit-model2`. Runbook:
`training/README.md` sections 1, 3a and 4. The DPI run (`cd dpi && make
submit`) uses the same image and can be queued alongside or after it; the
two are compared per rate, which needs the verification harness taught to
load a DPI checkpoint.

<details>
<summary>Previous status block (2026-09-11), kept for the record</summary>

Everything below is built and green locally. **Nothing has run on CHTC yet** — no
job has been submitted, no tier of `VERIFICATION.md` has been executed, no
training has started, and `/staging` has not been confirmed to hold any fastMRI
data.

| Piece | State |
|---|---|
| `verification/` harness (Tiers 0 + 1), Docker image, submit files, local `make local-run` | written, smoke-tested on synthetic data |
| `training/` driver + job executable + `train.sub`, local `make smoke` / `make job-smoke` | written, smoke-tested on synthetic data, checkpoint/resume contract asserted |
| Staging repack (`make_subset.sh`, `subset_val.sub`, `subset_train.sub`) | written, exercised on synthetic archives, **never run on CHTC** |
| `train.sub` image macro | set to `docker://genjigod/fastmri-train:2026-09` |
| `verify.sub` image macro | still `CHANGE_ME` — edit before the first verification submit |
| Data in `/staging/a/apryan3/fastmri/` | unconfirmed; assume nothing is there until `ls` says otherwise |
| Phase B (DPI) | designed in `plan.md`, no code written |

</details>

---

## Security note — do this first

Your previous CHTC project (`Research/` on your Desktop, the diffusion model collapse work) has a **live-looking Weights & Biases API key hardcoded in `inpainting.sub`**, committed to git history and pushed to the public repo `alexzhangryan/Diffusion-Model-Collapse`. Rotate that key in your W&B account settings before starting new work. Going forward, don't write secrets into `.sub` files that get committed — inject them via an untracked local env file or CHTC's environment-injection mechanisms instead.

---

## Definition of done for Phase A this week

- [x] fastMRI knee dataset access requested — approved week of 2026-09-08, presigned URLs in hand
- [x] CHTC GPU environment set up and reproducible — two Docker images (`verification/Dockerfile`, `training/Dockerfile`, both pinned to fastMRI `91f2df4`, Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, `linux/amd64`), submit files, and runbooks. Not yet exercised on a CHTC slot
- [x] Training runs end-to-end on a small local subset (smoke test) — `training/make smoke` (the driver) and `training/make job-smoke` (the job executable as HTCondor runs it, twice, to assert resume-after-eviction), both on synthetic phantoms
- [x] A real training job is submitted to CHTC and producing checkpoints - both baseline runs (model1, model2) trained to completion on brain, confirmed 2026-09-28
- [x] Known deviations from the paper's setup are documented — `VERIFICATION.md` sections 1 and 5.4, `training/README.md` "Known deviations from Sriram et al. 2020". The reduced-subset deviation (section 3b) is the biggest one and is new as of 2026-09-11

## Reusing your proven CHTC recipe

Your diffusion-collapse project already has a working CHTC pattern — reuse it rather than starting from generic CHTC docs (which point new users at Apptainer; you have something proven already):

- **Container:** `universe = container`, `container_image = docker://<dockerhub_user>/<image>:latest` — build locally, `docker push` to Docker Hub, reference it in the `.sub` file
- **Access point:** `ap2001.chtc.wisc.edu`
- **Submit wrapper:** a `submit.sh` that runs `mkdir -p output snapshots logs` before `condor_submit`, so required directories always exist
- **Resource lines to adapt from `Research/diffusion.sub`:** you previously used `request_cpus=4 request_memory=16GB request_disk=50GB request_gpus=1` with `requirements = (Target.CUDACapability >= 6.0) && (Target.CUDAGlobalMemoryMb >= 8192)` — size these up for VarNet (defaults to `--gpus 2`, multicoil k-space slices are much larger than 32x32 CIFAR images)
- **Real difference this time:** CIFAR-10 fit inside `experiment.tar.gz` via `transfer_input_files`. The fastMRI knee dataset (hundreds of GB) won't — that's why Phase 1 below still routes it through `/staging` rather than bundling it into the job's transferred tarball
- **Logs:** `stream_output = True` / `stream_error = True` worked well before — keep it, `tail -f logs/..._0.out` to watch training live
- **Resume-on-eviction:** your old setup handled this in your own Python code (snapshots per generation). VarNet's Lightning `ModelCheckpoint` + PL's own resume logic does this for you already — confirm it actually picks the latest checkpoint back up after an eviction/requeue before trusting a multi-day run to it

## Observability

You don't have to build this yourself. `fastmri`'s `MriModule` (the Lightning module VarNet's training module extends) already logs, out of the box, every validation epoch:

- scalar metrics: validation loss, NMSE, SSIM, PSNR
- up to 16 example reconstruction images (target / reconstruction / error map)
- a `ModelCheckpoint` callback tracking the best `validation_loss`

All of it goes to **TensorBoard by default** — PyTorch Lightning's default logger when `Trainer` isn't given one explicitly. Nothing to configure to get a first look at training curves and sample reconstructions.

If you want continuity with your diffusion-collapse project's workflow (which logged to Weights & Biases): pass a `WandbLogger` into the `Trainer` in `train_varnet_demo.py`, or call `wandb.init`/`wandb.log` manually the way `inpainting.py` did — using a freshly rotated key, injected as an environment variable, never hardcoded in the `.sub` file.

---

## Phase 0 — Dataset access (done)

- [x] Applied at https://fastmri.med.nyu.edu/ and accepted the **knee** Data Sharing Agreement (knee is what the original VarNet paper and `train_varnet_demo.py` defaults are built around) — approved week of 2026-09-08
- [x] Asked about a lab copy: the group has `/staging/groups/kamilov_group/Kamilov-SciAI-datasets`, but this account cannot read it as of 2026-09-10. Worth pursuing anyway — it is the only way the training split fits (see Phase 2)

The presigned S3 URLs in the approval email are credentials and expire roughly
90 days after issue (this batch around 2026-12-08). Re-request from
fastmri@med.nyu.edu if they lapse before the data is staged.

## Phase 1 — CHTC environment (do in parallel, don't wait on Phase 0)

- [ ] Confirm your CHTC account still works at `ap2001.chtc.wisc.edu` (same account as the diffusion-collapse work) or file a new account request if this is a separate allocation
- [x] fastMRI vendored as a submodule at commit `91f2df4` (`fastMRI/`), never modified. The DPI reference code is a second submodule, `parameter_interpolation/`
- [x] Docker images built for both stages, `linux/amd64`, conda `h5py` (pip's leaks over a multi-day run), fastMRI installed at the pinned commit, and a build-time self-check that asserts Lightning is 1.x. **Push to Docker Hub still to confirm**; `train.sub` already points at `docker://genjigod/fastmri-train:2026-09`, `verify.sub` still says `CHANGE_ME`
- [x] Submit files written: `training/train.sub` (container universe, 1 GPU, `ON_EXIT_OR_EVICT` so an evicted job resumes from its own `output/`), `verification/verify.sub` + `verify_tier0.sub`, and the two one-off repack jobs `training/subset_val.sub` / `subset_train.sub`. Each has a `submit.sh` wrapper that creates the output directories and refuses to submit without a W&B key unless `OFFLINE=1`
- [x] `/staging` confirmed working on `ap2001`: **`/staging/a/apryan3/`** (personal staging, sharded by the netid's first letter). `/home` is 40 GB and holds code only — the dataset was never going to fit there. The shared `/staging/groups/kamilov_group/Kamilov-SciAI-datasets` is not readable by this account

## Phase 2 — Data transfer

- [x] **Brain (2026-09-16..18, the current data).** Group-directory access
  granted. `verification/prepare_brain_staging.sh` fetched all 20 NYU brain
  tarballs (~1.37 TB: 10 train batches, 3 val, 3 test, 3 fully-sampled test,
  DICOM) into `/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/`
  and verified them against NYU's `SHA256`. Nothing is extracted there (10,000-item
  cap); jobs pull one `.tar.xz` per split into scratch. `train.sub` defaults to
  batch 0 of train and val.

The knee material below is what happened before that and is kept for the
record; the knee subsets still exist in personal staging.

Target (knee): **`/staging/a/apryan3/fastmri/`**. Archives stay packed in `/staging`;
HTCondor transfers them into job scratch, where `run_verify.sh` / `run_train.sh`
extract them. A job never reads `/staging` directly.

NYU ships each split as one or more large `.tar.xz`, and **neither full split
fits the 100 GB quota**, so there *is* repackaging to do — see the decision
below and `training/README.md` section 3b.

- [ ] Fetch `knee_multicoil_val.tar.xz` (93.8 GB) straight from the NYU presigned URL onto CHTC with `verification/prepare_staging.sh`, run on `transfer.chtc.wisc.edu` under tmux with `PARALLEL=1..4`
- [ ] Do **not** scp the desktop copy up: that connection is per-flow shaped, so a 94 GB upload takes days where an S3-to-campus fetch takes hours
- [ ] Verify against NYU's `SHA256` manifest — `prepare_staging.sh` does this automatically and is idempotent and resumable
- [ ] Sanity-check that a handful of files load correctly with `h5py` before trusting the full set
- [x] **Decision 2026-09-11: do not request a quota increase (last resort).** Instead repack `/staging` into subsets that fit: `training/README.md` section 3b (`make subset-val`, then delete the full val tarball, then `make subset-train`, which streams a ~65 GB prefix of NYU's `knee_multicoil_train_batch_0` (the split ships as five ~91 GB batches) straight into `/staging`). `train.sub` now defaults to the subset names. Both jobs and `run_train.sh` were tested end to end on synthetic archives in the training image; not yet run on CHTC
- [ ] **Quota gate — superseded by the line above unless the full splits are ever wanted:** `/staging/a/apryan3` is capped at 100 GB (confirmed 2026-09-10) and the val tarball is 100,694,526,932 bytes, i.e. 93.8 GiB or 100.7 GB decimal. It fits only under binary counting and leaves nothing over either way. Request an increase before the transfer; `multicoil_train` (~931 GB unpacked) needs one regardless, as does group-directory access if that comes through

## Phase 3 — Get training actually running

- [ ] Generate `fastmri_dirs.yaml` (or pass `--data_path` directly) pointing at the data — inside a job that means the copy HTCondor transferred into scratch, not `/staging` itself
- [ ] **Smoke test first**, locally or in a short job, on a tiny subset (10-20 files) before submitting a real GPU job — catches path/env bugs without burning queue time
- [ ] Submit the real job. This is now `training/submit.sh`, not a bare
  `train_varnet_demo.py` invocation: `train_wandb.py` builds the same
  `VarNetModule` / `FastMriDataModule` and adds the W&B logger and the
  auto-resume-from-`output/checkpoints` contract a multi-day CHTC run needs.
  **Since 2026-09-18: brain data, and model2 is the primary run:**
  ```bash
  # on ap2001, in ~/Fall26Research/training, WANDB_API_KEY exported
  make submit-model2 ARGS='max_epochs=1 extra_args="--limit_train_batches 50 --limit_val_batches 10"'   # short test first
  make submit-model2   # brain, accelerations 2 4 6 8, center_fractions 0.16 0.08 0.0533 0.04
  make submit-model1   # brain, acceleration 4 only (optional per-rate reference)
  ```
  Configuration: 12 cascades, 18 / 8 channels, Adam 3e-4 with x0.1 at epoch
  40, batch 1, 50 epochs, `equispaced_fraction`, seed 42, one GPU (the paper
  and the fastMRI brain leaderboard script; the demo's 8 cascades / 1e-3 were
  dropped 2026-09-18). Data: `brain_multicoil_train_batch_0` (455 volumes) and
  `brain_multicoil_val_batch_0` (460 volumes). model2 is the
  joint-training-no-conditioning Phase B baseline; model1 is one point of the
  per-rate ceiling. Full runbook: `training/README.md` section 4.
- [x] **Known deviations from the paper documented** — the register lives in
  `training/README.md` ("Known deviations from Sriram et al. 2020") and
  `VERIFICATION.md` sections 1 and 5.4. In short:
  - variable `center_fractions` rather than the paper's fixed center-line counts (29/15 at 368 wide vs 30/16)
  - model2 trains jointly across four rates rather than one model per rate, with no explicit conditioning signal — exactly the gap Phase B targets
  - `equispaced_fraction` is the corrected-density family and matches neither paper table; `random` is the knee-leaderboard convention
  - the released checkpoint's leaderboard numbers combined `train`+`val`, so `multicoil_val` is training data for it — fine for pipeline verification (Claim A), useless as a generalization estimate
  - **and now the big one:** both models train on the section-3b subsets (~120 of 973 train volumes, 20 of 199 val volumes), so absolute numbers are not comparable to the paper at all. Every Phase A/B comparison must use the same two subsets
- [ ] A 50-epoch multicoil run will very likely exceed a single GPU job's runtime cap — confirm checkpoint resume actually works before relying on it across multiple resubmissions

## Phase 4 — Monitor and sanity-check

- [ ] Watch TensorBoard (or W&B, if wired up) as checkpoints land; compare early-epoch behavior against fastMRI leaderboard VarNet numbers to catch a broken setup early
- [ ] Keep a running log (job IDs, GPU type assigned, wall-clock per epoch)

---

## Phase B (forward-looking): Deep Parameter Interpolation for acceleration-rate conditioning

Not this week's work — capturing it now so it isn't lost.

The DPI paper (Park et al., CVPR 2026) conditions diffusion/flow-matching models on a scalar (noise level or timestep) by keeping **two learnable parameter sets per layer** and interpolating between them with a learnable monotonic function of the scalar, implemented as a cumulative softmax (guaranteed monotonic, mapped to [0,1]). It's only demonstrated on image generation (FFHQ, diffusion + flow matching) — applying it to MRI acceleration-rate conditioning is a genuinely new extension, not something the paper covers.

Rough plan to adapt it to VarNet:
- Replace each cascade's learnable parameters (conv/linear weights) with paired sets `(ω0, ω1)`
- Add the learnable monotonic function mapping acceleration rate → interpolation weight in [0,1]
- Thread acceleration rate through as an explicit scalar input — VarNet currently only "sees" it implicitly via the undersampling mask, there's no explicit conditioning signal today
- Interpolate parameters at both train and inference time based on the current sample's acceleration rate
- Train jointly across multiple rates (4x and 8x, same as the baseline default, but now with explicit conditioning)
- Baselines: VarNet trained separately per rate (expensive, a ceiling), the current joint-training-no-conditioning baseline from Phase A, and a simpler conditioning baseline like FiLM-style embedding of the rate — mirroring the baseline set DPI's own paper used (NCSNv2 rescaling, constant maps, MLP embedding conditioning)

---

## Phase C (2026-09-24): per-rate reporting, DPI variants, SDUM baseline

Ordered so that the per-rate report (item 2) is ready for next week's
meeting while the two longer blocks run.

### Block A: per-rate performance (item 2, report next week)

- [x] (2026-09-25) `verification/` scores a DPI checkpoint and runs on the
  brain data: `verify_varnet.py` detects `lambda_table.phi`, rebuilds
  `DPIVarNet` from the checkpoint's `hyper_parameters` and passes each
  pass's nominal rate into the forward; `run_verify.sh` accepts NYU's
  `brain_multicoil_val_batch_0.tar.xz` (its glob was knee-only and would
  have exited after the 100 GB transfer); `verify.sub` defaults to brain
  val batch 0 in group staging, `request_disk = 320GB`, transfers
  `../dpi/dpi_varnet.py`, and files output under `runs/brain/<model>/`.
  Also fixed: a Lightning checkpoint's `loss.w` buffer reached the strict
  load, so scoring any trained checkpoint (not only DPI) would have failed.
  Proven by `make smoke-ckpt` and `make job-smoke` on toy checkpoints of
  both kinds; the real command is `verification/README.md` section 4.4
- [x] (2026-09-28/29) Scored the baseline (`mixed acceleration brain`) and the
  first DPI checkpoint: W&B runs `verify-model2-brain` and `verify-dpi-brain`,
  460 volumes / 7270 slices of brain val batch 0, every volume forced to
  R = 2, 4, 6, 8 plus the mixed pass
- [x] (2026-09-29) **Reported in `RESULTS.md`** — one table per metric, the
  paired per-volume differences DPI minus baseline at each rate per
  `VERIFICATION.md` section 6.3 (Wilcoxon signed-rank, median difference and
  bootstrap interval alongside the mean-of-means), and the zero-filled row.
  Regenerate for any new variant with
  `python verification/paired_from_wandb.py --treatment <run> --markdown`,
  which reads the `R<N>/per_volume` tables straight from W&B — no run outputs
  on disk and no `wandb` package needed
- [x] (2026-09-29) Caveat stated in `RESULTS.md`: one seed per model, sigma
  unknown, everything provisional under the eventual 2-sigma band
  (`VERIFICATION.md` section 6.2). The paired test controls volume difficulty
  and mask seed but NOT training-seed variance, and these effects are small
  enough that block D could erase or reverse them
- [x] (2026-09-29) Sanity invariants pass: SSIM falls monotonically 2x->8x for
  both models (baseline 0.973530 > 0.957180 > 0.946674 > 0.937724; DPI
  0.974122 > 0.957327 > 0.946489 > 0.937547), and `lambda/R4` = 0.657,
  `lambda/R6` = 0.847 moved off the initial line

**Result (2026-09-29): DPI makes a rate-dependent trade, not a wash.** It wins
at R2 (+0.253 dB PSNR, 460/460 volumes) and R4 (+0.039 dB), loses at R6
(-0.047 dB) and R8 (-0.023 dB); every direction is significant except R8 NMSE
(p = 0.063). The mixed pass reads only +0.051 dB because the gains and losses
nearly cancel. **The learned lambda explains it:** lambda at R2/R4/R6/R8 =
0.000 / 0.657 / 0.847 / 1.000 — two thirds of the range is spent between R2 and
R4, and R6/R8 are compressed into the top 15%, so DPI wins where lambda is
spread out and loses where it saturates. The cumulative softmax guarantees
monotonicity but nothing constrains spacing. **This reorders block B: the
lambda reparameterisation attacks the measured cause, whereas the scope
ablations mainly attack the 3.9x cost (101.7 h vs 26.4 h).** Full numbers,
method and caveats: `RESULTS.md`.

### Block B: DPI variants (item 3)

Grid and priority in `plan.md`, "DPI variant grid (2026-09-24)". Code
change needed first: re-add `--dpi_scope {full, shallow, io, dc, none}` to
`dpi/` as designed in `plan.md` sections 1 and 2 (the implementation kept
only `full` plus `--no_dpi_sens`). The unit tests already listed there
(equivalence at init per scope, parameter counts per scope, gradient flow)
extend the existing 21.

- [x] (2026-09-28) `--dpi_scope {full,shallow,io,dc,none}` in
  `dpi_varnet.py` / `dpi_module.py` / `train_dpi.py`, with per-scope
  parameter-count tests matching the table in `plan.md` section 2 exactly
  (checked against hand-derived arithmetic on the real 12-cascade config:
  `full` 29,937,966, `shallow` 946,094 = 3.16%, `io` 41,086 = 0.137%, `dc`
  1,012, `none` 0). Also per-scope equivalence-at-init, state-dict
  compatibility and gradient-flow tests; a `make scope-smoke` target that
  trains every scope through the CLI and greps the log for the model
  reporting the scope it actually built; `scope=io` in `dpi/submit.sh`, which
  routes a variant to its own `runs/brain/dpi-<scope>/`, log files and W&B
  run, and refuses to submit if `--dpi_scope` did not reach the job ad.
  `verification/verify_varnet.py` rebuilds a scoped checkpoint and reads
  `dpi_scope` with a `"full"` default, so the DPI checkpoints written before
  the flag existed still score; `make smoke-ckpt` now covers a scoped and a
  pre-flag checkpoint.
  **Local gate green (2026-09-28):** `dpi` `make local-run` (`test` 49
  passed, `smoke`, `scope-smoke` — all five scopes train and checkpoint
  through the CLI — `job-smoke`, `warm-smoke`); `verification`: the four toy
  checkpoints — baseline, DPI full, DPI io, and one written without a
  `dpi_scope` hyper-parameter — all load, rebuild with the right scope and
  pass the determinism invariant. `job-smoke` covers the scope under
  `training/run_train.sh`, the executable HTCondor runs, fresh and after a
  simulated eviction. Not yet run on a GPU slot. `full` is the default and is
  what every existing checkpoint is, so nothing already trained or queued
  changes. No image rebuild needed — the DPI code is transferred with the job;
  both `genjigod/fastmri-{train,verify}:2026-09-18` confirmed present on
  Docker Hub 2026-09-28, so the `.sub` image tags resolve.
  **Bug fixed on the way through:** the `[dpi] ...` parameter-summary line in
  `on_fit_start` used `self.print`, which routes through
  `TQDMProgressBar.print`; that method silently discards the message when no
  progress bar is active, which is always true at fit start. The line has
  therefore never appeared in any job log since 2026-09-19, including the DPI
  runs already on the cluster — so there is no log record of which scope or
  parameter count those runs used (the checkpoints themselves still say, via
  `hyper_parameters`). Now `rank_zero_info`, and `make scope-smoke` greps for
  it. Training behaviour was never affected.
- [ ] Submit in priority order (each is a 50-epoch model2-sized run, one GPU):
  1. `full`, warm-started from the blind checkpoint (`INIT=...`)
  2. [~] `io` (the light variant, +0.14% parameters): cluster 11442790, at
     epoch 40/50 on 2026-10-01, due ~2026-10-02
  3. [x] `dc` (one scalar per cascade, the cheapest test of the hypothesis):
     cluster 11442791, finished 2026-10-01; ties `full` on mixed val
     (`RESULTS.md`). Scored per rate 2026-10-01 (`verify-dpi-dc-brain`,
     cluster 11926920): beats blind at R2/4/6/8 and mixed, no R6/R8 loss
  4. `full --no_dpi_sens`
  5. `shallow`, only if `io` is clearly below `full`
  6. `--accel_min 4 --accel_max 8` on `full` (two independent sets, the
     per-rate ceiling inside one model)
- [ ] Every variant goes through block A's scoring; the table grows a row per
  variant

### Block C: SDUM baseline at matched parameters (item 4)

Design in `plan.md`, "Matched-parameter SDUM baseline (NV-Raw2insights-MRI)".
The chosen architecture is their 12-cascade Restormer at widths 64/128 with
their universal conditioning on, 29.93M parameters against VarNet's 29.94M;
its conditioning-off twin is the second row.

- [ ] `sdum/` sibling directory: port `restormer_mri` and
  `Cascaded_SkipConnected_MRI_Recon` (Apache-2.0) behind a fastMRI-style
  `forward(masked_kspace, mask, num_low_frequencies, acceleration)`, as a
  third `build_model` hook of `training/train_wandb.py`, reusing
  `run_train.sh`. Data, masks, loss, optimiser, epochs, seed and W&B stay the
  baseline's by construction, exactly as `dpi/` does it
- [ ] Add `monai`, `timm`, `einops` to `training/Dockerfile` (new image tag);
  the verification image gets the same so block A can score it
- [ ] **Correctness test against their code** (the supervisor's explicit
  ask): (i) instantiate the port at their released `small` configuration and
  assert its `state_dict` keys and shapes equal the released
  `nv_raw2insights_mri_small` checkpoint's; (ii) load those weights into both
  their unmodified `scripts/inference.py` path and the port, run the example
  case that ships in their repo through both, and assert the outputs match
  to float tolerance; (iii) on a fastMRI slice, assert the port's ACS region
  and z-score normalisation equal theirs. (i) and (ii) are what proves the
  port is their model and not a look-alike
- [ ] Submit `sdum-uc` (conditioning on) and `sdum-blind` (conditioning off),
  model2 rate list, from scratch, 50 epochs, Adam 3e-4 batch 1 (the
  baseline's optimiser, stated deviation from their Muon; see `plan.md`)
- [ ] Score both through block A; add to the per-rate table. The comparison
  the paper needs: VarNet-blind vs SDUM-blind (architecture), SDUM-uc vs
  SDUM-blind (their embedding-style conditioning), DPI vs SDUM-uc
  (conditioning mechanism at equal parameters), and the unseen-rate probe at
  3x / 5x / 10x, where SDUM's label lookup has no index for an unseen rate
  and DPI's lambda interpolates

### Block D: seed variance (gates every claim above)

- [ ] Baseline model2 at seeds 1337 and 2024 in addition to 42
  (`VERIFICATION.md` section 6.1); report sigma per rate. Queue these behind
  blocks B and C on the GPU queue, but before the paper's numbers are final

## Known risks / things likely to slow this down

- fastMRI approval timeline is undocumented — start Phase 0 today regardless of what else is ready
- Dataset size vs the 100 GB `/staging` quota — resolved by repacking into subsets rather than by a quota increase (Phase 2 decision, 2026-09-11). The residual cost: a full Tier 1 run on all 199 val volumes is only possible *before* `make subset-val` deletes the full val tarball, or after re-downloading it from the NYU URL (valid to ~2026-12-08). Run Tier 1 first if that number is wanted
- The subset path has never been exercised on CHTC. The two repack jobs are the riskiest unrun thing in the repo: `subset_train.sub` streams a 65 GB HTTP range request through `xz` in a CPU slot and writes its output straight into `/staging` via `transfer_output_remaps`, which goes on hold if the quota is short
- GPU queue times can be long for high-end tiers; a mid-tier GPU is fine for a first working run
- Job runtime caps mean the real training run will not finish in one submission — expected, not a bug
- Phase B has no existing reference implementation to check against — budget real time for getting the parameter-interpolation mechanics right before trusting any results from it

## Useful links

- Dataset request: https://fastmri.med.nyu.edu/
- Code: https://github.com/facebookresearch/fastMRI (VarNet example: `fastmri_examples/varnet/`)
- E2E VarNet paper: https://arxiv.org/abs/2004.06688
- DPI paper: "Deep Parameter Interpolation for Scalar Conditioning" (Park et al., CVPR 2026)
- CHTC GPU jobs guide: https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs
- CHTC ML workflow guide: https://chtc.cs.wisc.edu/uw-research-computing/machine-learning-htc
- Your prior CHTC recipe (reference): `Research/Dockerfile`, `Research/diffusion.sub`, `Research/submit.sh` on your Desktop

---
*Last updated: 2026-10-05*
