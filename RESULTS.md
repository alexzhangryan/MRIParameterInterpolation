# Per-rate results: DPI vs blind VarNet on fastMRI brain

**Phase C block A, item 2 of the 2026-09-24 supervisor feedback.**
Analysis run 2026-09-29, updated 2026-10-01 with the R8 specialist and the
first two `--dpi_scope` variants (see "Update 2026-10-01" below). Numbers
pulled from W&B, not from this checkout — the repo holds no run outputs.

Reproduce with `verification/paired_from_wandb.py` (needs `WANDB_API_KEY` from
the repo-root `.env`; no `wandb` package required — it uses the GraphQL API and
numpy only).

## Headline

DPI is **not** a wash against the blind baseline, and it is **not** a uniform
win. It makes a **rate-dependent trade**: it buys accuracy at low acceleration
and pays for it at high acceleration.

| | R2 | R4 | R6 | R8 |
|---|---|---|---|---|
| winner | **DPI** | **DPI** | baseline | baseline |
| dPSNR | +0.253 dB | +0.039 dB | -0.047 dB | -0.023 dB |

The aggregate hid this: the mixed-rate pass shows only +0.051 dB, because the
R2/R4 gains nearly cancel the R6/R8 losses.

## Models compared

Both are 12 cascades, 18 chans, Adam 3e-4, batch 1, 50 epochs, seed 42,
`equispaced_fraction` masks, rates 2/4/6/8, trained on brain `multicoil_train`
batch 0 — identical by construction (`dpi/train_dpi.py` calls
`training/train_wandb.py`'s `cli_main` with the model and transform hooks
swapped).

| W&B run | role | val_loss | runtime |
|---|---|---|---|
| `mixed acceleration brain` | blind VarNet, joint training, no rate input | 0.046375 | 26.4 h |
| `dpi mixed acceleration brain` | DPI, learnable monotone lambda(R) | 0.046288 | **101.7 h** |

DPI costs **3.9x** the baseline's wall clock.

Scored by `verify-model2-brain` and `verify-dpi-brain` on NYU
`brain_multicoil_val_batch_0`: **460 volumes / 7270 slices**, every volume
forced to each rate, `fastmri.evaluate` metric code.

## Method

Per `VERIFICATION.md` section 6.3. Volume difficulty dominates the variance in
these metrics, so the comparison is **paired over volumes**: Wilcoxon
signed-rank (normal approximation with tie correction), median paired
difference and its 95% bootstrap interval reported alongside the
mean-of-means, 8000 bootstrap resamples. Both models see identical volumes and
identical per-volume mask seeds, so the pairing is exact.

The zero-filled column is included per block A so the scale of the gains is
visible.

### SSIM

| rate | zero-filled | baseline | DPI | mean diff [95% CI] | median diff [95% CI] | DPI wins | p |
|---|---|---|---|---|---|---|---|
| R2 | 0.917676 | 0.973530 | 0.974122 | +0.000592 [+0.000565, +0.000619] | +0.000536 [+0.000504, +0.000573] | 459/460 | ~0 |
| R4 | 0.796991 | 0.957180 | 0.957327 | +0.000148 [+0.000126, +0.000169] | +0.000146 [+0.000132, +0.000164] | 373/460 | ~0 |
| R6 | 0.726390 | 0.946674 | 0.946489 | -0.000186 [-0.000226, -0.000148] | -0.000122 [-0.000156, -0.000093] | 146/460 | ~0 |
| R8 | 0.685284 | 0.937724 | 0.937547 | -0.000177 [-0.000237, -0.000118] | -0.000065 [-0.000122, -0.000016] | 200/460 | 3.0e-06 |
| mixed | 0.778490 | 0.953415 | 0.953499 | +0.000085 [+0.000034, +0.000132] | +0.000114 [+0.000075, +0.000169] | 283/460 | 5.2e-08 |

### PSNR (dB)

| rate | zero-filled | baseline | DPI | mean diff [95% CI] | median diff [95% CI] | DPI wins | p |
|---|---|---|---|---|---|---|---|
| R2 | 35.505 | 44.028 | 44.281 | +0.253 [+0.238, +0.267] | +0.219 [+0.210, +0.236] | 460/460 | ~0 |
| R4 | 29.454 | 40.354 | 40.393 | +0.039 [+0.033, +0.044] | +0.042 [+0.034, +0.049] | 353/460 | ~0 |
| R6 | 27.025 | 38.169 | 38.123 | -0.047 [-0.055, -0.039] | -0.034 [-0.045, -0.027] | 146/460 | ~0 |
| R8 | 25.860 | 36.557 | 36.534 | -0.023 [-0.035, -0.012] | -0.011 [-0.032, +0.000] | 210/460 | 9.4e-04 |
| mixed | 29.324 | 39.715 | 39.765 | +0.051 [+0.035, +0.066] | +0.032 [+0.022, +0.048] | 289/460 | 1.5e-10 |

### NMSE (lower is better)

| rate | zero-filled | baseline | DPI | mean diff [95% CI] | median diff [95% CI] | DPI wins | p |
|---|---|---|---|---|---|---|---|
| R2 | 0.012045 | 0.001791 | 0.001705 | -0.000086 [-0.000090, -0.000082] | -0.000077 [-0.000079, -0.000073] | 460/460 | ~0 |
| R4 | 0.047816 | 0.004079 | 0.004044 | -0.000035 [-0.000041, -0.000029] | -0.000033 [-0.000039, -0.000027] | 353/460 | ~0 |
| R6 | 0.082846 | 0.006628 | 0.006685 | +0.000057 [+0.000043, +0.000070] | +0.000051 [+0.000035, +0.000063] | 146/460 | ~0 |
| R8 | 0.108154 | 0.009606 | 0.009629 | +0.000023 [-0.000002, +0.000049] | +0.000025 [-0.000000, +0.000056] | 210/460 | **6.3e-02** |
| mixed | 0.064108 | 0.005533 | 0.005529 | -0.000005 [-0.000021, +0.000011] | -0.000037 [-0.000044, -0.000025] | 289/460 | 7.3e-04 |

R8 NMSE is the **only** cell that is not significant at 0.05; its CI spans
zero. Every other direction in every table is significant.

## Why the trade happens: lambda saturates

The learned monotone interpolation weight converged to

| rate | lambda |
|---|---|
| R2 | 0.000 |
| R4 | 0.657 |
| R6 | 0.847 |
| R8 | 1.000 |

lambda spends **66% of its range between R2 and R4**, then compresses R6 and R8
into the top 15%. Low rates get well-separated parameter sets; high rates are
squeezed against the omega-1 endpoint and effectively share one. **DPI wins
exactly where lambda is spread out and loses exactly where lambda saturates.**

The cumulative-softmax construction (arXiv:2511.21028 eq. 2, section 3.2)
guarantees **monotonicity** but nothing constrains **spacing**. That is the
mechanism behind the observed pattern, and it makes the lambda
reparameterisation the highest-information item in block B — it attacks the
measured cause, whereas the scope ablations (`io`, `dc`, `shallow`) mainly
attack the 3.9x cost.

The mechanism is therefore working as designed and is not itself the problem:
lambda is monotone, non-degenerate, and spans the full [0,1]. "Wired correctly,
conditioning does not pay uniformly" is a considerably stronger claim than "DPI
did not work."

## Caveats, in order of how much they should restrain the claim

1. **One seed per model, so sigma is unknown, so all of this is provisional.**
   `VERIFICATION.md` section 6.2: any improvement smaller than roughly 2 sigma
   is not reportable from a single run pair. The paired test controls volume
   difficulty and mask seed — it does **not** control training-seed variance.
   The p-values above say "on these two trained networks, the difference is
   consistent across volumes." They do **not** say "a retrained DPI would beat
   a retrained baseline." **Block D (seeds 1337 and 2024) gates every number
   here**, and these effects are small enough that it could plausibly erase or
   reverse them.
2. **The effects are tiny next to the reconstruction itself.** Zero-filled to
   baseline is +8.5 to +11.1 dB; the DPI delta is 0.02-0.25 dB, i.e. **34x to
   455x smaller**:

   | rate | zf -> baseline | DPI - baseline | ratio |
   |---|---|---|---|
   | R2 | +8.523 dB | +0.253 dB | 34x |
   | R4 | +10.900 dB | +0.039 dB | 279x |
   | R6 | +11.144 dB | -0.047 dB | 239x |
   | R8 | +10.697 dB | -0.023 dB | 455x |

3. **Absolute numbers are not comparable to published tables** — training used
   one NYU batch (455 of 4,469 train volumes) and `equispaced_fraction` masks
   rather than the leaderboard script's uncorrected `equispaced`. The
   comparisons here are paired and internally valid; the absolute values are
   not the paper's. See `ROADMAP.md`, "Answer to item 1".
4. This is validation data, not the benchmark test split.

## Sanity invariants (block A gate) — both pass

- SSIM falls monotonically 2x to 8x for both models:
  baseline 0.973530 > 0.957180 > 0.946674 > 0.937724;
  DPI 0.974122 > 0.957327 > 0.946489 > 0.937547.
- `lambda/R4` and `lambda/R6` moved off the initialisation line (0.657, 0.847).
- Baseline competence (section 6.4): beats zero-filled by 8.5-11.1 dB at every
  rate.

## Update 2026-10-01: R8 specialist and the first scope variants

Read from W&B (`fastmri-varnet-train`) on 2026-10-01 via the `wandb` SDK.
**None of the three runs below has been through the per-rate verification
harness yet**, so every number in this section is Lightning's training-time
validation, not the paired 460-volume scoring above. Lightning's mixed pass
draws one rate per volume and is close to, but not the same as, the harness's
mixed row (blind: 39.734 dB here vs 39.715 dB above).

| W&B run | cluster | state | epochs | wall clock |
|---|---|---|---|---|
| `blind r8 brain` | 11442792 (resume of 11380824) | finished 2026-09-30 | 50/50 | 27.9 h total |
| `dpi dc mixed acceleration brain` | 11442791 | finished 2026-10-01 | 50/50 | 33.0 h |
| `dpi io mixed acceleration brain` | 11442790 | **running** | 40/50 | 37.4 h so far, ~0.93 h/epoch |

### `dc` ties `full` at a fraction of the cost

`--dpi_scope dc` duplicates only each cascade's data-consistency step size
(`dc_weight`, 12 scalars) plus phi: **1,012 extra parameters**, against
`full`'s 29.9M. All 12 U-Nets are shared and rate-blind.

| final epoch (49) | blind | DPI `full` | DPI `dc` |
|---|---|---|---|
| val loss | 0.04637 | 0.04629 | 0.04622 |
| val PSNR (dB) | 39.734 | 39.784 | 39.771 |
| val SSIM | 0.95355 | 0.95363 | 0.95370 |
| parameters | 29.94M | 59.88M | 29.94M + 1,012 |
| wall clock | 26.4 h | 101.7 h | 33.0 h |
| lambda(R4), lambda(R6) | — | 0.656, 0.847 | **0.492, 0.776** |

- **The claim this supports is "`dc` matches `full`", not "`dc` beats
  `full`".** The three runs sit within 0.00015 in val loss. Before the LR drop
  at epoch 40, the same run moves by up to ~0.001 between consecutive epochs
  (`full` 0.04762 -> 0.04805 at 38 -> 39), and there is one seed per model.
- The same numbers admit a weaker reading: on the mixed metric **neither DPI
  variant is distinguishable from blind**, so "dc matches full" may only mean
  that neither does anything there. The per-rate scoring (below) is what tells
  these apart: `full`'s effect only became visible per rate.
- **lambda is less saturated under `dc`**: R4 sits at 0.49 instead of 0.66, so
  R6 and R8 are less compressed against the omega-1 endpoint. Given the
  saturation mechanism above, this predicts a smaller R6/R8 loss for `dc`.
  Untested until `dc` is scored per rate.
- Wall-clock is confounded by hardware: `full` ran on `patel0000`, `dc` on
  `gpu4006`, blind on `gpu5001`. Treat 3.9x vs 1.25x as indicative, not as a
  measured cost ratio.
- There is no `none` control yet (the DPI code path with nothing duplicated,
  which must track blind). Without it, a small `dc` edge cannot be separated
  from a code-path difference.

### R8 specialist: there is headroom at R8 that DPI does not capture

`blind r8 brain` (accelerations [8], center_fractions [0.04]) was resumed from
its checkpoint after the 2026-09-29 preemption and finished all 50
epochs. Its validation is R8 only.

| R8 | val PSNR (dB) | val SSIM | source |
|---|---|---|---|
| specialist (R8 only) | 36.778 | 0.93858 | Lightning val, epoch 49 |
| blind joint | 36.557 | 0.937724 | `verify-model2-brain`, R8 pass |
| DPI `full` joint | 36.534 | 0.937547 | `verify-dpi-brain`, R8 pass |

The specialist leads the joint blind model by **~+0.22 dB / +0.0009 SSIM at
R8**, while DPI `full` sits at -0.023 dB. So at this data scale joint training
does leave something on the table at R8, and the conditioning has not
recovered it — the opposite of the "no conditioning method can win at R8"
outcome the ceiling run was designed to detect.

**This comparison is not paired.** It sets Lightning's validation loop
against the verification harness, which may draw different masks, so it is
directional only. `verify-blind-r8-brain` (below) makes it paired.

### `io`: matched epochs only

`io` is at epoch 40/50 and should finish around 2026-10-02 00:00 CDT. Its W&B
summary (39.148 dB) is epoch 39, before the LR drop that added +0.1 to +0.3 dB
to every run, so it must not be read against the others' final numbers.

| epoch | blind loss | `full` loss | `dc` loss | `io` loss |
|---|---|---|---|---|
| 30 | 0.04873 | 0.04814 | 0.04901 | 0.04823 |
| 35 | 0.04819 | 0.04777 | 0.04764 | 0.04788 |
| 38 | 0.04768 | 0.04762 | 0.04769 | **0.04729** |
| 39 | 0.04749 | 0.04805 | 0.04810 | 0.04823 |

Tying so far, within epoch-to-epoch noise. lambda at epoch 39: R4 0.557, R6
0.819 — between `dc` and `full`.

### Next: per-rate scoring of all three

From `verification/` on the access point (confirm checkpoint directories with
`ls` first — the resumed R8 run may write under 11380824 or 11442792):

```
make verify MODEL=model2 ARGS='ckpt=../dpi/runs/brain/dpi/11442791/checkpoints/last.ckpt run_name=verify-dpi-dc-brain'
make verify MODEL=model2 ARGS='ckpt=../training/runs/brain/model1/<r8 cluster>/checkpoints/last.ckpt run_name=verify-blind-r8-brain'
make verify MODEL=model2 ARGS='ckpt=../dpi/runs/brain/dpi/11442790/checkpoints/last.ckpt run_name=verify-dpi-io-brain'   # after io finishes
```

Then pair each against `verify-model2-brain` with `paired_from_wandb.py`. For
the specialist only the R8 row is a fair comparison; its R2/R4/R6 rows measure
how a single-rate model degrades off its training rate.

## Resolved: the R8 per-rate ceiling baseline was preempted at epoch 20

Resolved 2026-09-30: resumed as cluster 11442792 and finished 50/50 (results
in "Update 2026-10-01" above). The diagnosis below stands, and so does the
`train.sub` requeue gap it exposed.

`blind r8 brain` (cluster 11380824, accelerations [8], center_fractions [0.04])
**stopped at epoch 20 of 50**.

**Cause, confirmed 2026-09-29 from the job ad — not a crash and not an
eviction:**

```
ExitBySignal = true   ExitSignal = 15 (SIGTERM)   NumJobStarts = 1
RemoteWallClockTime = 42491 s = 11.80 h
RequestMemory 49152 MB / MemoryUsage 39212 MB   (80%, fine)
RequestDisk 545259520 KB / DiskUsage 425000000 KB   (78%, fine)
```

Training was healthy to the last line: validation loss fell monotonically
0.10269 -> 0.06671 by epoch 18 and was still improving; the `.err` log ends
mid-stream after `Epoch 19, global step 144320` with no traceback and no CUDA
error. Memory and disk were both well inside their requests. The job was killed
on a clock at 11.8 h.

**`train.sub` has no requeue path for a signal kill.** Its guard is

```
on_exit_hold = (ExitBySignal == False) && (ExitCode == 4)
```

which is false for a SIGTERM, so the job was neither held nor requeued — it
simply left the queue as JobStatus 4 ("Completed"). `when_to_transfer_output =
ON_EXIT_OR_EVICT` did bring `output/` back, so `last.ckpt` and
`epoch=18-step=137104.ckpt` (344 MB each) are intact and a resume is possible;
nothing restarted it automatically. The eviction-resume contract that was tested
in 2026-09-12's `b246100` covers eviction, not a runtime-cap SIGTERM.

**Resolved 2026-09-29: the slot was preempted.** `GPUJobLength = "long"` and
`WantGPULab = true` both reached the job ad, and it ran on a genuine GPU Lab node
(`slot2_1@gpu5000.chtc.wisc.edu`, H200 143 GB), so no runtime class was exceeded -
`long` grants 7 days of eligibility, not immunity from preemption. As
`run_train.sh:138` documents, HTCondor vacates by sending SIGTERM to the
executable; the `on_term` trap fired, output transferred at 06:15:35, and the
job then exited by signal, so HTCondor recorded event 005 ("Abnormal
termination, signal 15") rather than 004 ("Job was evicted"). That is what
masked the vacate and why `on_exit_hold` never matched.

Memory was never the constraint: it grew 10393 -> 39212 MB over the first five
hours and then stayed flat for the final six, against a 49152 MB request.
`TimeExecute` was 40694 s (11.30 h) of a 42483 s (11.80 h) slot occupancy, the
difference being the 29 minutes spent transferring 206 GB of input.

**Cost of each restart:** 20 epochs took 11.3 h, about 34 min/epoch, so the
remaining 30 epochs need roughly 17 h - longer than this slot survived. Expect
two or three resubmissions, each paying ~30 min of input transfer, unless
`train.sub` gains a requeue rule. The candidate is
`on_exit_remove = (ExitBySignal == False)`, which sends a signal-killed job back
to the queue where `ON_EXIT_OR_EVICT` plus `run_train.sh`'s `last.ckpt` discovery
resumes it unattended; pair it with a `periodic_hold` on `NumJobStarts` so a
genuinely broken job cannot loop, and test it on a short run first.

**To resume,** take the exact argument list off the job ad
(`condor_history 11380824 -af Args`) rather than reconstructing it: `model1`
defaults to `accelerations=4 center_fractions=0.08`, and while the checkpoint
restores model hyperparameters, the rate and mask configuration comes from the
arguments — so a resume that omits them would silently retrain at R4. Only the
partial epoch 20 is lost, roughly 20 minutes of the 11.8 hours.

A timezone note, since it caused a false lead: the wandb run directory
(`run-20260929_000852`) is stamped **UTC** while the HTCondor log
(`06:15:45`) is **Central**. 00:08 UTC Sep 29 is 19:08 CDT Sep 28, so the run
spans 19:08 -> 06:15 = 11 h 07 m, consistent with both W&B's `_runtime` (11.08 h)
and Condor's `RemoteWallClockTime` (11.80 h). One continuous run.
