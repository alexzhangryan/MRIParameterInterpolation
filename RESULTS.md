# Per-rate results: DPI vs blind VarNet on fastMRI brain

**Phase C block A, item 2 of the 2026-09-24 supervisor feedback.**
Analysis run 2026-09-29. Numbers pulled from W&B, not from this checkout — the
repo holds no run outputs.

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

## Open: the R8 per-rate ceiling baseline is unfinished

`blind r8 brain` (cluster 11380824, accelerations [8], center_fractions [0.04])
**stopped at epoch 20 of 50**, so the per-rate ceiling row is not yet available.

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

**Unexplained and worth one check:** `train.sub` already sets
`+GPUJobLength = "long"` (7 days), so a kill at 11.8 h should not have happened
on a GPU Lab slot. Get `LastRemoteHost`, `GPUJobLength` and `WantGPULab` off the
job ad to see whether the attributes reached the ad and which machine it ran on;
`requirements` does not restrict to GPU Lab nodes, so a slot with its own
retirement time is one candidate.

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
