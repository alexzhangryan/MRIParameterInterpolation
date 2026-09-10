# Verification protocol: E2E VarNet baseline

Standalone companion to `ROADMAP.md` (Phase A infrastructure) and `plan.md` (Phase B DPI design).
This document covers one question only: which numbers from Sriram et al. 2020 can we actually
check, how do we check them, and what do we say when we cannot.

Nothing here depends on `plan.md`. Run this protocol before any DPI work begins, and re-run
Tier 1 whenever the environment, the data, or the eval code changes.

*Status: drafted 2026-09-09, not yet executed. No tier has been run.*

Tiers 0 and 1 are implemented in `verification/verify_varnet.py`, with the Docker image and
CHTC submit files alongside it. See `verification/README.md` for the run order.

---

## 0. Why this document exists

The supervisor's requirement is that the E2E VarNet numbers be verified before the group builds
on them. That requirement cannot be met in full, and the reason is structural rather than a
matter of compute budget. This protocol replaces "reproduce the paper" with a set of checks that
are each individually achievable and individually meaningful, and it states plainly which claims
are out of reach.

Three distinct claims get conflated under the word "reproducible". Keep them separate:

- Claim A, pipeline correctness. Our data loading, masking, coil combination, cropping, and
  metric code produce the same numbers other people get from the same model weights.
- Claim B, training reproduction. Training this repo from scratch recovers the paper's reported
  scores.
- Claim C, baseline validity. Our baseline is trained competently and its run-to-run variance is
  known, so a DPI improvement measured against it is real.

Claim A is verifiable and cheap. Claim B is partly verifiable at real cost. Claim C is the one
that actually gates the research, and no part of it depends on the paper.

---

## 1. Verifiability status of every published number

The paper contains three result tables.

### Tables 1 and 2, knee SSIM, separate model per acceleration

| Accel | Equispaced ACS lines | E2E-VN SSIM | Random frac ACS | E2E-VN SSIM |
|---|---|---|---|---|
| 4 | 30 | 0.923 | 0.08 | 0.910 |
| 6 | 22 | 0.907 | 0.06 | 0.892 |
| 8 | 16 | 0.893 | 0.04 | 0.878 |

SSIM only. No NMSE or PSNR is reported for these tables.

The evaluation split for Tables 1 and 2 is not stated anywhere in the paper. Validation is the
natural inference but the paper does not say it. Do not write "validation set" as a fact when
citing these numbers.

Status: reproducible in principle, at real cost, with two code changes (Section 5).

### Table 3, test split, trained on train+val for 100 epochs

| Data | 4x SSIM | 4x NMSE | 4x PSNR | 8x SSIM | 8x NMSE | 8x PSNR |
|---|---|---|---|---|---|---|
| Knee | 0.930 | 0.005 | 40 | 0.890 | 0.009 | 37 |
| Brain | 0.959 | 0.004 | 41 | 0.943 | 0.008 | 38 |

PSNR is reported to integer precision and NMSE to three decimals in the paper itself, so these
are not high-precision targets even in principle.

Status: permanently unverifiable. The knee test split has no public ground truth. The only
scoring route was the fastMRI leaderboard, which is gone. The fastmri.org domain transferred from
Meta to NYU on 2023-04-17 and the leaderboards were never rebuilt. The older EvalAI challenge
(challenge 153) closed in October 2019 and predates E2E VarNet entirely. No compute budget
changes this.

These are the headline numbers. If the supervisor's mental model of "the E2E numbers" is
Table 3, that model needs correcting at the start of the conversation, not at the end of a month
of GPU time.

### Ablations

The learned-versus-ESPIRiT sensitivity map ablation is Tables 1 and 2 themselves (the VN / VNU /
VNU-K / E2E-VN columns). The mask-type ablation is Table 1 versus Table 2. Figure 3 varies
equispaced mask parameters but its per-point values are only readable off a plot.

There is no cascade-count ablation in the paper. Do not go looking for one.

---

## 2. Reference values for Tier 1

These are third-party measurements of the released `knee_leaderboard_state_dict.pt` checkpoint,
not paper numbers. They are the anchors for Claim A.

| Source | Split | Accel | SSIM | PSNR | NMSE |
|---|---|---|---|---|---|
| PromptMR repo | own val-derived split | 4 | 0.9236 | 39.37 | 0.0053 |
| PromptMR repo | own val-derived split | 8 | 0.8936 | 37.30 | 0.0087 |
| HUMUS-Net Table 2 | knee validation | 8 | 0.8908 | not reported | not reported |

Both sources evaluate a checkpoint that was trained with `combine_train_val=True`, so the
validation split is training data for it. PromptMR states this explicitly and says its E2E VarNet
numbers "should be considered for reference purposes only". These values are therefore optimistic
as model-quality estimates and are useless as evidence about generalization.

They are still exactly the right anchors for pipeline verification, because pipeline verification
does not care whether the model is good, only whether our code reproduces someone else's
measurement of the same weights.

There is no clean held-out evaluation of the released knee checkpoint available to anyone.
`multicoil_test` has masks but no public ground truth, and `multicoil_val` is contaminated.

---

## 3. Tier 0, environment and eval-code verification

No data, no GPU, no external reference values. Everything here is an internal invariant, so a
failure is unambiguously a bug on our side. Run these first.

### 0.1 Framework API compatibility

The repo is written against the PyTorch Lightning 1.x argparse API (`Trainer.from_argparse_args`,
`add_argparse_args`, `gpus=`, `replace_sampler_ddp=`, `resume_from_checkpoint=`). All of those
were removed in Lightning 2.0.

```bash
python -c "import pytorch_lightning as pl; print(pl.__version__); assert pl.__version__.startswith('1.')"
python -c "import pytorch_lightning as pl; assert hasattr(pl.Trainer, 'from_argparse_args')"
```

Pass: both succeed. A `2.x` version here means every training script in the repo is dead on
arrival, and no later tier can run.

### 0.2 Repo's own test suite

```bash
pytest tests/ -x -q
```

Pass: green. This exercises the model, transforms, and Lightning modules against the repo's own
fixtures and catches most environment breakage before it costs GPU time.

### 0.3 Metric code identity check

Feed the ground truth in as the prediction. This validates `fastmri.evaluate` end to end with
zero ambiguity about what the correct answer is.

```python
# tools/verify_metrics_identity.py
import numpy as np
from fastmri.evaluate import ssim, psnr, nmse, mse

rng = np.random.RandomState(0)
gt = rng.rand(16, 320, 320).astype(np.float32)

assert np.isclose(ssim(gt, gt), 1.0, atol=1e-6), ssim(gt, gt)
assert nmse(gt, gt) == 0.0
assert mse(gt, gt) == 0.0
assert np.isinf(psnr(gt, gt)) or psnr(gt, gt) > 100
print("metric identity OK")
```

Pass: SSIM is $1.0$ to $10^{-6}$, NMSE and MSE are exactly $0$, PSNR is infinite or above 100.

### 0.4 Record the metric conventions

Read once and write down, because these determine whether our numbers are comparable to anyone
else's:

- SSIM is averaged over slices within a volume, then averaged over volumes. It is a per-volume
  statistic, not a per-slice one.
- NMSE and PSNR are computed on the whole volume at once, not per slice.
- Both target and reconstruction are center-cropped to a square of side equal to the target's
  last dimension before any metric is computed.
- `--acceleration` in `fastmri.evaluate` filters on `target.attrs["acceleration"]`, an attribute
  that exists only on test and challenge files. On the validation split this flag silently does
  nothing useful. Per-rate numbers on val require one inference pass per rate with a fixed mask,
  not one pass followed by filtering.

---

## 4. Tier 1, pipeline verification (Claim A)

Cost: one GPU, a few hours, no training. This is the highest value per hour in the project. Do
not start Tier 2 or any DPI work until this passes.

### 4.1 Data required

`knee_multicoil_val` only (93.8 GB). Verify against the `SHA256` file from the NYU download
before extracting. Extract to `<data>/multicoil_val/` as a flat directory of `.h5`.

### 4.2 The script change that is required

Implemented as `verification/verify_varnet.py tier1`. The description below is what it does
and why, so the file can be audited against this document.

`fastmri_examples/varnet/run_pretrained_varnet_inference.py` builds
`T.VarNetDataTransform()` with no `mask_func`. With no mask function the transform reads a mask
stored inside the file, and only the test and challenge splits store one. Run as-is against
`multicoil_val` and it fails.

Write a copy rather than editing the repo file, keeping `fastmri/` untouched in line with the
convention in `plan.md`:

```
tools/eval_pretrained_val.py
```

Differences from the original, and only these:

- `mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)`
- `data_transform = T.VarNetDataTransform(mask_func=mask)`
- `--mask_type`, `--center_fractions`, `--accelerations` exposed as CLI arguments
- a `--seed` argument feeding `np.random.seed` / `torch.manual_seed` so runs are repeatable

The model construction stays exactly `VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4,
sens_chans=8)`. Do not change it. That is the architecture the released state dict expects, and
`load_state_dict` will fail loudly if it drifts.

### 4.3 Run

One pass per acceleration. The random mask family is the one the reference values above used.

```bash
for R in 4 8; do
  case $R in 4) F=0.08 ;; 8) F=0.04 ;; esac
  python tools/eval_pretrained_val.py \
    --challenge varnet_knee_mc \
    --data_path  <data>/multicoil_val \
    --output_path runs/pretrained_val_R${R} \
    --mask_type random --center_fractions $F --accelerations $R \
    --seed 42
  python -m fastmri.evaluate \
    --target-path      <data>/multicoil_val \
    --predictions-path runs/pretrained_val_R${R}/reconstructions \
    --challenge multicoil \
    | tee runs/pretrained_val_R${R}/metrics.txt
done
```

### 4.4 Pass criteria

Against the Section 2 reference values. These thresholds are judgment calls, not derived
quantities, chosen to be loose enough to absorb mask-seed realization and split differences and
tight enough to catch a real bug.

| Check | Pass | Investigate | Fail |
|---|---|---|---|
| SSIM vs reference, either rate | within 0.010 | 0.010 to 0.020 | above 0.020 |
| PSNR vs reference, either rate | within 0.5 dB | 0.5 to 1.0 dB | above 1.0 dB |
| NMSE vs reference, either rate | within 15% relative | 15 to 30% | above 30% |

Note the reference values come from a different volume subset than the full official val split,
so a small systematic offset is expected and is not by itself a failure.

### 4.5 Internal invariants, no external reference needed

Run these alongside 4.4. A failure here is a definite bug regardless of what 4.4 says.

- Determinism. Two runs at `--seed 42` produce identical metrics to machine precision. If not,
  something in the mask generation or data ordering is unseeded, and every later comparison in
  the project is untrustworthy.
- Ordering. SSIM at 4x exceeds SSIM at 6x exceeds SSIM at 8x. A scrambled mask-to-sample mapping
  usually breaks this.
- Zero-filled control. Run the same evaluation with the model replaced by the zero-filled RSS
  reconstruction. Its scores must be far below the model's on identical masks. Record the actual
  numbers as the project's floor. Only the direction is being asserted here, not a target value.
- Shape agreement. Assert that reconstruction and target shapes match after `center_crop` for
  every volume, and that no volume was silently skipped. Count the volumes scored and confirm it
  equals the number of files in the target directory.

### 4.6 What passing Tier 1 does and does not establish

Establishes: the model definition, checkpoint loading, `.h5` reading, mask generation, k-space
handling, coil combination, cropping, and metric computation are all correct.

Does not establish: anything about training, anything about generalization, and nothing about the
paper's numbers. Tier 1 verifies Claim A only.

---

## 5. Tier 2, paper reproduction (Claim B)

Cost: GPU-weeks, one model per acceleration. This is the only genuine paper reproduction
available, and it targets Tables 1 and 2, never Table 3.

### 5.1 Two code changes the repo needs

Both are documented deviations, stated by the maintainers in
`fastmri_examples/varnet/README.md`, not discoveries.

Fixed ACS line counts. The paper uses a fixed number of center lines (30 at 4x, 22 at 6x, 16 at
8x for the equispaced family). The repo uses `center_fractions`, which scales with acquisition
width, so at a 368-wide knee acquisition a fraction of 0.08 gives 29 lines and 0.04 gives 15,
and the count drifts volume to volume. Reproducing Table 1 requires a `MaskFunc`
subclass parameterized by line count instead of fraction. This is a small class and it fits the
same pattern `plan.md` already uses for its recording mask functions.

Table 2 uses the random family with fractions 0.08 / 0.06 / 0.04, which the repo supports
directly, so Table 2 is reachable without the mask change. Reproduce Table 2 first.

Per-rate training. The repo trains jointly on multiple accelerations. The paper trained a
separate model per rate. Pass a single value to `--accelerations` and `--center_fractions` and
train one model per rate.

### 5.2 Configuration

From the paper: 12 cascades, Adam at $3 \times 10^{-4}$, 50 epochs, SSIM loss, no regularization
and no data augmentation, roughly 30M total parameters including the sensitivity module.

Not stated anywhere in the paper: batch size, GPU count, U-Net channel width, learning rate decay
schedule. These are genuinely unspecified. Use the repo's leaderboard-script values
(`chans=18`, `sens_chans=8`, `batch_size=1`, `lr_step_size=40`, `lr_gamma=0.1`) and record that
they are our choice filling a gap in the paper, not a paper-matching setting.

Note the local `fastmri_examples/varnet/varnet_reproduce_20201111/varnet_knee_leaderboard.py`
does not run on any modern Lightning (it calls `ModelCheckpoint(filepath=...)`, removed in 1.x,
and `accelerator="ddp"`, a pre-1.5 spelling). It is a source of hyperparameters, not an
executable. Port its values into `train_varnet_demo.py`. It is also the Table 3 configuration,
100 epochs on train+val, so it is the wrong config for Tables 1 and 2 regardless.

### 5.3 Pass criteria against Table 2

| Result | Interpretation |
|---|---|
| within 0.005 SSIM of the paper | strong reproduction, report as reproduced |
| 0.005 to 0.015 below | consistent with the documented deviations, report as approximate with the deviation list attached |
| 0.015 to 0.030 below | expected if training on a reduced data subset, report as a scaled reproduction, do not claim reproduction |
| more than 0.030 below | undertrained or buggy, do not report, debug |
| above the paper | suspect leakage or a metric mismatch, investigate before celebrating |

If training on one downloaded train batch rather than the full 973-volume training set, a deficit
is expected and the run should not be described as a reproduction attempt at all. Call it a
reduced-data baseline.

### 5.4 Deviation register

Maintain this list next to every number produced under Tier 2, and reproduce it in any writeup
that cites a comparison to the paper.

- Training data subset used, as a volume count and as a fraction of the official 973 train volumes
- Center-line handling, fixed count or fraction
- Joint or per-rate training
- Cascade count, if reduced below 12 for memory
- Channel width, batch size, and LR schedule, all filling gaps the paper leaves open
- Epochs actually completed versus 50
- Evaluation split, and whether it overlaps training data

---

## 6. Tier 3, baseline validity and seed variance (Claim C)

This is the tier that gates the DPI research, and no part of it depends on the paper being
reproduced. It is currently missing from `plan.md`, whose verification section asks that
`--dpi_scope none` track the baseline "to seed noise" without anywhere measuring seed noise.

### 6.1 Measure the noise floor

Train the same baseline configuration at three seeds, on identical data, and evaluate each on the
same held-out volumes with the same mask seed.

```bash
for S in 42 1337 2024; do
  python fastmri_examples/varnet/train_varnet_demo.py \
    --challenge multicoil --data_path <data> \
    --mask_type random --center_fractions 0.08 --accelerations 4 \
    --num_cascades 12 --chans 18 --sens_chans 8 \
    --lr 0.0003 --lr_step_size 40 --lr_gamma 0.1 \
    --batch_size 1 --max_epochs 50 --gpus 1 --seed $S \
    --default_root_dir runs/baseline_seed$S
done
```

Report the mean and standard deviation of held-out SSIM across seeds. Call that standard
deviation $\sigma$.

### 6.2 Minimum detectable effect

Any DPI improvement smaller than roughly $2\sigma$ is not reportable from a single run pair. If
the measured $\sigma$ turns out large relative to the effect DPI is expected to produce, that is
critical information, and it is much cheaper to learn now than after the full experiment grid.

### 6.3 Prefer paired comparisons

Volume difficulty dominates the variance in these metrics, so comparing two means across
approximately 199 validation volumes wastes most of the available statistical power. Record
per-volume SSIM for every run rather than only the aggregate, and compare methods with a paired
test over volumes (Wilcoxon signed-rank is the safe default, since per-volume SSIM is not
normally distributed). Report the median paired difference and its confidence interval alongside
the mean-of-means.

This requires per-volume output, which `fastmri.evaluate` does not currently emit. A small
wrapper that pushes per-volume values to a CSV instead of only to the `Statistics` aggregator is
worth writing once, at Tier 1, and reusing everywhere.

### 6.4 Baseline competence check

A DPI gain over a broken baseline is worthless. Before accepting a baseline as the comparison
point, confirm its validation SSIM is within the Tier 2 bands above, its training curve has
plateaued rather than still descending at the epoch cap, and it beats the zero-filled control
from 4.5 by a wide margin.

---

## 7. Language for writeups

Precision here protects the group from a reviewer objection later.

Accurate:

- "We verified our evaluation pipeline against the released E2E VarNet checkpoint, reproducing
  the values reported by PromptMR and HUMUS-Net to within X."
- "We reproduced the random-mask knee results of Sriram et al. Table 2 at 4x, 6x, and 8x, subject
  to the deviations listed in Section 5.4."
- "The test-split results in Table 3 cannot be verified, as the fastMRI leaderboard has been
  unavailable since April 2023 and the knee test split has no public ground truth."

Not accurate, and to be avoided:

- "We reproduced the E2E VarNet paper." Too broad, Table 3 is out of reach.
- "Our baseline matches the published SSIM of 0.930." That is a test-split number that cannot be
  computed locally.
- "Validated on held-out data" for anything involving the released checkpoint on
  `multicoil_val`. The checkpoint was trained on that split.
- "Reproduces the paper's validation numbers" for Tables 1 and 2. The paper never states which
  split those tables use.

---

## 8. Record keeping

One row per evaluation run, appended to `runs/results.csv`, never edited by hand:

```
run_id, date, git_sha, tier, model_source, checkpoint_path, split, n_volumes,
mask_type, center_fraction, acceleration, mask_seed, train_seed,
ssim_mean, ssim_std, psnr_mean, nmse_mean, per_volume_csv, notes
```

`model_source` is one of `pretrained`, `ours`. `git_sha` covers both the fastMRI checkout and
this project directory. `per_volume_csv` points at the per-volume output from 6.3. A number
without a row here does not exist.

---

## 9. Execution order

1. Tier 0, all four checks. No data needed.
2. Download `knee_multicoil_val`, verify SHA256.
3. Tier 1. Do not proceed past a failure.
4. Download one train batch, size the staging quota.
5. Tier 3.1, three baseline seeds. Establish $\sigma$ before any DPI comparison exists.
6. Tier 2, Table 2 reproduction at 4x, 6x, 8x, if the compute budget allows.
7. Tier 2 with the fixed-line mask for Table 1, optional.

Tiers 1 and 3 are required. Tier 2 is desirable and is the part most likely to be cut for
compute. Cutting it costs nothing scientifically, since every comparison in `plan.md` is internal
to this project.

---

## Sources

- E2E VarNet paper, Sriram, Zbontar et al. 2020: https://arxiv.org/abs/2004.06688
- fastMRI repository: https://github.com/facebookresearch/fastMRI
- VarNet example README, documenting the two deviations:
  https://github.com/facebookresearch/fastMRI/blob/main/fastmri_examples/varnet/README.md
- Leaderboard shutdown, fastMRI discussion #293:
  https://github.com/facebookresearch/fastMRI/discussions/293
- EvalAI challenge 153, closed October 2019:
  https://eval.ai/web/challenges/challenge-page/153/leaderboard/447
- HUMUS-Net, NeurIPS 2022, knee validation 8x reference:
  https://papers.neurips.cc/paper_files/paper/2022/file/a1bb3f96e255ae1e04325ae166bcef0f-Paper-Conference.pdf
- PromptMR, pretrained-checkpoint evaluation and contamination caveat:
  https://github.com/hellopipu/PromptMR/blob/main/promptmr_examples/fastmri/README.md
- fastMRI dataset paper, split volume and slice counts:
  https://arxiv.org/abs/1811.08839
