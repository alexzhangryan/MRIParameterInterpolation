# dpi/ — VarNet conditioned on the acceleration rate by Deep Parameter Interpolation

Phase B. Everything here is new work: DPI has only ever been applied to
diffusion and flow-matching image generation, never to an unrolled MRI
reconstruction network, so there is no reference implementation and no
published number to check against. The unit tests are the evidence that the
mechanics are right; `../VERIFICATION.md` is what decides whether the result
means anything.

*Status 2026-09-19: implemented, 21 unit tests and four local smoke targets
green on synthetic data. Not yet submitted on CHTC.*

## What DPI does, and what it is here

The paper (Park, McCann, Garcia-Cardona, Wohlberg, Kamilov, "Deep Parameter
Interpolation for Scalar Conditioning", [arXiv:2511.21028](https://arxiv.org/abs/2511.21028),
CVPR 2026; the group's own) conditions a network on a scalar by keeping two
learnable parameter sets and running the network on their interpolation.
Eq. (2):

    g(theta0, theta1, x, s) = f([1 - lambda(s)] theta0 + lambda(s) theta1, x)

with lambda monotone, `lambda(s_min) = 0`, `lambda(s_max) = 1`, and, section
3.2, lambda built as a cumulative sum of a softmax over a learnable vector
phi, which makes it monotone by construction:

    p = softmax(phi),    lambda(s_i) = sum_{j <= i} p_j.

In the paper the scalar is diffusion time or noise level. **Here it is the
MRI acceleration rate R**, and `f` is fastMRI's VarNet. That substitution is
the contribution; nothing else about the mechanism changes.

| Paper | Here |
|---|---|
| scalar s = timestep / noise level | acceleration rate R |
| s_min, s_max = the ends of the schedule | `--accel_min 2`, `--accel_max 8` |
| f = DRUNet or ADM | fastMRI VarNet, 12 cascades |
| duplicated: conv and transposed-conv weights, biases, group-norm affine | conv and transposed-conv weights, the one 1x1-conv bias per U-Net, and each cascade's `dc_weight`. VarNet's InstanceNorm has no affine parameters, so there is nothing else to duplicate |
| phi of length S = 1000 | `--lambda_length 1000` |
| separate lr 1e-3 for phi (section 4.1) | `--lambda_lr 1e-3` |
| one scalar per batch | one rate per forward call; a mixed batch is split per sample (never happens at batch_size 1) |
| warm start from a pretrained model | `--init_from_baseline <checkpoint>`, optional; the default is from scratch |

**Why the rate is a good scalar for this.** The aliasing VarNet has to undo
changes character with R, and the ACS width it estimates sensitivity maps
from spans a factor of four across 2x-8x. The baseline `mixed acceleration
brain` run has to serve all of that with one parameter set. See
`../plan.md`, "Expected gain from explicit rate conditioning", for the
hypothesis this run tests and for what each outcome would mean.

## The training setup is the baseline's, exactly

This is the point of the comparison, so it is enforced structurally rather
than by copying values: `train_dpi.py` imports `../training/train_wandb.py`
and calls its `cli_main` with two hooks replaced.

| Replaced | Everything else |
|---|---|
| `build_model`: `DPIVarNetModule` instead of `VarNetModule` | W&B preflight and logger, seeding, `FastMriDataModule`, `ModelCheckpoint`, auto-resume from `output/checkpoints`, the trainer, and every default: 12 cascades, 18/8 channels, Adam 3e-4, x0.1 at epoch 40, batch 1, 50 epochs, SSIM loss, `equispaced_fraction`, seed 42, one GPU |
| `build_transforms`: recording mask functions and `DPIVarNetDataTransform` | the mask family, the rate lists, the seeding (`use_seed=False` for train, filename-seeded for val) |

The job also reuses `../training/run_train.sh` as its executable, unchanged;
`train.sub` just sets `TRAIN_DRIVER=train_dpi.py`. So extraction, the W&B
preflight, eviction handling, the hold-on-failure behaviour and the
`output/` contract are literally the baseline's code path, not a copy of it.

Data: `brain_multicoil_train_batch_0` (455 volumes) and
`brain_multicoil_val_batch_0` (460 volumes) from the group staging
directory, the same two batches the baseline trains and validates on.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `dpi_varnet.py` | anywhere with `fastmri` | The model. `LambdaTable`, the duplicated conv layers, and `DPIUnet` / `DPINormUnet` / `DPISensitivityModel` / `DPIVarNetBlock` / `DPIVarNet` mirroring `fastmri/models/`. Also `load_baseline_state_dict` (the warm start) and `dpi_param_summary`. |
| `dpi_transforms.py` | same | Recording mask functions for all five fastMRI families, `DPIVarNetSample` (the baseline's eight fields plus `acceleration`), and `DPIVarNetDataTransform`. |
| `dpi_module.py` | same | `DPIVarNetModule`: the baseline Lightning module with the DPI network, `batch.acceleration` threaded through each step, `lambda/R{2,4,6,8}` logged each validation epoch, and phi in its own optimiser group. |
| `train_dpi.py` | inside the job | The driver: `../training/train_wandb.py` with the two hooks swapped. |
| `test_dpi.py` | laptop | 21 checks of the DPI mechanics. `make test`. |
| `train.sub` | access point | HTCondor submit file. `../training/run_train.sh` as the executable, `TRAIN_DRIVER=train_dpi.py`, the DPI flags. |
| `submit.sh` | access point | `./submit.sh [name="..."] [init=...] [key=value ...]`. Sources `../.env`, refuses to submit without a W&B key unless `OFFLINE=1`, proves with `-dry-run` that the preset and the DPI driver reached the job ad. |
| `Makefile` | laptop + access point | `make test/smoke/job-smoke/warm-smoke/local-run` and `make submit/resume/status/logs/why`. |

Nothing in `fastMRI/`, `parameter_interpolation/` or `../training/`'s
behaviour is modified. `../training/train_wandb.py` and `run_train.sh` grew
two extension points (the two hooks, and `TRAIN_DRIVER`); the baseline path
through them is unchanged, which `../training/make smoke job-smoke` still
proves.

## What the tests check, and why each one matters

`make test` (21 checks, in the verification image, which is the training
environment plus pytest):

| Check | Why it matters |
|---|---|
| DPI equals the baseline at init, every rate | eq. (2) with identical sets must collapse to `f(theta, x)`. If this fails the wiring is wrong. Compared relatively: `lam*w + (1-lam)*w` is only bit-exact at lambda 0 and 1, so intermediate rates agree to float32 round-off (~1e-7) |
| bit-exact at lambda 0 and 1 | the two endpoints really are the two parameter sets |
| lambda monotone, `[0] = 0`, `[-1] = 1` | the paper's guarantee; a non-monotone lambda would make the conditioning meaningless |
| log spacing puts 2/4/6/8 at bins 0/500/792/1000 | R=4 sits exactly halfway between 2x and 8x in log space |
| every baseline state-dict key exists verbatim; the only new ones are `*_copy` and `lambda_table.phi` | the warm start and any later scoring depend on it |
| parameters = 2x base + S | every learnable tensor is duplicated, nothing missed |
| `--no_dpi_sens` removes exactly the sensitivity net's copies | the ablation does what it says |
| phi's gradient is exactly zero while the sets are equal, nonzero once they differ | **inherent to DPI, not a bug**: dL/dlambda = <g, theta1 - theta0> = 0 when the sets match. The sets separate on the first step because theta1 gets lambda*g and theta0 gets (1-lambda)*g |
| both sets receive gradient | excluding `cascades.0.dc_weight`, whose soft data-consistency term is identically zero in the baseline too (measured) |
| different rates give different outputs once the sets differ | the conditioning actually does something |
| a mixed batch equals concatenated single-sample calls | the batch-splitting path is exact |
| phi is in its own optimiser group at its own lr, and every parameter is in exactly one group | the paper's separate learning rate |
| the transform records the rate the mask function drew, and its other eight fields are bit-identical to the baseline transform's | the scalar is the true nominal rate, and the data is otherwise untouched |
| a seeded validation volume draws one rate | matches how the baseline validates |

Beyond the tests, measured on a 12-step synthetic run: the two sets start
identical, diverge (max abs difference 2.4e-3 after 12 steps), lambda moves,
and the table stays monotone with its endpoints pinned.

At the real architecture (12 cascades, 18/8 channels), measured:

| | Parameters | vs baseline |
|---|---|---|
| baseline VarNet | 29,936,966 | |
| DPI, default | 59,874,932 | 2.0000x (+ phi, 1,000) |
| DPI, `--no_dpi_sens` | 59,390,034 | 1.9838x |

A DPI checkpoint is about 240 MB of weights, roughly twice the baseline's.

## Local checks before submitting

```bash
cd dpi
make test          # the 21 mechanics checks
make smoke         # train_dpi.py on synthetic phantoms, then a rerun that must resume
make job-smoke     # ../training/run_train.sh with TRAIN_DRIVER=train_dpi.py, twice (fresh + evicted)
make warm-smoke    # train a baseline checkpoint, then start DPI from it
make local-run     # all four
```

They need no W&B key and no real data. `make local-run` green means the
model, the transform, the driver, the job executable, the resume contract
and the warm start all work before a GPU slot is used.

## Run on CHTC

The image is `../training`'s, already built and pushed; the DPI code travels
with the job. The W&B key is the repo-root `../.env`, as everywhere else.

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research && git pull
cd dpi
chmod +x submit.sh ../training/run_train.sh
ls -la ../.env                     # the one secrets file

# 1. dry run: check the args, the DPI driver, and the transfer list
condor_submit -dry-run /dev/stdout train.sub model=dpi \
    run_name=dpi-mixed-acceleration-brain wandb_name="dpi mixed acceleration brain" \
  | grep -iE "^(Arguments|Environment|TransferInput|RequestDisk) "

# 2. short test job: image, preflight, extraction, 12 cascades x 2 parameter sets on a real GPU
make submit ARGS='max_epochs=1 extra_args="--limit_train_batches 50 --limit_val_batches 10"'
make logs

# 3. the real run
make submit
```

The `.out` must show, in order: `driver=train_dpi.py`, `W&B preflight OK`,
`multicoil_train: 455 volumes`, `[dpi] parameters: base 29,936,... + copies
29,936,... + phi 1,000`, `from scratch (both parameter sets randomly
initialised, identical)`, and a `W&B run: 'dpi mixed acceleration brain'`
line with a URL.

**W&B run:** `dpi mixed acceleration brain` (id
`dpi-mixed-acceleration-brain`) in project `fastmri-varnet-train`, next to
the baseline's `mixed acceleration brain`. `NAME='...'` picks a different
one; the id is its slug. As in `../training`, a job whose W&B key does not
work **holds itself** (exit 4) rather than training blind; `OFFLINE=1` is the
only opt-out.

**Warm start (optional).** The paper starts both sets from a pretrained
model. Once the baseline run has a checkpoint:

```bash
make submit INIT=../training/runs/brain/model2/<Cluster>/checkpoints/last.ckpt \
            NAME='dpi mixed acceleration brain warm'
```

`submit.sh` stages it as `baseline_init.ckpt`, which `run_train.sh`
deliberately leaves in the working directory rather than treating it as a
resume point. Loading asserts the architecture matches and that the only
unfilled parameters are the copies and phi, then sets each copy to its base
tensor, so training starts exactly at the baseline for every rate.

**Resuming** is the baseline's: `make resume CKPT=runs/brain/dpi/<Cluster>/checkpoints/last.ckpt`.

## Reading the result

Watch `lambda/R2`, `lambda/R4`, `lambda/R6`, `lambda/R8` in W&B. They start
on a near-straight line (phi is `randn`, so softmax is nearly uniform) with
`lambda/R2 = 0` and `lambda/R8 = 1` pinned by construction. If `lambda/R4`
and `lambda/R6` never move, phi is not learning: raise `--lambda_lr`, or warm
start, before reading anything into the metrics.

`val_metrics/ssim` is a mixture over the four rates (`../training/README.md`,
"Masks"), so it is comparable with the baseline run's but says nothing per
rate. Per-rate numbers come from `../verification/`, which forces every
volume to each rate in turn. Scoring a DPI checkpoint there needs the DPI
model class, which `verify_varnet.py` does not import yet: that is the next
piece of work, not something this directory does.

Before believing any difference, `../VERIFICATION.md` section 6: the
seed-to-seed standard deviation has to be measured, and a difference under
about 2 sigma is not reportable from one run pair.

## Deviations from the paper, deliberate

- **Optimiser.** The paper uses AdamW at 1e-5 with weight decay 0.05 for its
  diffusion backbones. This uses the fastMRI baseline's Adam at 3e-4 with no
  weight decay, because matching the run it is compared against matters more.
  phi keeps the paper's separate 1e-3.
- **phi's learning rate decays** with the backbone at epoch 40, because
  `StepLR` scales every parameter group. The paper holds phi constant.
- **From scratch by default.** The paper always warm starts from a pretrained
  model. Both are available here; from scratch is the default because the
  baseline it is compared against is also from scratch.
- **The scalar is nominal, not measured.** `equispaced_fraction` masks make
  the realised rate equal the nominal one, so the two agree; the estimator in
  `dpi_transforms.py` is only a fallback for the test split, which ships a
  stored mask and no mask function.
