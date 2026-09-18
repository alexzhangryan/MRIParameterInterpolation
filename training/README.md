# training/ — E2E VarNet on the fastMRI brain set, CHTC

Runbook for the training runs. Read top to bottom the first time; after
that, section 4 is the only part you come back to.

*Status 2026-09-18: repointed from the 2026-09 knee subsets (personal
`/staging`, 100 GB quota) to the fastMRI **brain** multicoil set in the
Kamilov group staging directory, which `verification/prepare_brain_staging.sh`
transferred and SHA256-verified. The configuration is now the paper's and the
fastMRI leaderboard scripts' (12 cascades, Adam 3e-4) instead of the demo
script's (8 cascades, 1e-3). Green on the local smoke tests; not yet
submitted on CHTC in this form. The knee-era jobs of 2026-09-12..14 were
short tests and are superseded.*

Nothing here modifies `fastMRI/`, `parameter_interpolation/`, or any file
outside this directory. The job runs the `fastmri` package as installed in the
Docker image (pinned to the same commit the `fastMRI/` submodule is checked
out at) through a small driver, `train_wandb.py`, that builds the same
`VarNetModule` and `FastMriDataModule` as
`fastmri_examples/varnet/train_varnet_demo.py`.

## What the 2026-09-18 meeting asked for, and where it lives

| Ask | What this directory does now |
|---|---|
| E2E VarNet on multicoil **brain** | `dataset = brain` in `train.sub`: `brain_multicoil_train_batch_0.tar.xz` (455 volumes) and `brain_multicoil_val_batch_0.tar.xz` (460 volumes) from `/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/` |
| Acceleration rates **2, 4, 6, 8** | the `model2` preset: `--accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04`, one pair drawn per training slice |
| Check the training configuration (batch size, lr, ...) | "Configuration" below: 12 cascades, 18 / 8 channels, Adam 3e-4, x0.1 at epoch 40, batch 1, 50 epochs, SSIM loss. Batch size and the LR schedule are not stated in the paper; they follow the fastMRI leaderboard script |
| Check how the acceleration mask is generated (random or uniform) | "Masks" below. Lines: equispaced (uniform spacing, random offset), density-corrected (`equispaced_fraction`). Rate: one of the four drawn uniformly at random per training slice; per volume, seeded by filename, at validation. The network is never told which |
| What is the gain from an explicit rate scalar over the blind joint model | `../plan.md`, section "Expected gain from explicit rate conditioning" |

## The presets

| Preset | `accelerations` | `center_fractions` | What it is |
|---|---|---|---|
| **model2** | `2 4 6 8` | `0.16 0.08 0.0533 0.04` | **The run the meeting asked for.** One network, four rates. For every training sample the mask function draws one (acceleration, center fraction) pair uniformly at random from the four. No explicit rate signal reaches the network: this is the "joint training, no conditioning" baseline that Phase B's DPI model will be measured against. |
| **model1** | `4` | `0.08` | One network, one rate. Optional per-rate reference (the paper trains one model per rate); the 4x point of the per-rate ceiling Phase B compares against. |

The two lists are paired elementwise by `fastmri.data.subsample.MaskFunc`;
center fractions follow the fastMRI convention of 0.32 / R. 2x and 6x are
not in the fastMRI challenge set; their fractions are that convention
extended.

## Configuration

Both presets are trained from scratch (random initialisation, no pretrained
weights, no checkpoint). The only things that differ between them are the two
mask lists above. Everything else is a `train_wandb.py` default, compared
here against the three sources it could follow:

| Setting | **Here** | Paper (Sriram et al. 2020, sec. 4.1) | fastMRI leaderboard script (`varnet_reproduce_20201111/varnet_brain_leaderboard.py`) | `train_varnet_demo.py` (used here until 2026-09-18) | Flag |
|---|---|---|---|---|---|
| Unrolled cascades | **12** | 12 (T = 12, ~29.5M params + 0.5M in the SME, ~30M total) | 12 | 8 ("lower memory consumption") | `--num_cascades` |
| Regulariser U-Net channels / pools | 18 / 4 | U-Net, width not stated | 18 / 4 | 18 / 4 | `--chans` / `--pools` |
| Sensitivity-map U-Net channels / pools | 8 / 4 | not stated | 8 / 4 | 8 / 4 | `--sens_chans` / `--sens_pools` |
| Optimiser, learning rate | **Adam, 3e-4** | Adam, 0.0003 | Adam, 0.0003 | Adam, 0.001 | `--lr` |
| LR schedule | x0.1 at epoch 40 | not stated | x0.1 at epoch 40 | x0.1 at epoch 40 | `--lr_step_size`, `--lr_gamma` |
| Weight decay / augmentation | none | none ("without any regularization or data augmentation") | none | none | `--weight_decay` |
| Loss | SSIM | SSIM | SSIM (fixed in `VarNetModule`) | SSIM | |
| Batch size | 1 per GPU, **1 GPU** | not stated | 1 per GPU, 32 GPUs DDP (effective 32) | 1 per GPU, 2 GPUs DDP | `--batch_size`, `--gpus` |
| Epochs | 50 | 50 (100 on train+val for the test-set table) | 50 | 50 | `--max_epochs` |
| Mask family | `equispaced_fraction` | equispaced M_e(m, l) and random M_r(R, f), fixed centre-line counts | `equispaced` (brain), `random` (knee) | `equispaced_fraction` | `--mask_type` |
| Rates | 2 / 4 / 6 / 8 jointly (model2) | one model per rate | 4 and 8 jointly | 4 (default) | `--accelerations` |
| Training data | brain `train_batch_0`, 455 of 4,469 volumes | full split | train + val combined | | `train_data=` macro |
| Validation data | brain `val_batch_0`, 460 of 1,378 volumes | not stated | | | `val_data=` macro |
| Seed / determinism | 42 / `deterministic=True` | | 42 / True | 42 / True | `--seed`, `--deterministic` |

Three things to know about that table:

- **The effective batch size is the one real deviation from the leaderboard
  recipe.** The leaderboard script ran 32 GPUs at batch 1 each, so one
  optimiser step saw 32 slices; here one step sees 1, at the same learning
  rate. The paper does not state its batch size, so there is no "paper value"
  to match. 3e-4 at batch 1 is the conservative side of that gap. If the
  training curve is clearly too slow, `extra_args="--lr 0.001"` is the demo
  script's answer to the same problem and a one-line documented change.
- **12 cascades fit a 24 GB GPU at batch 1.** Brain slices are up to ~20
  coils at 640x320 (some 768x396). The per-cascade U-Net runs on the single
  coil-combined image; only the sensitivity net and the expand/reduce
  operators touch all coils. Estimated peak is well under 24 GB. If a job
  dies with CUDA OOM, `gpu_mem=40000M` moves it to A100/L40-class slots, or
  `extra_args="--num_cascades 8"` falls back to the demo figure.
- **Everything is a flag.** `extra_args` in `train.sub` is appended verbatim
  to `train_wandb.py`, which accepts every `pl.Trainer`, `FastMriDataModule`
  and `VarNetModule` argument.

The full command each submission resolves to, exactly as `run_train.sh` runs
it inside the job (`train.sub` `arguments` line plus the two flags
`run_train.sh` adds):

```bash
# model 2 (brain, four rates)
python train_wandb.py --data_path data/fastmri --default_root_dir output \
    --run_name varnet-brain-model2 --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
    --mask_type equispaced_fraction --max_epochs 50 --gpus 1

# model 1 (brain, 4x)
python train_wandb.py --data_path data/fastmri --default_root_dir output \
    --run_name varnet-brain-model1 --accelerations 4 --center_fractions 0.08 \
    --mask_type equispaced_fraction --max_epochs 50 --gpus 1
```

The `.out` log prints the resolved architecture and optimiser on its second
`[train_wandb]` line, so a submission can be checked against this table
without opening the checkpoint.

## Masks: how the acceleration mask is generated

Everything below is `fastmri/data/subsample.py` at the pinned commit, read
rather than assumed.

**One `MaskFunc` for all rates.** `create_mask_for_mask_type("equispaced_fraction",
[0.16, 0.08, 0.0533, 0.04], [2, 4, 6, 8])` builds a single
`EquispacedMaskFractionFunc`. Every time it is called it:

1. **Draws the rate.** `choose_acceleration()` picks an index uniformly at
   random (`rng.randint(len(center_fractions))`) and uses that
   (center fraction, acceleration) pair. So each of 2x/4x/6x/8x has
   probability 1/4, independently per call. The network is not told which.
2. **Fills the centre.** `round(num_cols * center_fraction)` contiguous
   columns around the k-space centre are kept (the ACS lines the sensitivity
   net reads). At 320-wide brain k-space that is 51 / 26 / 17 / 13 lines for
   2x / 4x / 6x / 8x; at 368 wide (some scans) 59 / 29 / 20 / 15. The paper
   used a fixed count (e.g. 30 at 4x); this is the documented "centre
   fraction" deviation.
3. **Places the outer lines, equispaced.** Every `adjusted_accel`-th column
   from a random offset, where `adjusted_accel` is solved so that centre plus
   outer lines together sample 1/R of the columns, i.e. the *realised*
   acceleration equals the nominal R. The offset is `rng.randint(0,
   round(adjusted_accel))`, so the pattern is shifted at random per call.

The two alternatives, for the record:

| `--mask_type` | Outer lines | Realised rate at nominal 4x | Who uses it |
|---|---|---|---|
| `equispaced_fraction` (**here**) | every ~R-th column, spacing corrected for the centre | 4.0 | fastMRI's public multicoil test masks; `train_varnet_demo.py` default |
| `equispaced` | every R-th column, uncorrected; centre added on top | ~3.2 (0.310 of columns, measured in `verification` Tier 0) | the paper's M_e; the fastMRI brain leaderboard script |
| `random` | each column kept with a Bernoulli probability chosen so the expected total is 1/R | 4.0 in expectation | the paper's M_r; the fastMRI knee leaderboard script and the knee challenge |

`equispaced_fraction` is kept for the brain runs, and not the leaderboard
script's `equispaced`, for one reason that matters to Phase B: the model
will be conditioned on the nominal R, so nominal must mean realised. Under
`equispaced` a "4x" mask and an "8x" mask are really 3.2x and 6.2x (0.310 and
0.160 of the columns, Tier 0), and the scalar would be mislabelled. Pass `mask_type=equispaced` to reproduce the
leaderboard convention instead.

**Train versus validation.** The training transform is
`VarNetDataTransform(mask_func=mask, use_seed=False)`: a fresh draw (rate,
offset) for every slice, every epoch. The validation transform is
`VarNetDataTransform(mask_func=mask)` with `use_seed=True`: the RNG is seeded
from the filename, so every slice of a validation volume gets the same mask
and **each validation volume is locked to one rate for the whole run**. The
`val_metrics/ssim` Lightning logs is therefore a mean over a fixed mixture of
the four rates (roughly a quarter of the volumes each), not a per-rate
number. Per-rate numbers come from `../verification/` (`make verify
MODEL=model2 ARGS='ckpt=...'`), which forces every volume to each rate in
turn.

**What the network sees of the rate.** Nothing explicit. The mask is an
input to every cascade's data-consistency term and to the sensitivity net
(through `num_low_frequencies`), so R is *inferable* from the input, but
every U-Net and every `dc_weight` is shared across rates. That is the gap
Phase B targets; `../plan.md` "Expected gain" says what to expect from
closing it.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `train_wandb.py` | inside the job (or any `fastmri` env) | The driver. W&B logger, `last.ckpt`, auto-resume from `output/checkpoints/`, leaderboard-script defaults. |
| `run_train.sh` | inside the job | Job executable: extracts every `*multicoil_*.tar(.xz)` it is given, finds `multicoil_train` / `multicoil_val`, seeds a resume checkpoint, runs the driver, collects `output/`. Anatomy-agnostic. |
| `train.sub` | access point | HTCondor submit file, container universe. `dataset = brain`, group staging, batch 0 of each split; every default guarded. |
| `submit.sh` | access point | `./submit.sh model2\|model1 [name=value ...]`. Refuses to submit without a W&B key unless `OFFLINE=1`; proves with `-dry-run` that the preset reached the job ad. |
| `Makefile` | laptop + access point | `make build/push/smoke/job-smoke/job-evict` (Docker) and `make submit-model2/submit-model1/resume/status/logs/why` (condor). |
| `Dockerfile` | laptop (build), CHTC (run) | Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, wandb, fastMRI at `91f2df4`. Pinned to `linux/amd64`. |
| `make_subset.sh`, `subset_val.sub`, `subset_train.sub` | access point | **Knee only, 2026-09** (section 3b): the repack that fit the knee splits under the 100 GB personal quota. Not used for brain. |
| `../verification/prepare_brain_staging.sh` | CHTC transfer node | What put the brain set into group staging: 20 tarballs, ~1.37 TB, left compressed, SHA256-verified. |
| `../.env.example` | repo root | Template for the shared `../.env` (gitignored): `WANDB_API_KEY`, `WANDB_ENTITY`, the NYU URLs. `submit.sh` sources `../.env` itself. |

## Where things run

| Machine | How you get there | What happens there |
|---|---|---|
| Laptop (this Mac) | | Build and push the Docker image. Smoke-test on synthetic data. |
| CHTC access point (the "VM") | `ssh apryan3@ap2001.chtc.wisc.edu` | Holds this directory, submits jobs, receives checkpoints and logs. 40 GB home quota: code only. |
| CHTC transfer node | `ssh apryan3@transfer.chtc.wisc.edu` | Bulk data downloads into `/staging`. Done for brain. |
| CHTC execute node | never directly | Pulls the image from Docker Hub, runs `run_train.sh`, is wiped when the job ends. |

Two staging areas, one macro:

| `staging` macro | Quota | Holds | Used by |
|---|---|---|---|
| `/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain` (**default**) | 2.5 TB, **10,000 items** | the 20 NYU brain tarballs + `SHA256`, nothing extracted | brain runs |
| `/staging/a/apryan3/fastmri` | 100 GB | `knee_multicoil_{train,val}_subset.tar(.xz)`, the released knee checkpoint | `dataset=knee` runs, `../verification/` |

The item cap on the group directory is why nothing is ever extracted there
and why `run_train.sh` unpacks into job scratch: the brain splits are
thousands of `.h5` files.

## 0. Before anything

- **Rotate the W&B key** that is in the old `Research/inpainting.sub` git
  history (ROADMAP.md security note). Every `WANDB_API_KEY` below means the
  new key. It lives in the untracked `../.env` or is exported in a shell,
  never written to a file in this repo.
- **The brain tarballs are the gate.** `train.sub` expects
  `brain_multicoil_train_batch_0.tar.xz` and `brain_multicoil_val_batch_0.tar.xz`
  in the group directory (section 3a checks).
- Docker Desktop running on the laptop, logged in to Docker Hub
  (`docker login`).

## 1. Laptop: build and push the image (once per tag)

```bash
cd training
make build push IMAGE=<dockerhub_user>/fastmri-train:2026-09
```

The image is `linux/amd64`. CHTC is x86_64, the cu118 torch wheels exist only
for x86_64, and the pinned conda `hdf5` build string is a linux-64 build.
This Mac is Apple Silicon, so a bare `docker build` used to produce an arm64
image and fail in the conda layer with
`hdf5 1.10.6 nompi_h6a2412b_1114 does not exist`. The `FROM` line in the
Dockerfile now pins the platform and `make build` passes `--platform` too,
so either route works. Every local `docker run` of the image goes through
emulation: slow, fine for synthetic smoke tests.

Use a dated tag, never `:latest`, so a rebuild cannot change what a checkpoint
was produced with. The build ends with a self-check that imports `fastmri`,
`WandbLogger`, and asserts Lightning is 1.x. Nothing in the 2026-09-18 change
touches the image: `train_wandb.py` is transferred with the job, not baked
in, so the existing `docker://genjigod/fastmri-train:2026-09` is still right.

## 2. Laptop: smoke-test with no real data (minutes each, under emulation)

```bash
make smoke        # train_wandb.py on synthetic phantoms, 1 epoch, then a rerun that MUST resume from last.ckpt
make job-smoke    # run_train.sh exactly as HTCondor runs it, twice: fresh, then "after eviction"
make job-evict    # SIGTERM a running job: the cleanup must still run
```

They generate tiny fake `multicoil_train` / `multicoil_val` directories with
`../verification/make_synthetic_val.py`, pack them as
`synthetic_multicoil_{train,val}.tar` (named so `run_train.sh`'s
`*multicoil_*` glob finds them, as it finds the real brain batches), train a
2-cascade model on CPU, and fail loudly if `output/checkpoints/last.ckpt` is
not written or if the second run does not print `resume from:
output/checkpoints/last.ckpt`. `make smoke` uses the model 2 rate list,
`make job-smoke` the model 1 list. Green here means the image, the driver,
the job executable, extraction, and the checkpoint/resume contract all work
before a single GPU slot is used. The phantoms are knee-shaped; nothing in
the pipeline depends on that.

## 3. Copy to the access point (the "VM")

Only this directory is needed on CHTC. Neither submodule is: the image
carries its own copy of fastMRI at the pinned commit. `git pull` in the
clone on the access point is the simplest way (submodules can stay
uninitialised); or scp the files below.

| Copy | Why |
|---|---|
| `train.sub` | submit file |
| `submit.sh` | wraps `condor_submit`, applies the model presets, preflight |
| `run_train.sh` | the job executable (transferred to the execute node by HTCondor) |
| `train_wandb.py` | the driver (listed in `transfer_input_files`) |
| `Makefile` | for `make submit-model2` etc. Optional; `./submit.sh` works alone. |

What must not go up: `synthetic/`, `jobtest/`, `evicttest/`, `output/`,
`*.tar`, `*.ckpt`, `.make/`, `dataset_cache.pkl`, and the laptop's `../.env`
(create that one by hand on the access point from `../.env.example`).

```bash
# laptop, from the repo root
scp training/train.sub training/submit.sh training/run_train.sh training/train_wandb.py training/Makefile \
    apryan3@ap2001.chtc.wisc.edu:~/Fall26Research/training/
```

Then, on the access point, one-time checks:

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research/training
chmod +x run_train.sh submit.sh
grep '^  image' train.sub                 # must not say CHANGE_ME
```

### 3a. Check the brain data and size the job

```bash
ls -la /staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/
get_quotas /staging/groups/kamilov_group/Kamilov-SciAI-datasets
```

Expect 20 `brain_*.tar.xz` files plus `SHA256`; the two the defaults use are
`brain_multicoil_train_batch_0.tar.xz` and `brain_multicoil_val_batch_0.tar.xz`.
Their exact sizes set `request_disk`: a fastMRI `.tar.xz` extracts to about
1.9x, and `run_train.sh` deletes a tarball only after its extraction finishes,
so the peak (while the second tarball extracts) is roughly
`1.9 x train + 2.9 x val`. The `520GB` default in `train.sub` assumes two
~100 GB batches; override with `request_disk=` if the listing says otherwise.
Extraction alone is on the order of an hour per batch, paid again on every
eviction restart.

Let `condor_submit` parse the file without queueing anything:

```bash
condor_submit -dry-run /dev/stdout train.sub model=model2 accelerations="2 4 6 8" \
    center_fractions="0.16 0.08 0.0533 0.04" \
  | grep -iE "^(Arguments|TransferInput|RequestDisk|Requirements) "
```

`TransferInput` must list the two brain tarballs from the group directory.

### 3b. The knee subsets (2026-09, superseded)

`/staging/a/apryan3/fastmri/` still holds `knee_multicoil_val_subset.tar`
(20 of 199 val volumes) and `knee_multicoil_train_subset.tar.xz` (~120 of
973 train volumes, the first 65 GB of NYU's `train_batch_0`), built by
`make subset-val` / `make subset-train` because the full knee splits do not
fit the 100 GB personal quota. They are not used by the brain runs. To train
on them again:

```bash
./submit.sh model2 dataset=knee staging=/staging/a/apryan3/fastmri \
    train_data=file:///staging/a/apryan3/fastmri/knee_multicoil_train_subset.tar.xz \
    val_data=file:///staging/a/apryan3/fastmri/knee_multicoil_val_subset.tar request_disk=260GB
```

## 4. Run (access point)

Every submission starts the same way:

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research/training
# either export them, or put them in the shared root .env
# (cp ../.env.example ../.env; chmod 600 ../.env); submit.sh sources ../.env itself.
export WANDB_API_KEY=...          # freshly rotated
export WANDB_ENTITY=...           # optional
```

### 4.1 Short test job first (model 2, one epoch, 50 batches)

Checks the image, the group-staging transfer, the extraction, the disk
request, `/dev/shm`, the GPU, and 12 cascades' memory on real brain slices,
before committing a slot for days:

```bash
make submit-model2 ARGS='max_epochs=1 extra_args="--limit_train_batches 50 --limit_val_batches 10"'
make logs                          # tail -f the newest logs/*.out
```

The `.out` must show `[run_train] multicoil_train: 455 volumes`,
`multicoil_val: 460 volumes`, the `[train_wandb]` line with
`num_cascades=12 ... lr=0.0003`, `resume from: nothing (fresh run)`, one
epoch, and a `saving model to output/checkpoints/...` line. When it
finishes, `runs/brain/model2/<Cluster>/checkpoints/last.ckpt` must exist.
That `<Cluster>` directory is a throwaway; delete it so it is not mistaken
for a real run. Note the transfer + extraction time it reports: that is the
fixed cost of every (re)start.

### 4.2 Model 2: accelerations 2, 4, 6, 8, one drawn per sample, from scratch

```bash
make submit-model2
```

This is `./submit.sh model2`, which submits `train.sub` with
`model=model2 accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04"`
and no `resume=`, so training starts from random weights with the
configuration in the table above. Run name and W&B run: `varnet-brain-model2`.
Checkpoints land in `runs/brain/model2/<Cluster>/`. The `.out` must show
`resume from: nothing (fresh run)` on this first submission.

### 4.3 Model 1: acceleration 4 only, from scratch (optional)

```bash
make submit-model1
```

Same, with `accelerations="4" center_fractions="0.08"`, run name
`varnet-brain-model1`, checkpoints in `runs/brain/model1/<Cluster>/`. It does
**not** start from model 2's weights; it is its own from-scratch run. The two
never share a job, a checkpoint directory, or a W&B run. Queue it whenever
there is a second slot; it is a Phase B reference point, not a prerequisite
for model 2.

### 4.4 Overrides

`make submit-model2 ARGS='...'` and `./submit.sh model2 name=value ...` are
the same thing. Any `name=value` is a `condor_submit` macro override, so every
knob in `train.sub` (`dataset`, `staging`, `train_data`, `val_data`,
`request_disk`, `gpu_mem`, `max_epochs`, `run_name`, `mask_type`,
`extra_args`, ...) can be set per submission without editing the file.

That last sentence is only true because every default in `train.sub` is
wrapped in `if ! defined`. `condor_submit` parses a command-line `name=value`
*as if it were the first line of the submit file*, so an unguarded
`model = model1` further down overwrites it, silently. On 2026-09-13 that cost
a run: `make submit-model2` produced a job ad reading
`--run_name varnet-model1 --accelerations 4`, so it trained model 1's schedule
and, because `--run_name` is also the W&B run id, logged into the existing
`varnet-model1` run instead of its own. Any knob added to `train.sub` needs the
same guard; `submit.sh` dry-runs before every submission and refuses to
submit if the preset did not reach the job ad.

Two exceptions, both verified with `-dry-run`: `request_cpus` and
`request_memory` are already in `condor_submit`'s macro table before the file
is read, so `if ! defined` never fires for them and guarding them would drop
the job to 1 CPU. Override those two as `cpus=16` / `mem=64GB`.

`extra_args` is appended verbatim to `train_wandb.py`, which accepts every
`pl.Trainer`, `FastMriDataModule` and `VarNetModule` argument. Ones worth
knowing for brain:

| `extra_args=` | Effect |
|---|---|
| `"--val_volume_sample_rate 0.25"` | validate on a quarter of the 460 volumes (chosen by the seeded shuffle, so the same quarter every run). Cuts per-epoch validation time; does not cut disk |
| `"--num_workers 0"` | if the log shows "Bus error" / "insufficient shared memory" (slower) |
| `"--num_cascades 8"` | the demo-script architecture, if 12 cascades OOM (a documented deviation) |
| `"--lr 0.001"` | the demo-script learning rate, if batch-1 training at 3e-4 is clearly too slow |
| `"--deterministic false"` | if an op has no deterministic CUDA kernel |

A second train batch: `train_data` takes a comma-separated list and
`run_train.sh` extracts every tarball it finds, so
`train_data="file://.../brain_multicoil_train_batch_0.tar.xz, file://.../brain_multicoil_train_batch_1.tar.xz" request_disk=800GB`
doubles the training set (and the extraction time and the disk).

## 5. Monitor

```bash
make status                       # condor_q -nobatch
make logs                         # tail -f the newest logs/*.out
make why JOB=<Cluster>            # hold reason + log tails for a held job
```

`stream_output = True` in `train.sub` means the `.out` file updates live:
Lightning's progress bar, the per-epoch `validation_loss`, and
`ModelCheckpoint`'s "Epoch N, global step M: 'validation_loss' reached ...
saving model to output/checkpoints/..." lines.

W&B project `fastmri-varnet-train` (override with `WANDB_PROJECT` in the
shell), run names `varnet-brain-model2` / `varnet-brain-model1`. The run id
is the run name, with `resume="allow"`, so a resubmitted job continues the
same W&B run rather than starting a new one; pass
`run_name=varnet-brain-model2-seed7` for a genuinely new run. The 2026-09
knee runs are `varnet-model1` / `varnet-model2` in the same project and are
unrelated. `MriModule` logs `validation_loss`, `val_metrics/nmse`,
`val_metrics/ssim`, `val_metrics/psnr` and up to 16 target / reconstruction /
error images per validation epoch (remember: a mixture over the four rates,
see "Masks"). Checkpoints are not uploaded (`log_model=False`); they come
back with the job. Without a key the run is written offline to
`runs/brain/<model>/<Cluster>/wandb/` and `wandb sync <that dir>/offline-run-*`
uploads it later.

## 6. What comes back

`runs/brain/<model>/<Cluster>/`:

- `checkpoints/last.ckpt`: the most recent epoch. This is what resumes.
- `checkpoints/epoch=N-step=M.ckpt`: the best `validation_loss` so far (`save_top_k=1`).
- `wandb/`: the offline run, only when no key was available.
- `lightning_logs/`: TensorBoard, only with `--no_wandb`.

Copy checkpoints back to the laptop with `scp` when a run is done; the
access point's home is 40 GB and one 12-cascade VarNet checkpoint is a few
hundred MB, so a handful of runs fit, a semester's worth does not.

## 7. Resuming

A 50-epoch run may not finish inside one GPU Lab job. Three cases:

1. **Evicted (preempted) mid-run.** `when_to_transfer_output =
   ON_EXIT_OR_EVICT` makes HTCondor transfer `output/` back on eviction and
   deliver it into the sandbox when the same job restarts. `run_train.sh` sees
   `output/checkpoints/` already populated, `train_wandb.py` picks
   `last.ckpt`, and training continues from that epoch. The tarballs are
   re-transferred and re-extracted (an hour or two for brain), which is the
   price of scratch being ephemeral. Nothing to do, but **confirm it once**
   (section 8).
2. **Removed because it hit the `+GPUJobLength` cap, or you removed it.** The
   final `output/` is in `runs/brain/<model>/<Cluster>/`. Resubmit pointing at it:
   ```bash
   make resume MODEL=model2 CKPT=runs/brain/model2/<Cluster>/checkpoints/last.ckpt
   ```
   `submit.sh` adds the file to `transfer_input_files`; `run_train.sh` moves
   it into `output/checkpoints/` (only if that directory is empty) and the
   driver resumes from it. Same `run_name`, same W&B run. Each resubmission
   gets a new `<Cluster>` directory; the newest one holds the latest
   `last.ckpt`.
3. **A fresh run of the same model.** Pass a new `run_name` and no `resume=`.

Lightning restores epoch, global step, optimizer and LR-scheduler state from
the checkpoint, so a resumed run is equivalent to an uninterrupted one apart
from data order within the interrupted epoch.

## 8. Things to verify on CHTC before trusting a multi-day run

- **Resume after eviction actually happens.** Submit a short job
  (`max_epochs=3`), let it finish one epoch, then
  `condor_vacate_job <Cluster>` from the access point. When it restarts, the
  `.out` must show `output/checkpoints already populated` and
  `resume from: output/checkpoints/last.ckpt`. If it shows a fresh run
  instead, the fallback is case 2 above (manual `make resume`), which does
  not depend on HTCondor's evict-transfer behaviour.
- **`/dev/shm` is big enough for 4 DataLoader workers.** A brain slice is
  up to ~40 MB of k-space and the workers hand batches over shared memory.
  If the log shows "Bus error" or "insufficient shared memory", add
  `extra_args="--num_workers 0"` (slower) or a smaller worker count.
- **12 cascades fit the GPU.** The 4.1 test job answers this on real slices.
  The first validation pass is the other place memory can spike (largest
  volume in the set); `--limit_val_batches 10` in 4.1 does not cover that, so
  watch the first full validation epoch of the real run.
- **`deterministic=True`** (the leaderboard default, kept) makes torch raise
  on any op without a deterministic CUDA kernel. The fastMRI authors ran this
  configuration, so it is expected to be fine; `extra_args="--deterministic
  false"` is the escape hatch.

## Known deviations from Sriram et al. 2020

Read these before comparing any number to the paper (CLAUDE.md, ROADMAP.md,
`VERIFICATION.md` sections 1 and 5.4):

- **Data.** One NYU batch per split: 455 of the 4,469 brain training volumes
  and 460 of the 1,378 validation volumes. The paper trained on the full
  split (and on train+val for its test-set table). Every Phase A / Phase B
  comparison must use the same two batches; absolute numbers are not
  comparable to the paper. The brain test split now has public ground truth
  (`brain_multicoil_test_full`, in the group directory), so a test-set number
  in the paper's Table 3 sense *is* computable later, unlike knee.
- **Centre lines.** `center_fractions` gives a variable number of centre
  lines (scales with k-space width) instead of the paper's fixed count.
- **Mask family.** `equispaced_fraction` (realised rate = nominal) rather
  than the paper's uncorrected M_e or the leaderboard's `equispaced`;
  reasons under "Masks".
- **Joint training.** The paper trains one model per rate. model1 matches
  that for 4x. model2 is the repo's joint-training pattern extended to four
  rates and is a Phase B baseline, not a paper reproduction. There is no
  acceleration-rate input to the network in either model.
- **Effective batch size.** 1 (one GPU) against the leaderboard's 32; the
  paper does not state one. Same learning rate.
- **2x and 6x** are not in the paper's brain results or in the fastMRI
  challenge; their centre fractions follow the 0.32 / R convention.
- **Validation** is a fixed mixture over rates (see "Masks"); the paper's
  tables are per rate. Use `../verification/` for per-rate numbers.

## Without Docker

`train_wandb.py` needs only the `fastmri` package and its dependencies plus
`wandb`. In the `Fall26Research` conda env:

```bash
conda install -n Fall26Research python=3.9
conda run -n Fall26Research pip install torch "pytorch-lightning==1.9.5" "torchmetrics==0.11.4" \
    "numpy<2" h5py runstats "scikit-image<0.23" pyyaml "pandas<2.1" wandb requests tqdm
conda run -n Fall26Research pip install --no-deps -e ../fastMRI
conda run -n Fall26Research python train_wandb.py --data_path <dir with multicoil_train,multicoil_val> \
    --default_root_dir output --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
    --gpus 0 --no_wandb --fast_dev_run 1
```

The Lightning 1.x requirement is the one that cannot be relaxed.
