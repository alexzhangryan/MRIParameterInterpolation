# training/ — E2E VarNet on CHTC, model 1 and model 2

Runbook for the two training runs. Read top to bottom the first time; after
that, section 4 is the only part you come back to.

*Status 2026-09-11: everything here is written and passes the local smoke
tests (`make smoke`, `make job-smoke`) on synthetic data. Nothing in this
directory has run on CHTC yet — no job submitted, and the subsets in section
3b do not exist yet.*

Nothing here modifies `fastMRI/`, `parameter_interpolation/`, or any file
outside this directory. The job runs the `fastmri` package as installed in the
Docker image (pinned to the same commit the `fastMRI/` submodule is checked
out at) through a small driver, `train_wandb.py`, that builds the same
`VarNetModule` and `FastMriDataModule` as
`fastmri_examples/varnet/train_varnet_demo.py`.

## The two models

| Model | `accelerations` | `center_fractions` | What it is |
|---|---|---|---|
| **model1** | `4` | `0.08` | One network, one rate (4x). Phase A reproduction target. |
| **model2** | `2 4 6 8` | `0.16 0.08 0.0533 0.04` | One network, four rates. For every training sample the mask function draws one (acceleration, center fraction) pair uniformly at random from the four. No explicit rate signal reaches the network: this is the "joint training, no conditioning" baseline that Phase B's DPI model will be measured against. |

The two lists are paired elementwise by `fastmri.data.subsample.MaskFunc`;
center fractions follow the fastMRI convention of 0.32 / R.

## Both models are trained from scratch with the fastMRI defaults

"From scratch" means random initialisation, no pretrained weights, no
checkpoint. The only things that differ between the two runs are the two
mask lists above. Every other setting is the `train_varnet_demo.py` default
from the fastMRI repo, hard-wired as the defaults of `train_wandb.py`:

| Setting | Value | Flag (only if you want to change it) |
|---|---|---|
| Unrolled cascades | 8 | `--num_cascades` |
| Regulariser U-Net channels / pooling layers | 18 / 4 | `--chans` / `--pools` |
| Sensitivity-map U-Net channels / pooling layers | 8 / 4 | `--sens_chans` / `--sens_pools` |
| Optimiser | Adam, lr 0.001, weight decay 0 | `--lr`, `--weight_decay` |
| LR schedule | step: x0.1 at epoch 40 | `--lr_step_size`, `--lr_gamma` |
| Loss | SSIM (fixed in `VarNetModule`) | |
| Batch size | 1 | `--batch_size` |
| Epochs | 50 | `--max_epochs` |
| Mask type | `equispaced_fraction` | `--mask_type` |
| Challenge | `multicoil` | `--challenge` |
| Seed | 42 | `--seed` |
| Deterministic CUDA | on | `--deterministic false` |
| Training data | all of `multicoil_train` (`sample_rate` 1.0) | `--sample_rate` |
| Validation data | all of `multicoil_val` | |
| GPUs | 1 | `--gpus` |

The one deviation from the demo script is GPUs: the demo asks for 2 with DDP,
`train.sub` requests 1 and passes `--gpus 1`. Batch size is per GPU, so this
halves the effective batch size from 2 to 1 and doubles the number of
optimiser steps per epoch. Every other default is the same.

The full command each submission resolves to, exactly as `run_train.sh` runs
it inside the job (`train.sub` `arguments` line plus the two flags
`run_train.sh` adds):

```bash
# model 1
python train_wandb.py --data_path data/knee --default_root_dir output \
    --run_name varnet-model1 --accelerations 4 --center_fractions 0.08 \
    --mask_type equispaced_fraction --max_epochs 50 --gpus 1

# model 2
python train_wandb.py --data_path data/knee --default_root_dir output \
    --run_name varnet-model2 --accelerations 2 4 6 8 --center_fractions 0.16 0.08 0.0533 0.04 \
    --mask_type equispaced_fraction --max_epochs 50 --gpus 1
```

For model 2, `create_mask_for_mask_type("equispaced_fraction", [0.16, 0.08,
0.0533, 0.04], [2, 4, 6, 8])` builds a single `MaskFunc`; on every training
slice it calls `choose_acceleration`, which picks an index uniformly at
random and uses that (center fraction, acceleration) pair. The network is
never told which one was picked.

A fresh run starts from scratch automatically: `train_wandb.py` only resumes
when `output/checkpoints/` already holds a file, which in a new job happens
only if you pass `resume=` or HTCondor brings back an evicted job's output.
The first submission of each model has neither, and the `.out` log confirms
it with `resume from: nothing (fresh run)`.

**These are two separate training runs, done one after the other.** Each
submission queues exactly one HTCondor job for exactly one model. Train
model 1 to completion first (resuming as needed), then start model 2. The two
never share a job, a checkpoint directory (`runs/model1/`, `runs/model2/`),
or a W&B run (`varnet-model1`, `varnet-model2`). Nothing stops you queueing
both at once, but that is not the plan.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `train_wandb.py` | inside the job (or any `fastmri` env) | The driver. W&B logger, `last.ckpt`, auto-resume from `output/checkpoints/`. |
| `run_train.sh` | inside the job | Job executable: extracts the tarballs, finds the splits, seeds a resume checkpoint, runs the driver, collects `output/`. |
| `train.sub` | access point | HTCondor submit file, container universe. Model 1 defaults; model 2 via macros. |
| `submit.sh` | access point | `./submit.sh model1\|model2 [name=value ...]`. Refuses to submit without a W&B key unless `OFFLINE=1`. |
| `Makefile` | laptop + access point | `make build/push/smoke/job-smoke` (Docker) and `make submit-model1/submit-model2/resume/status/logs` (condor). |
| `Dockerfile` | laptop (build), CHTC (run) | Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, wandb, fastMRI at `91f2df4`. Pinned to `linux/amd64`. |
| `make_subset.sh` | inside the job | One-off (section 3b): cuts `/staging` down to subsets that fit the quota. `val` extracts and re-tars N volumes; `train` streams a prefix of NYU's `train_batch_0` and re-packs the complete volumes. Validates every kept `.h5` with h5py. |
| `subset_val.sub`, `subset_train.sub` | access point | The two CPU jobs that run `make_subset.sh`. Run once, in that order. |
| `.env.example` | access point | Template for `.env` (gitignored): `WANDB_API_KEY`, `WANDB_ENTITY`, `FASTMRI_TRAIN_URL`. `submit.sh` and the subset targets source `.env` themselves. |

## Where things run

| Machine | How you get there | What happens there |
|---|---|---|
| Laptop (this Mac) | | Build and push the Docker image. Smoke-test on synthetic data. |
| CHTC access point (the "VM") | `ssh apryan3@ap2001.chtc.wisc.edu` | Holds this directory, submits jobs, receives checkpoints and logs. 40 GB home quota: code only. |
| CHTC transfer node | `ssh apryan3@transfer.chtc.wisc.edu` | Bulk data downloads into `/staging/a/apryan3/fastmri/`. |
| CHTC execute node | never directly | Pulls the image from Docker Hub, runs `run_train.sh`, is wiped when the job ends. |

Personal staging is sharded by the first letter of the netid:
`/staging/a/apryan3`, **not** `/staging/apryan3`. The group directory
`/staging/groups/kamilov_group/Kamilov-SciAI-datasets` is not readable by this
account as of 2026-09-10; if that changes, the `staging` macro in `train.sub`
is the only thing to repoint.

## 0. Before anything

- **Rotate the W&B key** that is in the old `Research/inpainting.sub` git
  history (ROADMAP.md security note). Every `WANDB_API_KEY` below means the
  new key. It is only ever exported in a shell, never written to a file in
  this repo.
- **Training data is the gate.** `train.sub` expects
  `/staging/a/apryan3/fastmri/knee_multicoil_train_subset.tar.xz` and
  `knee_multicoil_val_subset.tar`, built once by section 3b from the full
  val tarball already staged and a streamed prefix of NYU's train tarball.
  The full splits do not fit the 100 GB personal staging quota (val alone is
  100.7 GB; `multicoil_train` is ~490 GB compressed). A quota increase or
  group-directory access is the only way to train on all of it.
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
`WandbLogger`, and asserts Lightning is 1.x.

Then set one line in `train.sub` and keep it that way in git:

```
image      = docker://<dockerhub_user>/fastmri-train:2026-09
```

It is currently `docker://genjigod/fastmri-train:2026-09`. Change it if you
push under a different Docker Hub account; `submit.sh` only refuses to submit
when it still says `CHANGE_ME`.

## 2. Laptop: smoke-test with no real data (minutes each, under emulation)

```bash
make smoke        # train_wandb.py on synthetic phantoms, 1 epoch, then a rerun that MUST resume from last.ckpt
make job-smoke    # run_train.sh exactly as HTCondor runs it, twice: fresh, then "after eviction"
```

Both generate tiny fake `multicoil_train` / `multicoil_val` directories with
`../verification/make_synthetic_val.py`, train a 2-cascade model on CPU, and
fail loudly if `output/checkpoints/last.ckpt` is not written or if the second
run does not print `resume from: output/checkpoints/last.ckpt`. `make smoke`
uses the model 2 rate list, `make job-smoke` the model 1 list, so both mask
configurations get exercised. Green here means the image, the driver, the
job executable, extraction, and the checkpoint/resume contract all work
before a single GPU slot is used.

## 3. Copy to the access point (the "VM")

Only this directory is needed on CHTC. Neither submodule is: the image
carries its own copy of fastMRI at the pinned commit. Cloning the whole repo
on the access point (without `--recurse-submodules`) is the simplest way.

What goes up:

| Copy | Why |
|---|---|
| `train.sub` | submit file, with your `image =` line edited |
| `submit.sh` | wraps `condor_submit`, applies the model presets |
| `run_train.sh` | the job executable (transferred to the execute node by HTCondor) |
| `train_wandb.py` | the driver (listed in `transfer_input_files`) |
| `Makefile` | for `make submit-model1` etc. Optional; `./submit.sh` works alone. |
| `make_subset.sh`, `subset_val.sub`, `subset_train.sub`, `.env.example` | the one-off staging repack in 3b |

What must not go up: `synthetic/`, `jobtest/`, `output/`, `*.tar`, `*.ckpt`,
`.env`, `.make/`, `dataset_cache.pkl`. They are laptop smoke-test leftovers.
(`dataset_cache.pkl` is a fastMRI slice-index cache keyed by data path. One
from a synthetic smoke run is currently committed by mistake; it is harmless
on CHTC — the paths in it do not match, so `SliceDataset` just rebuilds — but
it should not be there.) A `.ckpt` in the
directory is harmless (only the one named in `resume=` is transferred), but
the tarballs and phantoms would just waste home quota.

```bash
# laptop, from the repo root
ssh apryan3@ap2001.chtc.wisc.edu 'mkdir -p ~/Fall26Research/training'
scp training/train.sub training/submit.sh training/run_train.sh training/train_wandb.py training/Makefile \
    training/make_subset.sh training/subset_val.sub training/subset_train.sub training/.env.example \
    apryan3@ap2001.chtc.wisc.edu:~/Fall26Research/training/
```

Or, if the repo is cloned on the access point, `git pull` there. The
submodules can stay uninitialised.

Then, on the access point, one-time checks:

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research/training
chmod +x run_train.sh submit.sh
grep '^image' train.sub                 # must not say CHANGE_ME
get_quotas /staging/a/apryan3           # 100 GB / 1000 items by default
ls -la /staging/a/apryan3/fastmri/      # what is actually staged
```

Let `condor_submit` parse the file without queueing anything. This costs
nothing and catches macro or syntax mistakes on the access point itself:

```bash
condor_submit -dry-run /dev/stdout train.sub model=model2 accelerations="2 4 6 8" \
    center_fractions="0.16 0.08 0.0533 0.04" \
  | grep -iE "^(Arguments|TransferInput|RequestDisk|Requirements) "
```

### 3a. Stage the data (transfer node)

```bash
ssh apryan3@transfer.chtc.wisc.edu           # bulk data host, not the access point
ls -la /staging/a/apryan3/fastmri/           # val tarball + checkpoint from verification/prepare_staging.sh
```

`verification/prepare_staging.sh` fetches `knee_multicoil_val.tar.xz` from
the NYU presigned URL and verifies it (see `verification/README.md` step 3).
The training split ships as five batches (`knee_multicoil_train_batch_0..4.tar.xz`, ~91 GB each, same URL pattern); they would go in
the same directory the same way once there is room for it.

### 3b. Repack /staging into subsets (what actually fits under the quota)

`/staging/a/apryan3` is capped at 100 GB and the full val tarball alone is
100.7 GB. The train split (five ~91 GB batches, ~930 GB unpacked) cannot be
staged at all without a quota increase, and that request is the last resort.
So both splits are cut down to subsets that fit together:

| File in `/staging/a/apryan3/fastmri/` | Built by | Size | Contents |
|---|---|---|---|
| `knee_multicoil_val_subset.tar` | `make subset-val` | ~20 GB | `multicoil_val/`, 20 of the 199 volumes, evenly spaced through the sorted list |
| `knee_multicoil_train_subset.tar.xz` | `make subset-train` | ~65 GB | `multicoil_train/`, every complete volume in the first 65 GB of NYU's `train_batch_0` (~120 of 973) |

`train.sub` defaults to these two names, so once they exist `make submit-model1`
works with no overrides. Both jobs run `make_subset.sh` in the training image
on a CPU slot, validate every kept `.h5` by reading it fully, and write a
`*_subset_files.txt` next to the tarball listing exactly which volumes went in
(plus the archive's sha256). Keep those two text files in the repo directory;
they are the definition of the subset every model in Phase A and B trains on.

**The order matters.** Staging is full, so the val subset has to come back to
home first, the original val tarball is deleted by hand, and only then can
the train subset be written into staging.

**Step 1: val subset (about an hour, mostly extraction).**

```bash
cd ~/Fall26Research/training
make subset-val                      # N_VAL=30 for more volumes; ~1 GB each, home is 40 GB
condor_q -nobatch                    # < = transferring the 101 GB input, R = running
tail -f logs/subset_val_*.out
```

When it finishes, `knee_multicoil_val_subset.tar` and `val_subset_files.txt`
are in this directory. Check, then swap:

```bash
tail -3 val_subset_files.txt                      # sha256 and byte count
tar -tf knee_multicoil_val_subset.tar | head -3   # multicoil_val/file...h5
rm /staging/a/apryan3/fastmri/knee_multicoil_val.tar.xz      # the full val split; see below
mv knee_multicoil_val_subset.tar /staging/a/apryan3/fastmri/
get_quotas /staging/a/apryan3                     # should now show ~80 GB free
```

Deleting the full val tarball means Tier 1 of `VERIFICATION.md` (the released
checkpoint on all 199 val volumes) can no longer run without re-downloading
it. Run Tier 1 first if you want that number; otherwise the NYU URL can be
used again any time before it expires (~2026-12-08) or re-requested.

**Step 2: train subset (about an hour: streaming, decoding, re-packing).**

NYU ships the train split as five batches. Put the presigned URL for
`knee_multicoil_train_batch_0.tar.xz` (the line in the approval email ending
in `--output knee_multicoil_train_batch_0.tar.xz`) in `.env`
(`cp .env.example .env`, `chmod 600 .env`, paste it quoted). Then:

```bash
make subset-train                    # TRAIN_PREFIX_GB=50 to fit a smaller quota gap
tail -f logs/subset_train_*.out
```

The job fetches only the first `TRAIN_PREFIX_GB` gigabytes of that batch
with an HTTP range request and decodes them on the fly; nothing compressed is
ever stored. The archive is sequential, so that prefix is the same volumes
every time. The truncated last volume is dropped, the rest are re-packed, and
HTCondor writes the result straight into `/staging` via `transfer_output_remaps`.
`train_subset_files.txt` lands in this directory. If the job goes on hold
after finishing, the output transfer hit the quota: lower `TRAIN_PREFIX_GB`,
or free space, then `condor_release` it.

Then:

```bash
ls -la /staging/a/apryan3/fastmri/
condor_submit -dry-run /dev/stdout train.sub | grep -iE "^(TransferInput|RequestDisk) "
```

**What this costs scientifically.** Model 1 trains on ~120 volumes instead
of 973, so its numbers will not match the paper; see "Known deviations". For
Phase B this is fine as long as model 1, model 2 and the DPI model all train
and validate on the same two subsets, which the `*_subset_files.txt` lists
pin down. If the full train split is ever staged, `train_data=` and
`request_disk=` on the command line switch a run back to it.

## 4. Run the two models (access point)

Every submission starts the same way:

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research/training
# either export them, or put them in .env (cp .env.example .env; chmod 600 .env);
# submit.sh sources .env itself.
export WANDB_API_KEY=...          # freshly rotated
export WANDB_ENTITY=...           # optional
```

### 4.1 Short test job first (model 1, one epoch, 50 batches)

Checks the image, the data path, `/dev/shm`, and the GPU before committing a
slot for days:

```bash
make submit-model1 ARGS='max_epochs=1 extra_args="--limit_train_batches 50 --limit_val_batches 10"'
make logs                          # tail -f the newest logs/*.out
```

The `.out` must show `[run_train] multicoil_train: N volumes`,
`resume from: nothing (fresh run)`, one epoch, and a
`saving model to output/checkpoints/...` line. When it finishes,
`runs/model1/<Cluster>/checkpoints/last.ckpt` must exist. That `<Cluster>`
directory is a throwaway; delete it so it is not mistaken for a real run.

### 4.2 Model 1: acceleration 4 only, from scratch

```bash
make submit-model1
```

This is `./submit.sh model1`, which submits `train.sub` with
`model=model1 accelerations="4" center_fractions="0.08"` and no `resume=`,
so training starts from random weights with the defaults listed at the top
of this file. Run name and W&B run: `varnet-model1`. Checkpoints land in
`runs/model1/<Cluster>/`. The `.out` must show
`resume from: nothing (fresh run)` on this first submission.

### 4.3 Model 2: accelerations 2, 4, 6, 8, one drawn per sample, from scratch

Start after model 1 has finished (all 50 epochs, however many resubmissions
that took). Model 2 does **not** start from model 1's weights; it is its own
from-scratch run.

```bash
make submit-model2
```

This is `./submit.sh model2`, which submits with `model=model2
accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04"` and no
`resume=`. Run name and W&B run: `varnet-model2`. Checkpoints land in
`runs/model2/<Cluster>/`.

### 4.4 Overrides

`make submit-model1 ARGS='...'` and `./submit.sh model1 name=value ...` are
the same thing. Any `name=value` is a `condor_submit` macro override, so every
knob in `train.sub` (`request_disk`, `max_epochs`, `run_name`, `train_data`,
`val_data`, `mask_type`, `extra_args`, ...) can be set per submission without
editing the file. `extra_args` is appended verbatim to `train_wandb.py`,
which accepts every `pl.Trainer`, `FastMriDataModule` and `VarNetModule`
argument (`--num_workers`, `--sample_rate`, `--lr`, `--deterministic false`,
...).

Size `request_disk` from what is actually staged: peak scratch is every
tarball plus its extracted contents at the same time, because `run_train.sh`
deletes a tarball only after its extraction finishes. For the section-3b
subsets that is (65 GB train `.tar.xz` -> ~125 GB) + (20 GB val `.tar`, no
compression) ~= 210 GB, which is what the `request_disk = 260GB` default in
`train.sub` covers. Pointing `train_data=` at a full NYU batch instead needs
roughly 91 + 175 GB for that file alone: override with
`./submit.sh model1 train_data=... request_disk=400GB`.

## 5. Monitor

```bash
make status                       # condor_q -nobatch
make logs                         # tail -f the newest logs/*.out
condor_q -l <Cluster> | grep -i hold    # why a job is held, if it is
```

`stream_output = True` in `train.sub` means the `.out` file updates live:
Lightning's progress bar, the per-epoch `validation_loss`, and
`ModelCheckpoint`'s "Epoch N, global step M: 'validation_loss' reached ...
saving model to output/checkpoints/..." lines.

W&B project `fastmri-varnet-train` (override with `WANDB_PROJECT` in the
shell), run name `varnet-model1` / `varnet-model2`. The run id is the run
name, with `resume="allow"`, so a resubmitted job continues the same W&B run
rather than starting a new one; pass `run_name=varnet-model1-seed7` for a
genuinely new run. `MriModule` logs `validation_loss`, `val_metrics/nmse`,
`val_metrics/ssim`, `val_metrics/psnr` and up to 16 target / reconstruction /
error images per validation epoch; all of it lands in W&B through the
logger. Checkpoints are not uploaded (`log_model=False`); they come back
with the job. Without a key the run is written offline to
`runs/<model>/<Cluster>/wandb/` and `wandb sync <that dir>/offline-run-*`
uploads it later.

## 6. What comes back

`runs/<model>/<Cluster>/`:

- `checkpoints/last.ckpt`: the most recent epoch. This is what resumes.
- `checkpoints/epoch=N-step=M.ckpt`: the best `validation_loss` so far (`save_top_k=1`).
- `wandb/`: the offline run, only when no key was available.
- `lightning_logs/`: TensorBoard, only with `--no_wandb`.

Copy checkpoints back to the laptop with `scp` when a run is done; the
access point's home is 40 GB and one VarNet checkpoint is a few hundred MB,
so a handful of runs fit, a semester's worth does not.

## 7. Resuming

A 50-epoch multicoil run will not finish inside one GPU Lab job. Three cases:

1. **Evicted (preempted) mid-run.** `when_to_transfer_output =
   ON_EXIT_OR_EVICT` makes HTCondor transfer `output/` back on eviction and
   deliver it into the sandbox when the same job restarts. `run_train.sh` sees
   `output/checkpoints/` already populated, `train_wandb.py` picks
   `last.ckpt`, and training continues from that epoch. The tarballs are
   re-transferred and re-extracted, which is the price of scratch being
   ephemeral. Nothing to do, but **confirm it once** (section 8).
2. **Removed because it hit the `+GPUJobLength` cap, or you removed it.** The
   final `output/` is in `runs/<model>/<Cluster>/`. Resubmit pointing at it:
   ```bash
   make resume MODEL=model1 CKPT=runs/model1/<Cluster>/checkpoints/last.ckpt
   make resume MODEL=model2 CKPT=runs/model2/<Cluster>/checkpoints/last.ckpt
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

## 8. Two things to verify on CHTC before trusting a multi-day run

- **Resume after eviction actually happens.** Submit a short job
  (`max_epochs=3`, a subset tarball), let it finish one epoch, then
  `condor_vacate_job <Cluster>` from the access point. When it restarts, the
  `.out` must show `output/checkpoints already populated` and
  `resume from: output/checkpoints/last.ckpt`. If it shows a fresh run
  instead, the fallback is case 2 above (manual `make resume`), which does
  not depend on HTCondor's evict-transfer behaviour.
- **`/dev/shm` is big enough for 4 DataLoader workers.** One knee slice is
  ~28 MB and the workers hand batches over shared memory. If the log shows
  "Bus error" or "insufficient shared memory", add
  `extra_args="--num_workers 0"` (slower) or a smaller worker count.

Also worth a glance on the first real job: `deterministic=True` (the demo
default, kept) makes torch raise on any op without a deterministic CUDA
kernel. The fastMRI authors ran this configuration, so it is expected to be
fine; `extra_args="--deterministic false"` is the escape hatch.

## Known deviations from Sriram et al. 2020

Read these before comparing any number to the paper (CLAUDE.md, ROADMAP.md):

- `center_fractions` gives a variable number of centre lines (29 / 15 at 368
  wide for 0.08 / 0.04) instead of the paper's fixed 30 / 16.
- `equispaced_fraction` is the corrected-density mask family and corresponds
  to neither of the paper's tables; `random` is the fastMRI knee leaderboard
  convention. Pass `mask_type=random` to match the third-party reference
  values in `VERIFICATION.md` Section 2.
- The paper trains one model per rate. model1 matches that for 4x. model2 is
  the repo's joint-training pattern extended to four rates and is a Phase B
  baseline, not a paper reproduction. There is no acceleration-rate input to
  the network in either model.
- Validation here is the `multicoil_val` split; the paper's Table 3 numbers
  are on a test split with no public ground truth and cannot be reproduced.
- Under the default `/staging` quota both models train on the section-3b
  subsets: ~120 of the 973 train volumes and 20 of the 199 val volumes. At
  the same 50-epoch schedule that is ~8x fewer optimiser steps than the
  paper. Every Phase A/B comparison must use the same two subsets
  (`*_subset_files.txt`); absolute numbers are not comparable to the paper.

## Without Docker

`train_wandb.py` needs only the `fastmri` package and its dependencies plus
`wandb`. In the `Fall26Research` conda env:

```bash
conda install -n Fall26Research python=3.9
conda run -n Fall26Research pip install torch "pytorch-lightning==1.9.5" "torchmetrics==0.11.4" \
    "numpy<2" h5py runstats "scikit-image<0.23" pyyaml "pandas<2.1" wandb requests tqdm
conda run -n Fall26Research pip install --no-deps -e ../fastMRI
conda run -n Fall26Research python train_wandb.py --data_path <dir with multicoil_train,multicoil_val> \
    --default_root_dir output --accelerations 4 --center_fractions 0.08 --gpus 0 --no_wandb --fast_dev_run 1
```

The Lightning 1.x requirement is the one that cannot be relaxed.
