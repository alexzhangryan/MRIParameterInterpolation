# Fall26Research: E2E VarNet on fastMRI brain, then Deep Parameter Interpolation

Two phases (see `CLAUDE.md`, `ROADMAP.md`):

- **Phase A**: train End-to-End VarNet baselines on the fastMRI multicoil
  **brain** dataset on UW-Madison CHTC (HTCondor). Started on knee; moved to
  brain on 2026-09-18 once the group staging directory held the full set.
- **Phase B**: extend VarNet with Deep Parameter Interpolation (Park et al.,
  CVPR 2026) to condition on acceleration rate. Designed in `plan.md`, not
  started.

## Status (2026-09-18)

The brain set (20 NYU tarballs, ~1.37 TB) is in the Kamilov group staging
directory, transferred and SHA256-verified. `training/` is repointed at it
with the paper's configuration (12 cascades, Adam 3e-4) and the meeting's
rate list (2, 4, 6, 8), and passes the local smoke tests. **The brain runs
have not been submitted yet.** `ROADMAP.md` has the per-item checklist.

Next CHTC session: `git pull`, confirm the two batch-0 tarballs with `ls`,
the one-epoch test job, then `make submit-model2` (`training/README.md`
sections 3a and 4).

## Layout

| Path | What | Runbook |
|---|---|---|
| `verification/` | Evaluate the released VarNet checkpoint on `multicoil_val` (Tiers 0 and 1 of `VERIFICATION.md`). Proves the environment and the metric code before any training. | `verification/README.md` |
| `training/` | Train **model 2** (accelerations 2, 4, 6, 8, one drawn at random per sample; the primary run) and **model 1** (acceleration 4 only) on brain. Also holds the knee-era `/staging` repack jobs. | `training/README.md` |
| `fastMRI/` | Submodule, `facebookresearch/fastMRI` at commit `91f2df4`. Never modified. Not needed on CHTC: the images carry their own copy. | |
| `parameter_interpolation/` | Submodule, the DPI reference code. Phase B only. | |
| `Claude outputs/` | Archived first draft of the verification harness, superseded by `verification/`. Do not run. | |

| Document | What it is |
|---|---|
| `ROADMAP.md` | The live checklist. Check it before assuming what is done. |
| `VERIFICATION.md` | Which numbers from Sriram et al. 2020 can actually be checked, how, and what to say when they cannot. Defines Tiers 0-3. |
| `plan.md` | Phase B design: the DPI mechanism, the files to create, the scope table, related work. |
| `CLAUDE.md` | Project context for Claude Code sessions. |

## The training runs

| Model | `accelerations` | `center_fractions` | Purpose |
|---|---|---|---|
| model2 | `2 4 6 8` | `0.16 0.08 0.0533 0.04` | Joint training on four rates with no rate signal: the Phase B baseline. **The run the 2026-09-18 meeting asked for.** |
| model1 | `4` | `0.08` | One rate; the 4x point of the per-rate ceiling Phase B compares against. Optional. |

Both are trained from scratch (random init, no checkpoint) with the paper's
and the fastMRI leaderboard script's configuration for everything except the
mask lists: 12 cascades, 18 channels, Adam lr 3e-4 with a x0.1 step at epoch
40, batch 1, 50 epochs, `equispaced_fraction` masks, seed 42, one GPU. Neither
starts from the other's weights.

```bash
# on ap2001.chtc.wisc.edu, in ~/Fall26Research/training, WANDB_API_KEY exported
make submit-model2        # brain, accelerations 2 4 6 8, one drawn per sample
make submit-model1        # brain, acceleration 4 (optional)
```

Full instructions, including the short test job to run first, the
configuration audit against the paper, how the masks are generated, and how
to resume and monitor, are in `training/README.md`.

## Data: the brain set in group staging

`/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/` holds
all 20 NYU brain tarballs (10 train batches of ~450 volumes, 3 val batches
of ~460, 3 test, 3 fully-sampled test, DICOM; ~1.37 TB compressed), verified
against NYU's `SHA256`. The directory has a 10,000-item cap, so nothing is
extracted there; a job pulls one `.tar.xz` per split into scratch and
unpacks it.

`train.sub` defaults to `brain_multicoil_train_batch_0.tar.xz` (455 volumes)
and `brain_multicoil_val_batch_0.tar.xz` (460 volumes): one batch per split
is what fits a job's scratch, and it is the same volumes every time. Every
Phase A and Phase B model must train and validate on those same two batches.
**Absolute numbers are not comparable to the paper**, which trained on all
4,469 brain volumes; see "Known deviations" in `training/README.md`.

The knee subsets from 2026-09 (`knee_multicoil_{train,val}_subset` in
`/staging/a/apryan3/fastmri/`, built under the 100 GB personal quota) are
superseded and still reachable with `dataset=knee` overrides.

## What runs where

| Machine | Purpose |
|---|---|
| Laptop (Apple Silicon) | Build the `linux/amd64` Docker images, push to Docker Hub, smoke-test on synthetic data |
| `ap2001.chtc.wisc.edu` (access point) | Holds `verification/` and `training/`, submits jobs, receives results |
| `transfer.chtc.wisc.edu` | Downloaded the NYU tarballs into `/staging` (brain: group directory; knee: `/staging/a/apryan3/fastmri/`) |

Both Docker images must be built for `linux/amd64`; the Dockerfiles pin it
and the Makefiles pass `--platform`. A bare arm64 build fails in the conda
layer with `hdf5 1.10.6 nompi_h6a2412b_1114 does not exist`.

`training/train.sub` already points at
`docker://genjigod/fastmri-train:2026-09`. `verification/verify.sub` still
says `CHANGE_ME` — edit it before the first verification submit.

## Secrets

The Weights & Biases key and the NYU presigned URLs live in a single
untracked `.env` at the repo root (`cp .env.example .env; chmod 600 .env`),
shared by `training/` and `verification/` alike — both stages' `submit.sh`
and `Makefile` source `../.env` relative to themselves, so there is one key
to paste and one file to rotate. They can also just be exported in the
submitting shell; the file wins over the shell when both are set.
`WANDB_PROJECT` is deliberately not in it — the two stages log to different
projects and each `.sub` sets its own. They reach the job through HTCondor
`getenv`, and are never written to a `.sub` file or anything committed.
Rotate the key that is in the old `Research/inpainting.sub` history before
using any key here.
