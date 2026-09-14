# Fall26Research: E2E VarNet on fastMRI knee, then Deep Parameter Interpolation

Two phases (see `CLAUDE.md`, `ROADMAP.md`):

- **Phase A**: reproduce End-to-End VarNet on the fastMRI knee dataset on
  UW-Madison CHTC (HTCondor), and train two baseline models.
- **Phase B**: extend VarNet with Deep Parameter Interpolation (Park et al.,
  CVPR 2026) to condition on acceleration rate. Designed in `plan.md`, not
  started.

## Status (2026-09-11)

Both stages are built, documented, and green on synthetic data locally.
**Nothing has run on CHTC yet**: no job submitted, no verification tier
executed, no training started, and `/staging` not confirmed to hold any
fastMRI data. `ROADMAP.md` has the per-item checklist.

Next CHTC session, in order: confirm what is in `/staging` → push both images
→ Tier 0 → Tier 1 (on the full val split, while it still exists) → the
section 3b subset repack → `make submit-model1`.

## Layout

| Path | What | Runbook |
|---|---|---|
| `verification/` | Evaluate the released VarNet checkpoint on `multicoil_val` (Tiers 0 and 1 of `VERIFICATION.md`). Proves the environment and the metric code before any training. | `verification/README.md` |
| `training/` | Train **model 1** (acceleration 4 only) and **model 2** (accelerations 2, 4, 6, 8, one drawn at random per sample). Also holds the one-off `/staging` repack jobs. | `training/README.md` |
| `fastMRI/` | Submodule, `facebookresearch/fastMRI` at commit `91f2df4`. Never modified. Not needed on CHTC: the images carry their own copy. | |
| `parameter_interpolation/` | Submodule, the DPI reference code. Phase B only. | |
| `Claude outputs/` | Archived first draft of the verification harness, superseded by `verification/`. Do not run. | |

| Document | What it is |
|---|---|
| `ROADMAP.md` | The live checklist. Check it before assuming what is done. |
| `VERIFICATION.md` | Which numbers from Sriram et al. 2020 can actually be checked, how, and what to say when they cannot. Defines Tiers 0-3. |
| `plan.md` | Phase B design: the DPI mechanism, the files to create, the scope table, related work. |
| `CLAUDE.md` | Project context for Claude Code sessions. |

## The two training runs

| Model | `accelerations` | `center_fractions` | Purpose |
|---|---|---|---|
| model1 | `4` | `0.08` | Phase A reproduction target, one rate |
| model2 | `2 4 6 8` | `0.16 0.08 0.0533 0.04` | Joint training on four rates with no rate signal: the Phase B baseline |

Both are trained from scratch (random init, no checkpoint) with the
`train_varnet_demo.py` defaults for everything except the mask lists: 8
cascades, 18 channels, Adam lr 1e-3 with a x0.1 step at epoch 40, batch 1,
50 epochs, `equispaced_fraction` masks, seed 42, one GPU. They run one after
the other, model 1 first, and model 2 does not start from model 1's weights.

```bash
# on ap2001.chtc.wisc.edu, in ~/Fall26Research/training, WANDB_API_KEY exported
make submit-model1        # acceleration 4
make submit-model2        # accelerations 2 4 6 8, one drawn per sample
```

Full instructions, including what to copy to CHTC, the short test job to run
first, and how to resume and monitor, are in `training/README.md`.

## Data: the 100 GB staging quota shapes everything

`/staging/a/apryan3` is capped at 100 GB. The full val tarball alone is
100.7 GB and the train split ships as five ~91 GB batches, so neither full
split fits, and requesting an increase is a last resort (decision 2026-09-11).
Instead `/staging` holds two subsets, built once by `training/README.md`
section 3b:

| File | Built by | Size | Contents |
|---|---|---|---|
| `knee_multicoil_val_subset.tar` | `make subset-val` | ~20 GB | 20 of the 199 val volumes |
| `knee_multicoil_train_subset.tar.xz` | `make subset-train` | ~65 GB | ~120 of the 973 train volumes, streamed as a prefix of NYU's `train_batch_0` |

`train.sub` defaults to these. Every Phase A and Phase B model must train and
validate on the same two subsets — the `*_subset_files.txt` lists those jobs
emit are the definition of that set. **Absolute numbers from these runs are
not comparable to the paper**; see "Known deviations" in
`training/README.md`.

Building the val subset deletes the full val tarball from `/staging`, which
makes a full-split Tier 1 run impossible without re-downloading it (the NYU
presigned URLs are valid to roughly 2026-12-08). Run Tier 1 first if that
number matters.

## What runs where

| Machine | Purpose |
|---|---|
| Laptop (Apple Silicon) | Build the `linux/amd64` Docker images, push to Docker Hub, smoke-test on synthetic data |
| `ap2001.chtc.wisc.edu` (access point) | Holds `verification/` and `training/`, submits jobs, receives results |
| `transfer.chtc.wisc.edu` | Downloads the NYU tarballs into `/staging/a/apryan3/fastmri/` |

Both Docker images must be built for `linux/amd64`; the Dockerfiles pin it
and the Makefiles pass `--platform`. A bare arm64 build fails in the conda
layer with `hdf5 1.10.6 nompi_h6a2412b_1114 does not exist`.

`training/train.sub` already points at
`docker://genjigod/fastmri-train:2026-09`. `verification/verify.sub` still
says `CHANGE_ME` — edit it before the first verification submit.

## Secrets

The Weights & Biases key and the NYU presigned URLs live in an untracked
`.env` in each stage directory (`cp .env.example .env; chmod 600 .env`) or
are exported in the submitting shell. They reach the job through HTCondor
`getenv`, and are never written to a `.sub` file or anything committed.
Rotate the key that is in the old `Research/inpainting.sub` history before
using any key here.
