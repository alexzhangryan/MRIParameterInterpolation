# verification/ — scoring an E2E VarNet checkpoint on CHTC

Runbook for evaluating a checkpoint on `multicoil_val`. It is the same
setup as `../training/` on purpose: the same Docker recipe, the same
`submit.sh model1|model2` presets, the same `train.sub`-shaped submit file,
the same `run_*.sh` job executable, the same `logs/` and
`runs/<model>/<Cluster>/` layout, the same resources. The one difference is
what runs inside the job: `verify_varnet.py` **loads a checkpoint and scores
it** instead of training one from scratch. If you know `training/README.md`,
only sections 4 and 6 here are new.

Nothing here modifies `fastMRI/`, `parameter_interpolation/`, or any file
outside this directory. It imports the `fastmri` package as installed in the
image and reads its data files, that is all.

## The two models, evaluated

| Model | `accelerations` | `center_fractions` | What the job does |
|---|---|---|---|
| **model1** | `4` | `0.08` | Score the checkpoint at 4x over the whole val set. |
| **model2** | `2 4 6 8` | `0.16 0.08 0.0533 0.04` | Score the checkpoint at each of the four rates, one full pass over the val set per rate. |

Same preset lists as training, paired elementwise by
`fastmri.data.subsample.MaskFunc`, same `equispaced_fraction` masks. Where
training draws one (acceleration, centre fraction) pair per sample,
evaluation scores every pair separately, so a model 2 job reports four
per-rate results and the cross-rate invariant (SSIM must fall as the rate
rises).

**Which checkpoint.** By default the released fastMRI knee model
(`knee_leaderboard_state_dict.pt`, 12 cascades) staged next to the data:
that is "pretrained instead of trained from scratch", and it is the only
thing that has third-party reference numbers to be judged against. Pass
`ckpt=runs/model1/<Cluster>/checkpoints/last.ckpt` to score a checkpoint you
trained instead (section 4.4). The harness reads the architecture
(cascades, channels, pools) out of the file, so the 8-cascade training
checkpoints and the 12-cascade released one load through the same path.

## Files

| File | Runs where | Purpose | Training counterpart |
|---|---|---|---|
| `verify_varnet.py` | inside the job (or any `fastmri` env) | The harness. `tier0` (environment checks) and `tier1` (score a checkpoint). | `train_wandb.py` |
| `run_verify.sh` | inside the job | Job executable: extracts the tarball, finds the split, finds the checkpoint, runs the harness, collects `output/`. | `run_train.sh` |
| `verify.sub` | access point | HTCondor submit file, container universe. Model 1 defaults; model 2 via macros. | `train.sub` |
| `verify_tier0.sub` | access point | Tier 0 on a GPU node, no data. Once per image tag. | |
| `submit.sh` | access point | `./submit.sh model1\|model2\|tier0 [name=value ...]`. Refuses to submit without a W&B key unless `OFFLINE=1`. | `submit.sh` |
| `Makefile` | laptop + access point | `make build/push/tier0/smoke/job-smoke` (Docker) and `make submit-tier0/submit-model1/submit-model2/status/logs` (condor). | `Makefile` |
| `Dockerfile` | laptop (build), CHTC (run) | Same pins as training's: Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, fastMRI at `91f2df4`, plus pytest for Tier 0. `linux/amd64`. | `Dockerfile` |
| `make_synthetic_val.py` | laptop | Tiny fake `multicoil_val` for the smoke tests. Training's smoke tests use it too. | |
| `prepare_staging.sh` | CHTC transfer node | Downloads `knee_multicoil_val.tar.xz` and the released checkpoint into `/staging/a/apryan3/fastmri/`, verifies SHA256. | |

The two images are kept separate (`fastmri-verify`, `fastmri-train`) so a
training rebuild can never change what a verification result was produced
with; their pins are identical.

## Where things run

Same machines as training (`training/README.md`, "Where things run"):
laptop builds and pushes the image and runs the smoke tests; the access
point `ap2001.chtc.wisc.edu` holds this directory, submits, and receives
`runs/`; `transfer.chtc.wisc.edu` is for staging the data; the execute node
is never touched directly. Personal staging is `/staging/a/apryan3`
(sharded by the netid's first letter), 100 GB quota, and the val tarball is
93.8 GiB / 100.7 GB decimal, so it fits only under binary counting with
about 6 GiB to spare for the checkpoint and nothing else.

## 0. Before anything

- **Rotate the W&B key** that is in the old `Research/inpainting.sub` git
  history (ROADMAP.md security note). Every `WANDB_API_KEY` below means the
  new key. It is only ever exported in a shell, never written to a file in
  this repo.
- `verify.sub` expects `/staging/a/apryan3/fastmri/knee_multicoil_val.tar.xz`
  and `knee_leaderboard_state_dict.pt` (section 3a). Until the full tarball
  is staged, a **subset** tarball (section 3b) is the runnable configuration.
- Docker Desktop running on the laptop, logged in to Docker Hub.

## 1. Laptop: build and push the image (once per tag)

```bash
cd verification
make build push IMAGE=<dockerhub_user>/fastmri-verify:2026-09
```

`linux/amd64` for the same reasons as training (CHTC is x86_64, the cu118
wheels and the pinned `hdf5` build only exist there); on an Apple Silicon
laptop every local run goes through emulation, which is fine for synthetic
data. Use a dated tag, never `:latest`. Then set the same line in both
submit files and keep it that way in git:

```
image      = docker://<dockerhub_user>/fastmri-verify:2026-09
```

## 2. Laptop: smoke-test with no real data (minutes each)

```bash
make tier0        # environment + metric invariants + the fastMRI test suite; REAL verdicts, must pass
make smoke        # verify_varnet.py on synthetic phantoms, model 2 rate list, random weights
make job-smoke    # run_verify.sh exactly as HTCondor runs it, model 1 rate list, offline W&B
```

`tier0` is the one whose verdicts count: a failure means the image is wrong
and nothing produced from it can be trusted. `smoke` and `job-smoke` run a
2-cascade random model on fake data, so their reference-comparison and
`beats_zero_filled` lines are *expected* to say FAIL; what they prove is
that parsing, masking, model I/O, the metric code, the CSV/JSON writers,
tarball extraction, split and checkpoint discovery, and the `output/`
contract all work. `job-smoke` runs W&B offline rather than disabled and
then asserts that no symlink is left under `output/`: W&B writes
`wandb/latest-run` as a symlink to a directory, HTCondor refuses to transfer
those and holds the job, and `run_verify.sh` removes them before exit.

Green here means the image, the harness and the job executable all work
before a single GPU slot is used.

## 3. Copy to the access point

Only this directory is needed on CHTC; neither submodule is.

| Copy | Why |
|---|---|
| `verify.sub`, `verify_tier0.sub` | submit files, with your `image =` line edited |
| `submit.sh` | wraps `condor_submit`, applies the model presets |
| `run_verify.sh` | the job executable |
| `verify_varnet.py` | the harness (listed in `transfer_input_files`) |
| `Makefile` | for `make submit-model1` etc. Optional; `./submit.sh` works alone. |
| `prepare_staging.sh` | only if the data still has to be staged (section 3a) |

Not `synthetic/`, `jobtest/`, `output/`, `*.tar`, `.env`, `.make/`.

```bash
# laptop, from the repo root
scp verification/verify.sub verification/verify_tier0.sub verification/submit.sh \
    verification/run_verify.sh verification/verify_varnet.py verification/Makefile \
    apryan3@ap2001.chtc.wisc.edu:~/Fall26Research/verification/
```

Or `git pull` a clone on the access point. Then, one-time checks there:

```bash
cd ~/Fall26Research/verification
chmod +x run_verify.sh submit.sh
grep '^image' verify.sub verify_tier0.sub    # must not say CHANGE_ME
ls -la /staging/a/apryan3/fastmri/           # what is actually staged
condor_submit -dry-run /dev/stdout verify.sub model=model2 accelerations="2 4 6 8" \
    center_fractions="0.16 0.08 0.0533 0.04" \
  | grep -iE "^(Arguments|TransferInput|RequestDisk|Requirements) "
```

### 3a. Stage the data (transfer node)

```bash
ssh apryan3@transfer.chtc.wisc.edu           # bulk data host, not the access point
export FASTMRI_VAL_URL='...knee_multicoil_val.tar.xz?AWSAccessKeyId=...'   # from the NYU email
export FASTMRI_SHA_URL='...SHA256?AWSAccessKeyId=...'
PARALLEL=4 nohup ./prepare_staging.sh > prepare.log 2>&1 &
```

It lands in `/staging/a/apryan3/fastmri/`, which is what the `staging`
macro already points at, along with the released checkpoint, and verifies
the SHA256. It is idempotent and resumable. Check `get_quotas
/staging/a/apryan3` first; the tarball is within a rounding error of the
whole quota. Do not scp a laptop copy up instead: the per-flow shaping that
makes `PARALLEL` necessary applies outbound too.

### 3b. Subset tarballs (the only thing that runs under the default quota)

A plain `.tar` holding a `multicoil_val/` directory with as many volumes as
fit, put in `/staging/a/apryan3/fastmri/`. A one-off interactive job
(`condor_submit -i`) that extracts the full tarball once and re-tars a
subset is the fastest way to make one. Then:

```bash
./submit.sh model1 \
    val_data=osdf:///chtc/staging/a/apryan3/fastmri/knee_multicoil_val_subset.tar \
    request_disk=60GB
```

CHTC's rule: `osdf:///chtc/staging/<path>` for 1-30 GB inputs,
`file:///staging/<path>` from 30 GB up. `run_verify.sh` accepts `.tar` and
`.tar.xz` and finds `multicoil_val` at any depth inside. The same subset
tarball serves `training/`'s `val_data=`. Do not stage the synthetic
phantoms as a "subset"; they only exist for `make smoke`.

## 4. Run (access point)

Every submission starts the same way:

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research/verification
export WANDB_API_KEY=...          # freshly rotated; never written to a file
export WANDB_ENTITY=...           # optional
```

### 4.1 Tier 0 once per image tag (minutes)

```bash
make submit-tier0                 # ./submit.sh tier0
```

Runs the environment checks and the fastMRI test suite inside the container
on a GPU node. `runs/tier0/<Cluster>/tier0_report.json` must say
`"overall": "pass"`.

### 4.2 Short test job first (model 1, 5 volumes)

```bash
make submit-model1 ARGS='volume_limit=5'
make logs                          # tail -f the newest logs/*.out
```

The `.out` must show `[run_verify] multicoil_val: N volumes`,
`[run_verify] checkpoint: knee_leaderboard_state_dict.pt`, `loaded ...
(29.9M params)` and a per-volume progress line with a running SSIM. When it
finishes, `runs/model1/<Cluster>/tier1_report.json` must exist and
`condor_q` must not show the job held (section 5). That `<Cluster>`
directory is a throwaway.

### 4.3 Model 1 and model 2

```bash
make submit-model1                 # ./submit.sh model1: accelerations="4" center_fractions="0.08"
make submit-model2                 # ./submit.sh model2: accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04"
```

Run names and W&B runs: `verify-model1`, `verify-model2`. Output lands in
`runs/model1/<Cluster>/`, `runs/model2/<Cluster>/`. Both jobs score the
released checkpoint unless told otherwise.

### 4.4 Scoring a checkpoint you trained

The point of matching the training setup. When `../training/` has produced
`runs/model1/<Cluster>/checkpoints/last.ckpt`, score it under the same rate
list it was trained on:

```bash
make submit MODEL=model1 ARGS='ckpt=../training/runs/model1/<Cluster>/checkpoints/last.ckpt run_name=verify-model1-trained'
make submit MODEL=model2 ARGS='ckpt=../training/runs/model2/<Cluster>/checkpoints/last.ckpt run_name=verify-model2-trained'
```

`submit.sh` checks the file exists, HTCondor transfers it into the sandbox,
`run_verify.sh` hands whatever `*.pt` / `*.ckpt` it finds to the harness,
and the harness prints `architecture read from last.ckpt: {'num_cascades':
8, ...}` before loading. A trained checkpoint has no reference values, so
its rates are "recorded only"; the invariants (determinism, beats
zero-filled, volume count, SSIM monotone in R) still run. `run_name` is
given explicitly so it does not continue the released-checkpoint W&B run.

### 4.5 Overrides

`make submit-model1 ARGS='...'` and `./submit.sh model1 name=value ...` are
the same thing. Any `name=value` is a `condor_submit` macro override, so
every knob in `verify.sub` (`val_data`, `ckpt`, `request_disk`,
`volume_limit`, `run_name`, `mask_type`, `extra_args`, ...) can be set per
submission without editing the file. `extra_args` is appended verbatim to
`verify_varnet.py` (`--num_workers 0`, `--determinism_volumes 0`,
`--image_volumes 5`, `--cpu`, ...).

`mask_type=random accelerations="4 8" center_fractions="0.08 0.04"` is the
fastMRI knee convention and the only configuration the third-party reference
values in `verify_varnet.py` apply to (section 6); `equispaced_fraction`,
the training default, is recorded but not judged.

Resources are the training ones (8 CPUs, 48 GB, 350 GB scratch, a 24 GB
GPU) so both jobs land on the same class of slot. Inference needs less;
`request_memory=32GB gpus_minimum_memory=16000M` widens the pool of matching
slots if the queue is slow, and `request_disk` should be sized from the
tarball actually staged (peak scratch is compressed plus extracted at once:
93.8 GB + ~192 GB for the full val set, ~3x the tarball for a plain `.tar`
subset).

## 5. Monitor

```bash
make status                       # condor_q -nobatch
make logs                         # tail -f the newest logs/*.out
condor_q -l <Cluster> | grep -i holdreason
```

`stream_output = True` means the `.out` updates live: extraction, the volume
count, then `R4 10/199 volumes, running SSIM 0.9xxx, 8.1s/volume` lines.

W&B project `fastmri-varnet-verify` (override with `WANDB_PROJECT` in the
shell). As in training, the run id is the run name with `resume="allow"`,
so a resubmitted job continues the same W&B run; pass a new `run_name` for
a genuinely new one. Per-rate aggregates, the per-volume table, example
target / reconstruction / zero-filled / error images, the reference
comparison and every invariant land there. Without a key the run is written
offline to `runs/<model>/<Cluster>/wandb/` and
`wandb sync <that dir>/offline-run-*` uploads it later.

A job that finishes but goes on **hold** with `Transfer output files
failure ... Transfer of symlinks to directories is not supported` means a
`wandb/latest-run` symlink was left in `output/`. `run_verify.sh` removes
those before exit (and `make job-smoke` checks that it does), so this
should not recur; if it does, `condor_release` will not help, resubmit.

## 6. What comes back, and the verdicts

`runs/<model>/<Cluster>/` (the job's `output/`):

- `tier1_report.json`: every check, its verdict, the config, the
  architecture that was loaded
- `per_volume_R<N>.csv`: SSIM / PSNR / NMSE / MSE per volume, model and
  zero-filled, one file per rate
- `results.csv`: one row per (run, rate) in the `VERIFICATION.md` Section 8
  schema, append-only
- `wandb/`: the offline W&B run, only when no key was available

`tier1` compares each rate against the third-party measurements of the
released checkpoint listed in `VERIFICATION.md` Section 2 and prints
`PASS` / `INVESTIGATE` / `FAIL` per source using the Section 4.4 thresholds.
It also runs the Section 4.5 invariants: determinism (re-runs the first
volumes and diffs the output), model beats zero-filled by a margin, volume
count matches, SSIM monotone in the acceleration rate. Exit code 0 means
every check is pass or investigate, 1 means at least one fail.

The reference values are for the fastMRI knee convention (`random` masks,
0.08 / 0.04 at 4x / 8x). Other mask types, rates and checkpoints are
recorded but not judged.

## Mask family notes (from the Tier 0 output)

At a 368-wide knee acquisition, `center_fractions` 0.08 / 0.04 give 29 / 15
centre lines against the paper's fixed 30 / 16. That is the documented
centre-line deviation, now with numbers. Two further things Tier 0 makes
visible:

- `random` samples close to $1/R$ overall (0.258 at 4x, 0.128 at 8x).
- `equispaced` samples well above $1/R$ (0.310 at 4x, 0.160 at 8x) because
  it adds the centre lines on top of every $R$-th line without correcting for
  them. That is the paper's own definition of $M_e(r, l)$, so it is the right
  choice for a Table 1 comparison. `equispaced_fraction` is the corrected
  variant, what training uses, and does not correspond to either paper table.

## Laptop: the real data locally (optional)

Evaluation is the one job that can also run on a laptop with a GPU, because
the val split is all it needs. `training/` has no equivalent.

```bash
cp .env.example .env               # paste the NYU presigned URLs and, optionally, the W&B key
make data                          # download the val set + checkpoint into DATA_DIR, verify SHA256
make extract                       # unpack (~2 h)
make tier1 MODEL=model1            # the scored run; MODEL=model2 for the four-rate list
make report
```

`DATA_DIR` defaults to `~/fastmri-data`, outside the repo and outside
OneDrive. `PARALLEL` (default 64) is the download's connection count, tuned
for a per-flow-shaped dorm ISP; on a normal network `PARALLEL=1`.

## W&B key handling

The key is read only from the `WANDB_API_KEY` environment variable of the
submitting shell and copied into the job by HTCondor's `getenv`. It is never
written into a `.sub` file, a script, or this directory. It is visible in the
job ClassAd (`condor_q -l`) to you and CHTC admins, which is the standard
CHTC pattern.

## Running without Docker

The harness needs only the `fastmri` package and its dependencies plus
`wandb`. In a fresh environment:

```bash
pip install "pytorch-lightning==1.9.5" "torchmetrics==0.11.4" "numpy<2" \
            h5py runstats "scikit-image<0.23" pyyaml "pandas<2.1" wandb requests tqdm
pip install --no-deps -e ../fastMRI
```

`torch` must be installed first for your platform. The Lightning 1.x
requirement is the one that cannot be relaxed.

## Not covered here

Training the two models is `../training/`. Tier 2 (Table 2 reproduction)
and Tier 3 (seed variance) from `VERIFICATION.md` Sections 5 and 6 are
training runs and go through `training/`; the per-volume CSV writer here is
the piece Tier 3 needs for paired comparisons, and `volume_metrics()` and
`crop_like_evaluate()` in `verify_varnet.py` can be imported for that. Read
`training/README.md` "Known deviations from Sriram et al. 2020" before
comparing any number to the paper.
