# verification/ — scoring an E2E VarNet checkpoint on CHTC

Runbook for evaluating a checkpoint on `multicoil_val`. It is the same
setup as `../training/` on purpose: the same Docker recipe, the same
`submit.sh model1|model2` presets (`make verify-model1` here is `make submit-model1` there), a `train.sub`-shaped submit file with the
same guarded macros, the same `run_*.sh` job executable with the same
eviction handling, the same `logs/` and `runs/<model>/<Cluster>/` layout,
the same resources, the same shared `../.env`. The one difference is what
runs inside the job: `verify_varnet.py` **loads a checkpoint and scores it**
instead of training one from scratch. If you know `training/README.md`,
sections 4 and 6 here are the new material.

Nothing here modifies `fastMRI/`, `parameter_interpolation/`, or any file
outside this directory. It imports the `fastmri` package as installed in the
image and reads its data files, that is all.

*Status 2026-09-14: green locally (`make local-run`, synthetic data). On CHTC
a `model1` job has run to completion once; its output transfer was held by
the W&B symlink bug that `run_verify.sh` now fixes (section 5). No tier has
produced a real number yet.*

## The presets

| Preset | `accelerations` | `center_fractions` | `mask_type` | Data | What the job does |
|---|---|---|---|---|---|
| **model1** | `4` | `0.08` | `equispaced_fraction` | val subset (20 vol) | `make verify-model1`: score the checkpoint at 4x. Pairs with `training: make submit-model1`. |
| **model2** | `2 4 6 8` | `0.16 0.08 0.0533 0.04` | `equispaced_fraction` | val subset (20 vol) | `make verify-model2`: one full pass per rate, **plus** a mixed pass that draws one rate per volume the way training validates. Pairs with `training: make submit-model2`. |
| **tier1** | `4 8` | `0.08 0.04` | `random` | full val (199 vol) | The `VERIFICATION.md` Claim A sweep: the fastMRI knee convention the third-party reference values are keyed on. PASS / INVESTIGATE / FAIL verdicts. |
| **tier0** | | | | none | Environment checks + the fastMRI test suite on a GPU node. Once per image tag. |

model1 / model2 are training's lists, paired elementwise by
`fastmri.data.subsample.MaskFunc`, under training's mask family and on the
same 20-volume val subset training validates against.

### Per-rate and mixed: model 2 reports both

A multi-rate job scores the checkpoint two different ways, because they
answer two different questions:

- **Per-rate** (`R2/`, `R4/`, `R6/`, `R8/`): every volume forced to the same
  rate, one full pass each. This says how the model does at each individual
  acceleration, it is what the cross-rate invariant checks (SSIM must fall as
  the rate rises), and it is the only form the reference values can be
  compared against. Four passes for model 2.
- **Mixed** (`mixed/`): one extra pass in which each volume gets one rate,
  drawn the way training draws it. This is the number to put next to a
  training run's `val_metrics/ssim`.

The mixed pass exists because training and a per-rate pass are not measuring
the same thing, so their SSIMs cannot agree. Training's `val_transform` is
`VarNetDataTransform(mask_func=mask)` over the **full** rate lists with
`use_seed=True`, so `MaskFunc.choose_acceleration()` is seeded from the
filename (`seed = tuple(map(ord, fname))`) and every validation volume is
locked to one rate for the whole run. What Lightning logs each epoch is the
mean over volumes of that mixture. The mixed pass rebuilds exactly that mask
function, so `mixed/ssim_mean` is directly comparable; the per-rate numbers
bracket it.

The metric definitions already agree, which is worth knowing because it means
the mixing was the entire discrepancy: `MriModule` averages per-slice SSIM at
`data_range=attrs["max"]` over volumes, and `fastmri.evaluate.ssim` (what this
harness uses) averages per-slice SSIM at `data_range=target.max()` — the same
number whenever `attrs["max"]` is the volume's own max, which is how the
fastMRI knee files are written.

The mixed pass is a genuine extra pass, not a regrouping of the per-rate
results. `choose_acceleration()` spends a different amount of the seeded
random state on `randint(1)` than on `randint(4)`, so a volume that drew 4x in
the mixture gets a different mask from the same volume in the `R4` pass
(measured: 17 of 40 filenames differ). It costs one more pass over the data,
about 25% on top of model 2's four; `--no_mixed_pass` in `extra_args` skips
it. A single-rate job (model1, tier0) skips it automatically, since there it
would just repeat the one per-rate pass.

**Which checkpoint.** By default the released fastMRI knee model
(`knee_leaderboard_state_dict.pt`, 12 cascades) staged next to the data:
that is "pretrained instead of trained from scratch", and it is the only
thing with third-party reference numbers to be judged against. Pass
`ckpt=../training/runs/model1/<Cluster>/checkpoints/last.ckpt` to score a
checkpoint you trained instead (section 4.4). The harness reads the
architecture (cascades, channels, pools) out of the file, so the 8-cascade
training checkpoints and the 12-cascade released one load through the same
path.

## Files

| File | Runs where | Purpose | Training counterpart |
|---|---|---|---|
| `verify_varnet.py` | inside the job (or any `fastmri` env) | The harness. `tier0` (environment checks) and `tier1` (score a checkpoint). | `train_wandb.py` |
| `run_verify.sh` | inside the job | Job executable: extracts the tarball, finds the split, finds the checkpoint, runs the harness, survives a vacate, collects `output/`. | `run_train.sh` |
| `verify.sub` | access point | HTCondor submit file, container universe, every default guarded with `if ! defined`. Model 1 defaults; the other presets via macros. | `train.sub` |
| `verify_tier0.sub` | access point | Tier 0 on a GPU node, no data. | |
| `submit.sh` | access point | `./submit.sh model1\|model2\|tier1\|tier0 [name=value ...]`. Sources `../.env`, refuses to submit without a W&B key unless `OFFLINE=1`, proves with `-dry-run` that the preset reached the job ad. | `submit.sh` |
| `Makefile` | laptop + access point | `make build/push/tier0/smoke/job-smoke/job-evict` (Docker) and `make verify-*/status/logs/why` (condor). | `Makefile` |
| `Dockerfile` | laptop (build), CHTC (run) | Same pins as training's: Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, fastMRI at `91f2df4`, plus pytest for Tier 0. `linux/amd64`. | `Dockerfile` |
| `make_synthetic_val.py` | laptop | Tiny fake `multicoil_val` for the smoke tests. Training's smoke tests use it too. | |
| `prepare_staging.sh` | CHTC transfer node | Downloads `knee_multicoil_val.tar.xz` and the released checkpoint into `/staging/a/apryan3/fastmri/`, verifies SHA256. | |
| `../.env.example` | repo root | Template for the shared `../.env` (gitignored): `WANDB_API_KEY`, `WANDB_ENTITY`, the NYU URLs, `NETID`. One file for both stages. | same file |

The two images are kept separate (`fastmri-verify`, `fastmri-train`) so a
training rebuild can never change what a verification result was produced
with; their pins are identical.

## Where things run, and the one ordering constraint

Same machines as training (`training/README.md`, "Where things run").
Personal staging is `/staging/a/apryan3` (sharded by the netid's first
letter), 100 GB quota. The full val tarball is 93.8 GiB / 100.7 GB decimal,
so it fits only under binary counting with nothing to spare, and the
project's answer (decision 2026-09-11, `ROADMAP.md` Phase 2) is **not** a
quota increase but `training/README.md` section 3b: repack both splits into
subsets that fit together. `make subset-val` there builds
`knee_multicoil_val_subset.tar` (20 volumes, ~20 GB) and then the full val
tarball is deleted by hand.

**Run `tier1` before the repack.** It is the only preset that needs all 199
volumes, and once the full tarball is gone a full-split number means
re-downloading 94 GB from the NYU URL (valid to roughly 2026-12-08). Claim A
does not strictly need it: `./submit.sh tier1 val_data=file:///staging/a/apryan3/fastmri/knee_multicoil_val_subset.tar request_disk=60GB`
still checks the metric code against someone else's measurement of the same
weights, just on 20 volumes. What you lose is the ability to quote a
full-split number later. `model1` / `model2` default to the subset and are
unaffected.

## 0. Before anything

- **Rotate the W&B key** that is in the old `Research/inpainting.sub` git
  history (ROADMAP.md security note). Put the new one in `../.env` (from
  `../.env.example`, `chmod 600`); `submit.sh` and the `make data` /
  `make tier1` targets source it. It is never written to a `.sub` file.
- `verify.sub` expects `/staging/a/apryan3/fastmri/knee_multicoil_val_subset.tar`
  (or the full `.tar.xz` for `tier1`) and `knee_leaderboard_state_dict.pt`
  (section 3a).
- Docker Desktop running on the laptop, logged in to Docker Hub.

## 1. Laptop: build and push the image (once per tag)

```bash
cd verification
make build push IMAGE=genjigod/fastmri-verify:2026-09-18
```

**Rebuild required as of 2026-09-18.** The `2026-09` image pins
`wandb==0.18.7`, which rejects W&B's current 86-character API keys before
contacting the server ("API key must be 40 characters long"); the key in
`../.env` is one of those and verifies fine with wandb 0.26.1. The
Dockerfile now pins `wandb==0.26.1` and `verify.sub` / `verify_tier0.sub`
name the `2026-09-18` tag, so build and push before the next submission.
Same change, same reason, as `training/` (which additionally holds a job
whose W&B preflight fails).

`linux/amd64` for the same reasons as training (CHTC is x86_64, the cu118
wheels and the pinned `hdf5` build only exist there); on an Apple Silicon
laptop every local run goes through emulation, which is fine for synthetic
data. Use a dated tag, never `:latest`. Then set the same line in both
submit files and keep it that way in git:

```
  image = docker://<dockerhub_user>/fastmri-verify:2026-09
```

## 2. Laptop: smoke-test with no real data (minutes each)

```bash
make local-run    # = clean, build, tier0, smoke, job-smoke, job-evict
```

or one at a time:

```bash
make tier0        # environment + metric invariants + the fastMRI test suite; REAL verdicts, must pass
make smoke        # verify_varnet.py on synthetic phantoms, model 2 rate list, random weights
make job-smoke    # run_verify.sh exactly as HTCondor runs it, model 1 rate list, offline W&B
make job-evict    # SIGTERM a running job: the cleanup must still run
```

`tier0` is the one whose verdicts count: a failure means the image is wrong
and nothing produced from it can be trusted. `smoke` and `job-smoke` run a
2-cascade random model on fake data, so their reference-comparison and
`beats_zero_filled` lines are *expected* to say FAIL; what they prove is
that parsing, masking, model I/O, the metric code, the CSV/JSON writers,
tarball extraction, split and checkpoint discovery, and the `output/`
contract all work. `job-smoke` runs W&B offline rather than disabled and
then asserts that no symlink is left under `output/` (section 5).
`job-evict` sends the SIGTERM HTCondor sends on a vacate and checks the
script stops and still runs that cleanup.

## 3. Copy to the access point

Only this directory is needed on CHTC; neither submodule is. `git pull` a
clone there, or scp:

| Copy | Why |
|---|---|
| `verify.sub`, `verify_tier0.sub` | submit files, with your `image =` line edited |
| `submit.sh` | wraps `condor_submit`, applies the presets, runs the preflight |
| `run_verify.sh` | the job executable |
| `verify_varnet.py` | the harness (listed in `transfer_input_files`) |
| `Makefile` | for `make verify-model1` etc. Optional; `./submit.sh` works alone. |
| `prepare_staging.sh` | only if the data still has to be staged (section 3a) |
| `../.env.example` | to create `../.env` by hand on the access point |

Not `synthetic/`, `jobtest/`, `evicttest/`, `output/`, `*.tar`, `.make/`,
and not the laptop's `../.env`.

```bash
# laptop, from the repo root
scp verification/verify.sub verification/verify_tier0.sub verification/submit.sh \
    verification/run_verify.sh verification/verify_varnet.py verification/Makefile \
    apryan3@ap2001.chtc.wisc.edu:~/Fall26Research/verification/
scp .env.example apryan3@ap2001.chtc.wisc.edu:~/Fall26Research/
```

Then, one-time checks there:

```bash
cd ~/Fall26Research/verification
chmod +x run_verify.sh submit.sh
grep '^  image' verify.sub verify_tier0.sub    # must not say CHANGE_ME
ls -la /staging/a/apryan3/fastmri/             # what is actually staged
condor_submit -dry-run /dev/stdout verify.sub model=model2 accelerations="2 4 6 8" \
    center_fractions="0.16 0.08 0.0533 0.04" \
  | grep -iE "^(Arguments|TransferInput|RequestCpus|RequestMemory|RequestDisk|Requirements) "
```

The last line is what `submit.sh` does automatically before every submit:
it proves the preset reached the job ad. `condor_submit` parses a
command-line `name=value` as if it sat at the top of the file, so an
unguarded assignment in the `.sub` would silently overwrite it; every
default in `verify.sub` is therefore wrapped in `if ! defined`, and
`request_cpus` / `request_memory` (pre-seeded by condor_submit, so the
guard never fires) are overridden as `cpus=` / `mem=`. See `train.sub` for
the three-hour incident behind this.

### 3a. Stage the data (transfer node)

```bash
ssh apryan3@transfer.chtc.wisc.edu           # bulk data host, not the access point
set -a; . ~/Fall26Research/.env; set +a      # FASTMRI_VAL_URL, FASTMRI_SHA_URL from the NYU email
PARALLEL=4 nohup ./prepare_staging.sh > prepare.log 2>&1 &
```

It lands in `/staging/a/apryan3/fastmri/`, which is what the `staging`
macro already points at, along with the released checkpoint, and verifies
the SHA256. It is idempotent and resumable. Check `get_quotas
/staging/a/apryan3` first; the tarball is within a rounding error of the
whole quota. Do not scp a laptop copy up instead: the per-flow shaping that
makes `PARALLEL` necessary applies outbound too.

### 3b. The val subset

`training/README.md` section 3b, step 1 (`cd ../training && make
subset-val`) builds `knee_multicoil_val_subset.tar` from the staged full
tarball and, after the swap, it is the file `verify.sub` defaults to.
`run_verify.sh` accepts `.tar` and `.tar.xz` and finds `multicoil_val` at
any depth inside. CHTC's rule: `osdf:///chtc/staging/<path>` for 1-30 GB
inputs, `file:///staging/<path>` from 30 GB up; `file://` works for both.
Do not stage the synthetic phantoms as a "subset"; they only exist for
`make smoke`.

## 4. Run (access point)

Every submission starts the same way:

```bash
ssh apryan3@ap2001.chtc.wisc.edu
cd ~/Fall26Research/verification            # ../.env holds WANDB_API_KEY; or export it
```

### 4.1 Tier 0 once per image tag (minutes)

```bash
make verify-tier0                 # ./submit.sh tier0
```

`runs/tier0/<Cluster>/tier0_report.json` must say `"overall": "pass"`.

### 4.2 Short test job first (model 1, 5 volumes)

```bash
make verify-model1 ARGS='volume_limit=5'
make logs                          # tail -f the newest logs/*.out
```

`submit.sh` first prints `job args: tier1 --run_name verify-model1
--accelerations 4 ...` and the four resolved `Request*` values (8 CPUs,
48 GB, 60 GB, 1 GPU), then submits. The `.out` must show
`[run_verify] multicoil_val: 20 volumes`,
`[run_verify] checkpoint: knee_leaderboard_state_dict.pt`, `loaded ...
(29.9M params)` and per-volume progress lines with a running SSIM. When it
finishes, `runs/model1/<Cluster>/tier1_report.json` must exist and
`condor_q` must not show the job held (section 5). That `<Cluster>`
directory is a throwaway.

### 4.3 The presets

```bash
make verify-model1                 # accelerations="4"       center_fractions="0.08"                  mask_type=equispaced_fraction
make verify-model2                 # accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04" mask_type=equispaced_fraction
make verify-tier1                  # accelerations="4 8"     center_fractions="0.08 0.04"             mask_type=random, full val, request_disk=320GB
```

Run names and W&B runs: `verify-model1`, `verify-model2`, `verify-tier1`.
Output lands in `runs/<preset>/<Cluster>/`. All three score the released
checkpoint unless told otherwise.

### 4.4 Scoring a checkpoint you trained

The point of matching the training setup. When `../training/` has produced
`runs/model1/<Cluster>/checkpoints/last.ckpt`, score it under the same rate
list and val subset it was trained against:

```bash
make verify MODEL=model1 ARGS='ckpt=../training/runs/model1/<Cluster>/checkpoints/last.ckpt run_name=verify-model1-trained'
make verify MODEL=model2 ARGS='ckpt=../training/runs/model2/<Cluster>/checkpoints/last.ckpt run_name=verify-model2-trained'
```

`submit.sh` checks the file exists, HTCondor transfers it into the sandbox,
`run_verify.sh` hands whatever `*.pt` / `*.ckpt` it finds to the harness,
and the harness prints `architecture read from last.ckpt: {'num_cascades':
8, ...}` before loading. A trained checkpoint has no reference values, so
its rates are "recorded only"; the invariants (determinism, beats
zero-filled, volume count, SSIM monotone in R) still run. `run_name` is
given explicitly so it does not continue the released-checkpoint W&B run.

### 4.5 Overrides

`make verify-model1 ARGS='...'` and `./submit.sh model1 name=value ...` are
the same thing. Any `name=value` is a `condor_submit` macro override, so
every knob in `verify.sub` (`val_data`, `ckpt`, `request_disk`,
`volume_limit`, `run_name`, `mask_type`, `extra_args`, `cpus`, `mem`,
`gpu_job_length`, `bad_nodes`, ...) can be set per submission without
editing the file. `extra_args` is appended verbatim to `verify_varnet.py`
(`--num_workers 0`, `--determinism_volumes 0`, `--image_volumes 5`,
`--cpu`, ...). The preflight only asserts on knobs you did not override.

Resources are the training ones (8 CPUs, 48 GB, a 24 GB GPU of capability
7.0 to 9.0) so both jobs land on the same class of slot; `request_disk`
defaults to 60 GB for the plain-tar subset and the `tier1` preset passes
320 GB for the `.xz` full split (93.8 GB + ~192 GB extracted coexist).
Inference needs less than training: `mem=32GB` widens the pool of matching
slots if the queue is slow.

## 5. Monitor

```bash
make status                       # condor_q -nobatch
make logs                         # tail -f the newest logs/*.out
make why JOB=<Cluster>            # hold reason + log tails
```

`stream_output = True` means the `.out` updates live: extraction, the volume
count, then `R4 10/20 volumes, running SSIM 0.9xxx, 8.1s/volume` lines.

W&B project `fastmri-varnet-verify`. As in training, the run id is the run
name with `resume="allow"`, so a resubmitted job continues the same W&B run;
pass a new `run_name` for a genuinely new one. Per-rate aggregates, the
per-volume table, example target / reconstruction / zero-filled / error
images, the reference comparison and every invariant land there. Without a
key the run is written offline to `runs/<model>/<Cluster>/wandb/` and
`wandb sync <that dir>/offline-run-*` uploads it later.

**The symlink hold.** A job that finishes and then goes on hold with
`Transfer output files failure ... Transfer of symlinks to directories is
not supported` (job 10461449 on 2026-09-14) means a `wandb/latest-run`
symlink was left in `output/`. `run_verify.sh` now deletes every symlink
under `output/` before exit, on eviction too, and `make job-smoke` /
`make job-evict` check that it does. A job held this way cannot be
released usefully; resubmit. Hold codes 6 and 13 (a broken execute node, a
stalled `/staging` read) are released automatically up to five times by
`periodic_release`, the same as `train.sub`.

## 6. What comes back, and the verdicts

`runs/<model>/<Cluster>/` (the job's `output/`):

- `tier1_report.json`: every check, its verdict, the config, the
  architecture that was loaded
- `per_volume_R<N>.csv`: SSIM / PSNR / NMSE / MSE per volume, model and
  zero-filled, one file per rate
- `per_volume_mixed.csv`: the same columns for the mixed pass, with each
  volume's `acceleration` / `center_fraction` column showing the rate its
  filename actually drew
- `results.csv`: one row per (run, rate) in the `VERIFICATION.md` Section 8
  schema, append-only
- `wandb/`: the offline W&B run, only when no key was available

In the report JSON, `per_rate` holds the four per-rate blocks and `mixed`
holds the mixture, including `rate_counts` (how many volumes drew each rate)
and `within_mixture` (how the volumes that drew each rate scored on their
own — disjoint subsets of the one mixed pass, not the per-rate passes). In
`results.csv` the mixed row carries `acceleration = mixed`. In W&B the keys
are `mixed/ssim_mean`, `mixed/psnr_mean`, `mixed/nmse_mean` alongside the
`R<N>/...` ones, and the extra invariants are `mixed_beats_zero_filled`,
`mixed_volume_count` and `mixed_within_per_rate_span` (a sanity check that
the mixture lands inside the span of the per-rate means).

`tier1` compares each rate against the third-party measurements of the
released checkpoint listed in `VERIFICATION.md` Section 2 and prints
`PASS` / `INVESTIGATE` / `FAIL` per source using the Section 4.4 thresholds.
It also runs the Section 4.5 invariants: determinism (re-runs the first
volumes and diffs the output), model beats zero-filled by a margin, volume
count matches, SSIM monotone in the acceleration rate. Exit code 0 means
every check is pass or investigate, 1 means at least one fail.

The reference values are for the fastMRI knee convention (`random` masks,
0.08 / 0.04 at 4x / 8x): the `tier1` preset. Under `equispaced_fraction`
(model1 / model2) the harness logs `no reference values for
(equispaced_fraction, R); recorded only` and emits metrics without a
pass/fail. Those runs measure; `tier1` is what verifies. Record the volume
count with every number; the report and `results.csv` carry it.

### Pretrained vs. from-scratch (the 2x2)

| | 4x only | 2x / 4x / 6x / 8x |
|---|---|---|
| **Released weights** | `verification: make verify-model1` | `verification: make verify-model2` |
| **Trained here** | `training: make submit-model1`, then `verification: make verify MODEL=model1 ARGS='ckpt=...'` | `training: make submit-model2`, then `verification: make verify MODEL=model2 ARGS='ckpt=...'` |

What is held identical across the row: the acceleration / centre-fraction
lists, the mask family, the 20-volume val subset, seed 42, and the scorer.

**Comparing a verification number with the training run it came from.** Use
`mixed/ssim_mean` in `fastmri-varnet-verify`, not `R4/ssim_mean`, against
`val_metrics/ssim` in `fastmri-varnet-train`. Those two are the same
measurement. The remaining difference is the data: training validates on
whatever `val_data` its job was given, so the two only line up when the
verification job scored the same subset, which the `model1` / `model2`
presets do by default.

**The released-weights row is a ceiling reference, not a matched control.**
Three things differ from the trained models, all of them in its favour:

- **It saw the val split.** The leaderboard checkpoint was trained on
  `train`+`val` combined, so the 20 subset volumes are training data for it
  and useless as a generalisation estimate. model1 / model2 never see them.
- **It is a bigger model**: 12 cascades (~29.9M parameters) against the demo
  default 8 that `training/train_wandb.py` uses (~20.1M).
- **It trained on all 973 train volumes**, against the ~120-volume subset.

So read it as "how far from a fully-trained reference are we", never as
"pretrained beats from-scratch". `VERIFICATION.md` Claim A is the same
point. Two smaller caveats: 2x and 6x are not in the paper (their centre
fractions 0.16 and 0.0533 come from the 0.32/R convention extended by us),
and under `equispaced_fraction` there is no reference verdict.

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
cp ../.env.example ../.env         # paste the NYU presigned URLs and, optionally, the W&B key
make data                          # download the val set + checkpoint into DATA_DIR, verify SHA256
make extract                       # unpack (~2 h)
make tier1 MODEL=tier1             # the scored run under the knee convention; MODEL=model1|model2 for training's lists
make report
```

`DATA_DIR` defaults to `~/fastmri-data`, outside the repo and outside
OneDrive. `PARALLEL` (default 64) is the download's connection count, tuned
for a per-flow-shaped dorm ISP; on a normal network `PARALLEL=1`.

## W&B key handling

The key is read from `../.env` (or the shell) by `submit.sh` and copied
into the job by HTCondor's `getenv`. It is never written into a `.sub`
file, a script, or this directory. It is visible in the job ClassAd
(`condor_q -l`) to you and CHTC admins, which is the standard CHTC pattern.

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
