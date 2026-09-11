# verification/

Self-contained harness for `VERIFICATION.md` Tiers 0 and 1. Nothing here
modifies `fastMRI/`, `parameter_interpolation/`, or any file outside this
directory. It imports the `fastmri` package and reads its data files, that is
all.

| File | Runs where | Purpose |
|---|---|---|
| `verify_varnet.py` | anywhere with the `fastmri` env | The harness. `tier0` and `tier1` subcommands. |
| `make_synthetic_val.py` | laptop | Tiny fake `multicoil_val` for smoke-testing the harness and the CHTC job before the real data lands. |
| `Dockerfile` | laptop (build), CHTC (run) | The environment: Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, fastMRI at commit `91f2df4`. |
| `prepare_staging.sh` | CHTC transfer node | Downloads `knee_multicoil_val.tar.xz` and the released checkpoint into `/staging/a/apryan3/fastmri/`, verifies SHA256. |
| `verify.sub`, `verify_tier0.sub` | CHTC access point | HTCondor submit files, container universe. |
| `run_verify.sh` | inside the job | Job executable: extracts data, runs the harness, collects `results/`. |
| `submit.sh` | CHTC access point | Wraps `condor_submit`, refuses to submit without a W&B key unless `OFFLINE=1`. |

## What a run produces

`results/` (transferred back to `runs/<Cluster>/`):

- `tier0_report.json` or `tier1_report.json`: every check, its verdict, and the config
- `per_volume_R<N>.csv`: SSIM / PSNR / NMSE / MSE per volume, model and zero-filled, one file per rate
- `results.csv`: one row per (run, rate) in the `VERIFICATION.md` Section 8 schema, append-only
- `wandb/`: the offline W&B run, only when no key was available

In W&B (project `fastmri-varnet-verify`): per-rate aggregates, the per-volume
table, example target / reconstruction / zero-filled / error images, the
reference-comparison table, every invariant as a summary field, and the JSON
report and CSVs as artifacts.

## Verdicts

`tier1` compares each rate against the third-party measurements of the
released checkpoint listed in `VERIFICATION.md` Section 2 and prints
`PASS` / `INVESTIGATE` / `FAIL` per source using the Section 4.4 thresholds.
It also runs the Section 4.5 invariants: determinism (re-runs the first
volumes and diffs the output), model beats zero-filled by a margin, volume
count matches, SSIM monotone in the acceleration rate. Exit code 0 means every
check is pass or investigate, 1 means at least one fail.

The reference values are for the fastMRI knee convention (`random` masks,
0.08 / 0.04). Other mask types and rates are recorded but not judged.

## Where things live on CHTC

| What | Where | Constraint |
|---|---|---|
| This repo, submit files, logs | `/home/apryan3` on `ap2001.chtc.wisc.edu` | 40 GB quota, code only |
| `knee_multicoil_val.tar.xz` (93.8 GiB), released checkpoint | `/staging/a/apryan3/fastmri/` | 100 GB quota — the tarball alone is 100.7 GB decimal |
| The container image | Docker Hub, pulled by the execute node | never stored on CHTC |

Personal staging is sharded by the first letter of the netid:
`/staging/a/apryan3`, **not** `/staging/apryan3`. The shared group directory
`/staging/groups/kamilov_group/Kamilov-SciAI-datasets` is not readable by this
account as of 2026-09-10. If that changes, the only edits needed are the
`staging` macro in `verify.sub` and `STAGING_DIR` for `prepare_staging.sh`.

A job never reads `/staging` directly. HTCondor transfers the listed inputs
into the job's scratch directory, which is why `verify.sub` asks for
`HasCHTCStaging` slots and requests enough `request_disk` to hold the tarball
and its extracted contents at the same time.

The 100 GB quota on `/staging/a/apryan3` (confirmed 2026-09-10) is the binding
constraint, and it is tighter than it looks. The validation tarball is
100,694,526,932 bytes — 93.8 GiB, or 100.7 GB decimal — so it fits only if
`get_quotas` counts in binary units, and even then leaves about 6 GiB for
everything else. Nothing of consequence can be staged alongside it, and
`multicoil_train` (~931 GB unpacked) is out of reach entirely. A full Tier 1
run and any from-scratch training both need a quota increase or access to the
group directory.

## Step by step

### 1. Build and push the image (laptop, once per tag)

```bash
cd verification
make build push IMAGE=<you>/fastmri-verify:2026-09
# equivalent by hand:
docker build --platform linux/amd64 --build-arg IMAGE_TAG=<you>/fastmri-verify:2026-09 -t <you>/fastmri-verify:2026-09 .
docker push <you>/fastmri-verify:2026-09
```

The image must be `linux/amd64`: CHTC is x86_64, the cu118 torch wheels only
exist for x86_64, and the pinned conda `hdf5` build string is a linux-64
build. This laptop is Apple Silicon, so a bare `docker build` used to produce
an arm64 image and die in the conda layer with
`hdf5 1.10.6 nompi_h6a2412b_1114 does not exist`. The `FROM` line now pins
the platform and `make build` passes `--platform` too, so either route works.
Every local `docker run` of this image on the Mac goes through emulation:
slow, but correct for the synthetic smoke tests.

Use a dated tag, never `:latest`, so a later rebuild cannot change what an
old result was produced with. The build ends with a self-check that prints the
installed versions and asserts Lightning is 1.x.

### 2. Smoke-test the harness locally, no real data (laptop, minutes)

```bash
docker run --rm -it --platform linux/amd64 -v "$PWD":/work -w /work <you>/fastmri-verify:2026-09 bash
python make_synthetic_val.py --out synthetic/multicoil_val --volumes 3 --slices 4
python verify_varnet.py tier0 --fastmri_repo /opt/fastMRI --run_pytest --no_wandb
python verify_varnet.py tier1 --data_path synthetic/multicoil_val --random_init \
    --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2 \
    --accelerations 4 8 --center_fractions 0.08 0.04 --no_wandb --cpu --num_workers 0
```

The second command is expected to print `FAIL` on the reference comparison
and on `beats_zero_filled`: the model is random and the data is fake. What it
proves is that file parsing, masking, model I/O, metric code, CSV / JSON output
and the verdict logic all run. To smoke-test the CHTC job the same way:

```bash
tar -cf knee_multicoil_val_synthetic.tar -C synthetic multicoil_val
# copy to /staging/a/apryan3/fastmri/ then:
./submit.sh data=osdf:///chtc/staging/a/apryan3/fastmri/knee_multicoil_val_synthetic.tar \
            request_disk=20GB extra_args="--random_init --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2 --num_workers 0"
```

### 3. Stage the real data (transfer node, one to three hours)

```bash
ssh apryan3@transfer.chtc.wisc.edu   # not the access point; this host is for bulk data
export FASTMRI_VAL_URL='...knee_multicoil_val.tar.xz?AWSAccessKeyId=...'   # from the NYU email
export FASTMRI_SHA_URL='...SHA256?AWSAccessKeyId=...'
PARALLEL=4 nohup ./prepare_staging.sh > prepare.log 2>&1 &
```

It lands in `/staging/a/apryan3/fastmri/`, which is what `verify.sub`'s
`staging` macro already points at. `PARALLEL` defaults to 16 to beat a dorm
ISP's per-flow shaping; from campus to S3 a few flows are plenty.

Check the quota first (`get_quotas /staging/a/apryan3` on the access point).
It is 100 GB, and the tarball is 100,694,526,932 bytes — 93.8 GiB or 100.7 GB
depending on how that limit is counted, so it either just fits or just does
not. Establish which before committing to a multi-hour transfer. Under a
100 GB ceiling the val set is the one file worth keeping: it is the only split
with ground truth the harness can score.

Do not scp the local copy up from the desktop instead. The shaping that makes
`PARALLEL=16` necessary applies to outbound traffic too, so a 94 GB upload
takes days where the S3-to-campus fetch takes hours.

### 3b. Files already present, or running on your own GPU

Everything is idempotent, so nothing is re-downloaded or re-extracted:

- `prepare_staging.sh` skips any file whose size already matches the server
  and resumes a partial one. Point it at a local directory with
  `STAGING_DIR=/path ./prepare_staging.sh` to use it off-cluster.
- `verify_varnet.py --state_dict <path>` uses the checkpoint if the file
  exists and only downloads (with `--download_state_dict`) if it does not.
- `verify_varnet.py --data_path <dir>` just reads the `.h5` files in place.
  If you already have `multicoil_val/` extracted, point at it and skip the
  tarball entirely.
- `run_verify.sh` skips extraction when `data/multicoil_val/` exists. It
  deletes the tarball after extracting unless `KEEP_TARBALL=1`, so set that
  for a local run or, simpler, extract by hand once
  (`xz -dc -T0 knee_multicoil_val.tar.xz | tar -x`) and call
  `verify_varnet.py` directly.

### 4. Tier 0 on CHTC (minutes)

```bash
export WANDB_API_KEY=...        # freshly rotated, see ROADMAP.md security note
./submit.sh tier0
```

Do this once per image tag. It runs the repo's own test suite inside the
container on a GPU node. If it fails, the image is wrong and nothing produced
from it counts.

### 5. Tier 1 on CHTC

```bash
./submit.sh volume_limit=20            # ~30 min including extraction, sanity
./submit.sh                            # full 199 volumes, 4x and 8x, ~2-4 h
```

Extraction of the 94 GB `.xz` inside the job takes about 2 hours and is the
reason `request_disk` is 320 GB. `run_verify.sh` deletes the tarball only
after `xz | tar` finishes, so peak scratch holds both copies at once: 93.8 GB
compressed plus ~192 GB extracted, about 290 GB, plus results. It prints
nothing at all while it runs.

More cores will not speed it up, but not for the reason you would guess:
the archive is 8180 blocks and the image ships xz 5.8.3, so `-T0` can and
does parallelise the decode. It is I/O bound. Measured locally, xz sits at
~42% of a single core while output runs at ~24 MB/s -- an order of magnitude
under what the filesystem sustains -- because it reads 93.8 GB and writes
191.7 GiB across the same mount while creating 199 one-gigabyte files.
Local rate was ~1.4 volumes/min. On CHTC, where the job runs against local
scratch rather than a Docker bind mount, expect this to be faster; time it
before trusting the 2 h figure for `request_disk` planning.

A one-off interactive job
(`condor_submit -i`) can extract once and re-tar a 20-volume subset as a
plain `.tar` for fast reruns; point `data=` at it and drop `request_disk`.

`tail -f logs/tier1_<Cluster>_0.out` shows per-volume progress with a running
SSIM. `runs/<Cluster>/tier1_report.json` has the verdicts.

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
  variant and does not correspond to either paper table.

The reference values in the harness are for `random`, the knee convention.

## W&B key handling

The key is read only from the `WANDB_API_KEY` environment variable of the
submitting shell and copied into the job by HTCondor's `getenv`. It is never
written into a `.sub` file, a script, or this directory. It is visible in the
job ClassAd (`condor_q -l`) to you and CHTC admins, which is the standard
CHTC pattern. Without a key the harness logs offline under `results/wandb/`
and `wandb sync runs/<Cluster>/wandb/offline-run-*` uploads it later.

Rotate the key that is in the old `Research/inpainting.sub` git history
before using any key here.

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

## Training is not here: see `../training/`

This directory only evaluates the released checkpoint. Training the two
models from scratch lives in `../training/` and has its own runbook,
`../training/README.md`:

| Model | `accelerations` | `center_fractions` | Submit with |
|---|---|---|---|
| model1 | `4` | `0.08` | `make submit-model1` |
| model2 | `2 4 6 8` (one drawn at random per sample) | `0.16 0.08 0.0533 0.04` | `make submit-model2` |

Both use the `train_varnet_demo.py` defaults for everything else (8
cascades, 18 channels, Adam lr 1e-3, batch 1, 50 epochs,
`equispaced_fraction` masks). The training image is a separate
`Dockerfile` in `training/` with the same pins as this one.

Tier 2 (Table 2 reproduction) and Tier 3 (seed variance) from
`VERIFICATION.md` Sections 5 and 6 are also training runs and would go
through `training/`, not this harness. The per-volume CSV writer here is the
piece Tier 3 needs for paired comparisons; `volume_metrics()` and
`crop_like_evaluate()` in `verify_varnet.py` can be imported for that.
