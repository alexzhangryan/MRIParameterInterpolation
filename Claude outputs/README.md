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
| `prepare_staging.sh` | CHTC access point | Downloads `knee_multicoil_val.tar.xz` and the released checkpoint into `/staging`, verifies SHA256. |
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

## Step by step

### 1. Build and push the image (laptop, once per tag)

```bash
cd verification
docker build --build-arg IMAGE_TAG=<you>/fastmri-verify:2026-09 -t <you>/fastmri-verify:2026-09 .
docker push <you>/fastmri-verify:2026-09
```

Use a dated tag, never `:latest`, so a later rebuild cannot change what an
old result was produced with. The build ends with a self-check that prints the
installed versions and asserts Lightning is 1.x.

### 2. Smoke-test the harness locally, no real data (laptop, minutes)

```bash
docker run --rm -it -v "$PWD":/work -w /work <you>/fastmri-verify:2026-09 bash
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
# copy to /staging/<netid>/fastmri/ then:
./submit.sh data=osdf:///chtc/staging/<netid>/fastmri/knee_multicoil_val_synthetic.tar \
            request_disk=20GB extra_args="--random_init --num_cascades 2 --chans 4 --sens_chans 4 --pools 2 --sens_pools 2 --num_workers 0"
```

### 3. Stage the real data (access point, one to three hours)

```bash
export FASTMRI_VAL_URL='...knee_multicoil_val.tar.xz?AWSAccessKeyId=...'   # from the NYU email
export FASTMRI_SHA_URL='...SHA256?AWSAccessKeyId=...'
nohup ./prepare_staging.sh > prepare.log 2>&1 &
```

Check your `/staging` quota first (`get_quotas /staging/<netid>` on the
access point). The tarball is 94 GB and stays there. If quota is the
constraint, the val set is the one file worth keeping: it is the only split
with ground truth that the harness can score.

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

Extraction of the 94 GB `.xz` inside the job takes 20 to 40 minutes on
4 cores and is the reason `request_disk` is 230 GB (tarball plus extracted
volumes, the tarball is deleted once extracted). A one-off interactive job
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

## Not covered here

Tier 2 (Table 2 reproduction) and Tier 3 (seed variance) are training runs
and use `fastmri_examples/varnet/train_varnet_demo.py` from the repo, not
this harness. Their commands are in `VERIFICATION.md` Sections 5 and 6. The
per-volume CSV writer here is the piece Tier 3 needs for paired comparisons;
`volume_metrics()` and `crop_like_evaluate()` in `verify_varnet.py` can be
imported for that.
