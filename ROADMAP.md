# fastMRI E2E VarNet Reproduction + Deep Parameter Interpolation — Roadmap

**Project (two phases):**
- **Phase A (this week):** reproduce End-to-End Variational Network (VarNet) results on the fastMRI knee dataset on CHTC.
- **Phase B (the actual research contribution, later):** extend VarNet with **Deep Parameter Interpolation (DPI)** — from Park et al., "Deep Parameter Interpolation for Scalar Conditioning" (CVPR 2026) — to condition the network on MRI acceleration rate, instead of training one model per rate or training jointly with no explicit conditioning (VarNet's current default behavior).

Phase A exists to get a working, correctly-instrumented baseline before touching the DPI idea — you need a VarNet you understand and trust end-to-end before swapping parameters inside its cascades.

**Lab:** Kamilov Computational Imaging Group (continuing from the diffusion model collapse / ECE 399 work on CHTC).

---

## Where this actually stands (2026-09-18)

**The dataset changed.** Group-directory access came through and the full
fastMRI **brain** multicoil set (20 NYU tarballs, ~1.37 TB, compressed) is in
`/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/`,
transferred and SHA256-verified by `verification/prepare_brain_staging.sh`.
The 2026-09-18 meeting set the training target: **E2E VarNet on multicoil
brain, acceleration rates 2, 4, 6, 8**, with the training configuration and
the mask generation checked against the paper. `training/` was repointed the
same day; the knee subsets in personal staging are superseded but still
reachable through `dataset=knee` macros.

| Piece | State |
|---|---|
| `training/` (`train.sub`, `run_train.sh`, `train_wandb.py`, `submit.sh`) | repointed to brain batch 0 of each split, group staging (listing confirmed 2026-09-18), `runs/brain/<model>/`; model2 logs to the W&B run **"mixed acceleration brain"** (id `mixed-acceleration-brain`) by convention, `NAME='...'` / `name=` to choose, model1 to `varnet-brain-model1`; the key comes from the repo-root `.env`; a missing or rejected key is caught before extraction and **holds the job** (exit 4, `on_exit_hold`), so nothing trains without W&B unless `OFFLINE=1`; defaults now 12 cascades / Adam 3e-4 (paper + leaderboard script) instead of the demo's 8 / 1e-3; smoke-tested locally on synthetic data. **Not yet submitted in this form.** |
| Configuration audit (meeting item 1) | `training/README.md` "Configuration": every setting against the paper, the leaderboard script, and the demo. Batch size and LR schedule are not in the paper; effective batch 1 vs the leaderboard's 32 is the one real deviation |
| Mask audit (meeting item 3) | `training/README.md` "Masks": equispaced lines with density correction, rate drawn uniformly per training slice, per volume at validation, no rate input to the network |
| Gain hypothesis (meeting item 4) | `plan.md` "Expected gain from explicit rate conditioning" |
| Docker images | **must be rebuilt and pushed as `genjigod/fastmri-train:2026-09-18` and `genjigod/fastmri-verify:2026-09-18`** (both Dockerfiles now pin `wandb==0.26.1`). Found 2026-09-18 by the new preflight: the `2026-09` images' wandb 0.18.7 rejects the 86-character W&B key in `.env` on length; the key itself verifies with wandb 0.26.1. Both `.sub` files already name the new tags |
| `verification/` | wandb pin and image tag bumped with training's; otherwise unchanged; scores brain checkpoints via `val_data=` / `ckpt=` overrides (the defaults still say knee) |
| Knee subsets in `/staging/a/apryan3/fastmri/` | built 2026-09-12..14 (`knee_multicoil_{train,val}_subset`), used by short test jobs only, superseded |
| Phase B (DPI) | designed in `plan.md`, no code written |

The immediate next actions: on the laptop, push the rebuilt images
(`make push IMAGE=genjigod/fastmri-train:2026-09-18` in `training/`, same
for `verification/`); then a CHTC session: `git pull` in `~/Fall26Research`,
`ls -la` the brain directory to size `request_disk`, the 4.1 short test job
(which now also proves W&B end to end), then `make submit-model2`. Runbook:
`training/README.md` sections 1, 3a and 4.

<details>
<summary>Previous status block (2026-09-11), kept for the record</summary>

Everything below is built and green locally. **Nothing has run on CHTC yet** — no
job has been submitted, no tier of `VERIFICATION.md` has been executed, no
training has started, and `/staging` has not been confirmed to hold any fastMRI
data.

| Piece | State |
|---|---|
| `verification/` harness (Tiers 0 + 1), Docker image, submit files, local `make local-run` | written, smoke-tested on synthetic data |
| `training/` driver + job executable + `train.sub`, local `make smoke` / `make job-smoke` | written, smoke-tested on synthetic data, checkpoint/resume contract asserted |
| Staging repack (`make_subset.sh`, `subset_val.sub`, `subset_train.sub`) | written, exercised on synthetic archives, **never run on CHTC** |
| `train.sub` image macro | set to `docker://genjigod/fastmri-train:2026-09` |
| `verify.sub` image macro | still `CHANGE_ME` — edit before the first verification submit |
| Data in `/staging/a/apryan3/fastmri/` | unconfirmed; assume nothing is there until `ls` says otherwise |
| Phase B (DPI) | designed in `plan.md`, no code written |

</details>

---

## Security note — do this first

Your previous CHTC project (`Research/` on your Desktop, the diffusion model collapse work) has a **live-looking Weights & Biases API key hardcoded in `inpainting.sub`**, committed to git history and pushed to the public repo `alexzhangryan/Diffusion-Model-Collapse`. Rotate that key in your W&B account settings before starting new work. Going forward, don't write secrets into `.sub` files that get committed — inject them via an untracked local env file or CHTC's environment-injection mechanisms instead.

---

## Definition of done for Phase A this week

- [x] fastMRI knee dataset access requested — approved week of 2026-09-08, presigned URLs in hand
- [x] CHTC GPU environment set up and reproducible — two Docker images (`verification/Dockerfile`, `training/Dockerfile`, both pinned to fastMRI `91f2df4`, Lightning 1.9.5, torch 2.0.1+cu118, conda h5py, `linux/amd64`), submit files, and runbooks. Not yet exercised on a CHTC slot
- [x] Training runs end-to-end on a small local subset (smoke test) — `training/make smoke` (the driver) and `training/make job-smoke` (the job executable as HTCondor runs it, twice, to assert resume-after-eviction), both on synthetic phantoms
- [ ] A real training job is submitted to CHTC and producing checkpoints
- [x] Known deviations from the paper's setup are documented — `VERIFICATION.md` sections 1 and 5.4, `training/README.md` "Known deviations from Sriram et al. 2020". The reduced-subset deviation (section 3b) is the biggest one and is new as of 2026-09-11

## Reusing your proven CHTC recipe

Your diffusion-collapse project already has a working CHTC pattern — reuse it rather than starting from generic CHTC docs (which point new users at Apptainer; you have something proven already):

- **Container:** `universe = container`, `container_image = docker://<dockerhub_user>/<image>:latest` — build locally, `docker push` to Docker Hub, reference it in the `.sub` file
- **Access point:** `ap2001.chtc.wisc.edu`
- **Submit wrapper:** a `submit.sh` that runs `mkdir -p output snapshots logs` before `condor_submit`, so required directories always exist
- **Resource lines to adapt from `Research/diffusion.sub`:** you previously used `request_cpus=4 request_memory=16GB request_disk=50GB request_gpus=1` with `requirements = (Target.CUDACapability >= 6.0) && (Target.CUDAGlobalMemoryMb >= 8192)` — size these up for VarNet (defaults to `--gpus 2`, multicoil k-space slices are much larger than 32x32 CIFAR images)
- **Real difference this time:** CIFAR-10 fit inside `experiment.tar.gz` via `transfer_input_files`. The fastMRI knee dataset (hundreds of GB) won't — that's why Phase 1 below still routes it through `/staging` rather than bundling it into the job's transferred tarball
- **Logs:** `stream_output = True` / `stream_error = True` worked well before — keep it, `tail -f logs/..._0.out` to watch training live
- **Resume-on-eviction:** your old setup handled this in your own Python code (snapshots per generation). VarNet's Lightning `ModelCheckpoint` + PL's own resume logic does this for you already — confirm it actually picks the latest checkpoint back up after an eviction/requeue before trusting a multi-day run to it

## Observability

You don't have to build this yourself. `fastmri`'s `MriModule` (the Lightning module VarNet's training module extends) already logs, out of the box, every validation epoch:

- scalar metrics: validation loss, NMSE, SSIM, PSNR
- up to 16 example reconstruction images (target / reconstruction / error map)
- a `ModelCheckpoint` callback tracking the best `validation_loss`

All of it goes to **TensorBoard by default** — PyTorch Lightning's default logger when `Trainer` isn't given one explicitly. Nothing to configure to get a first look at training curves and sample reconstructions.

If you want continuity with your diffusion-collapse project's workflow (which logged to Weights & Biases): pass a `WandbLogger` into the `Trainer` in `train_varnet_demo.py`, or call `wandb.init`/`wandb.log` manually the way `inpainting.py` did — using a freshly rotated key, injected as an environment variable, never hardcoded in the `.sub` file.

---

## Phase 0 — Dataset access (done)

- [x] Applied at https://fastmri.med.nyu.edu/ and accepted the **knee** Data Sharing Agreement (knee is what the original VarNet paper and `train_varnet_demo.py` defaults are built around) — approved week of 2026-09-08
- [x] Asked about a lab copy: the group has `/staging/groups/kamilov_group/Kamilov-SciAI-datasets`, but this account cannot read it as of 2026-09-10. Worth pursuing anyway — it is the only way the training split fits (see Phase 2)

The presigned S3 URLs in the approval email are credentials and expire roughly
90 days after issue (this batch around 2026-12-08). Re-request from
fastmri@med.nyu.edu if they lapse before the data is staged.

## Phase 1 — CHTC environment (do in parallel, don't wait on Phase 0)

- [ ] Confirm your CHTC account still works at `ap2001.chtc.wisc.edu` (same account as the diffusion-collapse work) or file a new account request if this is a separate allocation
- [x] fastMRI vendored as a submodule at commit `91f2df4` (`fastMRI/`), never modified. The DPI reference code is a second submodule, `parameter_interpolation/`
- [x] Docker images built for both stages, `linux/amd64`, conda `h5py` (pip's leaks over a multi-day run), fastMRI installed at the pinned commit, and a build-time self-check that asserts Lightning is 1.x. **Push to Docker Hub still to confirm**; `train.sub` already points at `docker://genjigod/fastmri-train:2026-09`, `verify.sub` still says `CHANGE_ME`
- [x] Submit files written: `training/train.sub` (container universe, 1 GPU, `ON_EXIT_OR_EVICT` so an evicted job resumes from its own `output/`), `verification/verify.sub` + `verify_tier0.sub`, and the two one-off repack jobs `training/subset_val.sub` / `subset_train.sub`. Each has a `submit.sh` wrapper that creates the output directories and refuses to submit without a W&B key unless `OFFLINE=1`
- [x] `/staging` confirmed working on `ap2001`: **`/staging/a/apryan3/`** (personal staging, sharded by the netid's first letter). `/home` is 40 GB and holds code only — the dataset was never going to fit there. The shared `/staging/groups/kamilov_group/Kamilov-SciAI-datasets` is not readable by this account

## Phase 2 — Data transfer

- [x] **Brain (2026-09-16..18, the current data).** Group-directory access
  granted. `verification/prepare_brain_staging.sh` fetched all 20 NYU brain
  tarballs (~1.37 TB: 10 train batches, 3 val, 3 test, 3 fully-sampled test,
  DICOM) into `/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/`
  and verified them against NYU's `SHA256`. Nothing is extracted there (10,000-item
  cap); jobs pull one `.tar.xz` per split into scratch. `train.sub` defaults to
  batch 0 of train and val.

The knee material below is what happened before that and is kept for the
record; the knee subsets still exist in personal staging.

Target (knee): **`/staging/a/apryan3/fastmri/`**. Archives stay packed in `/staging`;
HTCondor transfers them into job scratch, where `run_verify.sh` / `run_train.sh`
extract them. A job never reads `/staging` directly.

NYU ships each split as one or more large `.tar.xz`, and **neither full split
fits the 100 GB quota**, so there *is* repackaging to do — see the decision
below and `training/README.md` section 3b.

- [ ] Fetch `knee_multicoil_val.tar.xz` (93.8 GB) straight from the NYU presigned URL onto CHTC with `verification/prepare_staging.sh`, run on `transfer.chtc.wisc.edu` under tmux with `PARALLEL=1..4`
- [ ] Do **not** scp the desktop copy up: that connection is per-flow shaped, so a 94 GB upload takes days where an S3-to-campus fetch takes hours
- [ ] Verify against NYU's `SHA256` manifest — `prepare_staging.sh` does this automatically and is idempotent and resumable
- [ ] Sanity-check that a handful of files load correctly with `h5py` before trusting the full set
- [x] **Decision 2026-09-11: do not request a quota increase (last resort).** Instead repack `/staging` into subsets that fit: `training/README.md` section 3b (`make subset-val`, then delete the full val tarball, then `make subset-train`, which streams a ~65 GB prefix of NYU's `knee_multicoil_train_batch_0` (the split ships as five ~91 GB batches) straight into `/staging`). `train.sub` now defaults to the subset names. Both jobs and `run_train.sh` were tested end to end on synthetic archives in the training image; not yet run on CHTC
- [ ] **Quota gate — superseded by the line above unless the full splits are ever wanted:** `/staging/a/apryan3` is capped at 100 GB (confirmed 2026-09-10) and the val tarball is 100,694,526,932 bytes, i.e. 93.8 GiB or 100.7 GB decimal. It fits only under binary counting and leaves nothing over either way. Request an increase before the transfer; `multicoil_train` (~931 GB unpacked) needs one regardless, as does group-directory access if that comes through

## Phase 3 — Get training actually running

- [ ] Generate `fastmri_dirs.yaml` (or pass `--data_path` directly) pointing at the data — inside a job that means the copy HTCondor transferred into scratch, not `/staging` itself
- [ ] **Smoke test first**, locally or in a short job, on a tiny subset (10-20 files) before submitting a real GPU job — catches path/env bugs without burning queue time
- [ ] Submit the real job. This is now `training/submit.sh`, not a bare
  `train_varnet_demo.py` invocation: `train_wandb.py` builds the same
  `VarNetModule` / `FastMriDataModule` and adds the W&B logger and the
  auto-resume-from-`output/checkpoints` contract a multi-day CHTC run needs.
  **Since 2026-09-18: brain data, and model2 is the primary run:**
  ```bash
  # on ap2001, in ~/Fall26Research/training, WANDB_API_KEY exported
  make submit-model2 ARGS='max_epochs=1 extra_args="--limit_train_batches 50 --limit_val_batches 10"'   # short test first
  make submit-model2   # brain, accelerations 2 4 6 8, center_fractions 0.16 0.08 0.0533 0.04
  make submit-model1   # brain, acceleration 4 only (optional per-rate reference)
  ```
  Configuration: 12 cascades, 18 / 8 channels, Adam 3e-4 with x0.1 at epoch
  40, batch 1, 50 epochs, `equispaced_fraction`, seed 42, one GPU (the paper
  and the fastMRI brain leaderboard script; the demo's 8 cascades / 1e-3 were
  dropped 2026-09-18). Data: `brain_multicoil_train_batch_0` (455 volumes) and
  `brain_multicoil_val_batch_0` (460 volumes). model2 is the
  joint-training-no-conditioning Phase B baseline; model1 is one point of the
  per-rate ceiling. Full runbook: `training/README.md` section 4.
- [x] **Known deviations from the paper documented** — the register lives in
  `training/README.md` ("Known deviations from Sriram et al. 2020") and
  `VERIFICATION.md` sections 1 and 5.4. In short:
  - variable `center_fractions` rather than the paper's fixed center-line counts (29/15 at 368 wide vs 30/16)
  - model2 trains jointly across four rates rather than one model per rate, with no explicit conditioning signal — exactly the gap Phase B targets
  - `equispaced_fraction` is the corrected-density family and matches neither paper table; `random` is the knee-leaderboard convention
  - the released checkpoint's leaderboard numbers combined `train`+`val`, so `multicoil_val` is training data for it — fine for pipeline verification (Claim A), useless as a generalization estimate
  - **and now the big one:** both models train on the section-3b subsets (~120 of 973 train volumes, 20 of 199 val volumes), so absolute numbers are not comparable to the paper at all. Every Phase A/B comparison must use the same two subsets
- [ ] A 50-epoch multicoil run will very likely exceed a single GPU job's runtime cap — confirm checkpoint resume actually works before relying on it across multiple resubmissions

## Phase 4 — Monitor and sanity-check

- [ ] Watch TensorBoard (or W&B, if wired up) as checkpoints land; compare early-epoch behavior against fastMRI leaderboard VarNet numbers to catch a broken setup early
- [ ] Keep a running log (job IDs, GPU type assigned, wall-clock per epoch)

---

## Phase B (forward-looking): Deep Parameter Interpolation for acceleration-rate conditioning

Not this week's work — capturing it now so it isn't lost.

The DPI paper (Park et al., CVPR 2026) conditions diffusion/flow-matching models on a scalar (noise level or timestep) by keeping **two learnable parameter sets per layer** and interpolating between them with a learnable monotonic function of the scalar, implemented as a cumulative softmax (guaranteed monotonic, mapped to [0,1]). It's only demonstrated on image generation (FFHQ, diffusion + flow matching) — applying it to MRI acceleration-rate conditioning is a genuinely new extension, not something the paper covers.

Rough plan to adapt it to VarNet:
- Replace each cascade's learnable parameters (conv/linear weights) with paired sets `(ω0, ω1)`
- Add the learnable monotonic function mapping acceleration rate → interpolation weight in [0,1]
- Thread acceleration rate through as an explicit scalar input — VarNet currently only "sees" it implicitly via the undersampling mask, there's no explicit conditioning signal today
- Interpolate parameters at both train and inference time based on the current sample's acceleration rate
- Train jointly across multiple rates (4x and 8x, same as the baseline default, but now with explicit conditioning)
- Baselines: VarNet trained separately per rate (expensive, a ceiling), the current joint-training-no-conditioning baseline from Phase A, and a simpler conditioning baseline like FiLM-style embedding of the rate — mirroring the baseline set DPI's own paper used (NCSNv2 rescaling, constant maps, MLP embedding conditioning)

---

## Known risks / things likely to slow this down

- fastMRI approval timeline is undocumented — start Phase 0 today regardless of what else is ready
- Dataset size vs the 100 GB `/staging` quota — resolved by repacking into subsets rather than by a quota increase (Phase 2 decision, 2026-09-11). The residual cost: a full Tier 1 run on all 199 val volumes is only possible *before* `make subset-val` deletes the full val tarball, or after re-downloading it from the NYU URL (valid to ~2026-12-08). Run Tier 1 first if that number is wanted
- The subset path has never been exercised on CHTC. The two repack jobs are the riskiest unrun thing in the repo: `subset_train.sub` streams a 65 GB HTTP range request through `xz` in a CPU slot and writes its output straight into `/staging` via `transfer_output_remaps`, which goes on hold if the quota is short
- GPU queue times can be long for high-end tiers; a mid-tier GPU is fine for a first working run
- Job runtime caps mean the real training run will not finish in one submission — expected, not a bug
- Phase B has no existing reference implementation to check against — budget real time for getting the parameter-interpolation mechanics right before trusting any results from it

## Useful links

- Dataset request: https://fastmri.med.nyu.edu/
- Code: https://github.com/facebookresearch/fastMRI (VarNet example: `fastmri_examples/varnet/`)
- E2E VarNet paper: https://arxiv.org/abs/2004.06688
- DPI paper: "Deep Parameter Interpolation for Scalar Conditioning" (Park et al., CVPR 2026)
- CHTC GPU jobs guide: https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs
- CHTC ML workflow guide: https://chtc.cs.wisc.edu/uw-research-computing/machine-learning-htc
- Your prior CHTC recipe (reference): `Research/Dockerfile`, `Research/diffusion.sub`, `Research/submit.sh` on your Desktop

---
*Last updated: 2026-09-18*
