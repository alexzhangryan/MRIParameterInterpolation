# fastMRI E2E VarNet Reproduction + Deep Parameter Interpolation — Roadmap

**Project (two phases):**
- **Phase A (this week):** reproduce End-to-End Variational Network (VarNet) results on the fastMRI knee dataset on CHTC.
- **Phase B (the actual research contribution, later):** extend VarNet with **Deep Parameter Interpolation (DPI)** — from Park et al., "Deep Parameter Interpolation for Scalar Conditioning" (CVPR 2026) — to condition the network on MRI acceleration rate, instead of training one model per rate or training jointly with no explicit conditioning (VarNet's current default behavior).

Phase A exists to get a working, correctly-instrumented baseline before touching the DPI idea — you need a VarNet you understand and trust end-to-end before swapping parameters inside its cascades.

**Lab:** Kamilov Computational Imaging Group (continuing from the diffusion model collapse / ECE 399 work on CHTC).

---

## Security note — do this first

Your previous CHTC project (`Research/` on your Desktop, the diffusion model collapse work) has a **live-looking Weights & Biases API key hardcoded in `inpainting.sub`**, committed to git history and pushed to the public repo `alexzhangryan/Diffusion-Model-Collapse`. Rotate that key in your W&B account settings before starting new work. Going forward, don't write secrets into `.sub` files that get committed — inject them via an untracked local env file or CHTC's environment-injection mechanisms instead.

---

## Definition of done for Phase A this week

- [ ] fastMRI knee dataset access requested (ideally approved and downloaded)
- [ ] CHTC GPU environment set up and reproducible (container + repo cloned + env documented)
- [ ] `train_varnet_demo.py` runs successfully end-to-end on a small local subset (smoke test)
- [ ] A real training job is submitted to CHTC and producing checkpoints
- [ ] Known deviations from the paper's setup are documented so results get interpreted correctly

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

## Phase 0 — Today: start the clock on dataset access (the actual bottleneck)

Approval time isn't published anywhere, so treat this as the critical path and kick it off before anything else.

- [ ] Go to https://fastmri.med.nyu.edu/, fill out the application form, and accept the **knee** Data Sharing Agreement (knee is what the original VarNet paper and `train_varnet_demo.py` defaults are built around)
- [ ] Check whether Kamilov's lab already has an active fastMRI data use agreement or a shared copy of the data
- [ ] If nothing back in a few days, follow up at fastmri@med.nyu.edu

## Phase 1 — CHTC environment (do in parallel, don't wait on Phase 0)

- [ ] Confirm your CHTC account still works at `ap2001.chtc.wisc.edu` (same account as the diffusion-collapse work) or file a new account request if this is a separate allocation
- [ ] `git clone https://github.com/facebookresearch/fastMRI` into your project directory
- [ ] Build a Docker image (extend the pattern from `Research/Dockerfile`) with PyTorch matching a CUDA version CHTC's GPUs support, `pip install -e .` for the `fastmri` package, and `conda install h5py=3.6.0` specifically (pip's h5py has a known memory leak that matters for a multi-day training job) — push it to Docker Hub
- [ ] Write `varnet.sub` following the `diffusion.sub` template: `universe = container`, your new image, resource requests sized for VarNet, `transfer_output_files` pointed at wherever checkpoints/logs land
- [ ] Request `/staging` access on your access point ahead of time (some access points don't have it) — the knee dataset (raw k-space + DICOM, hundreds of GB) needs `/staging`, not `/home` or a transferred tarball

## Phase 2 — Data transfer (once approved)

- [ ] Download the knee dataset (multicoil, since that's what VarNet trains on) from the NYU portal
- [ ] Package it as a small number of large archives (`.tar.gz`) rather than transferring thousands of individual `.h5` files
- [ ] Move the archive(s) to `/staging` on CHTC, unpack there
- [ ] Sanity-check that a handful of files load correctly with `h5py` before trusting the full set

## Phase 3 — Get training actually running

- [ ] Generate `fastmri_dirs.yaml` (or pass `--data_path` directly) pointing at your `/staging` data
- [ ] **Smoke test first**, locally or in a short job, on a tiny subset (10-20 files) before submitting a real GPU job — catches path/env bugs without burning queue time
- [ ] Submit the real job with the paper-matching flags:
  ```
  python train_varnet_demo.py \
    --challenge multicoil \
    --data_path /staging/<you>/fastmri/knee \
    --mask_type equispaced_fraction \
    --center_fractions 0.08 \
    --accelerations 4 \
    --num_cascades 8 \
    --chans 18 \
    --lr 0.001 \
    --batch_size 1 \
    --gpus 2 \
    --max_epochs 50
  ```
- [ ] **Document known deviations from the paper** before trusting numbers against it:
  - this implementation uses a variable `center_fractions` rather than the paper's fixed center lines
  - it trains jointly on 4x and 8x acceleration rather than separate models per the paper (this matters directly for Phase B — right now there's no explicit conditioning signal for acceleration rate at all, joint training just relies on the mask pattern differing)
  - the reported leaderboard numbers combined `train`+`val` splits — keep them separate for a valid held-out comparison
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
- Dataset size vs `/staging` quota — check your quota before the full download lands
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
*Last updated: 2026-09-07*
