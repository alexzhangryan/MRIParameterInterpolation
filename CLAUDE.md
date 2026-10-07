# Project: fastMRI E2E VarNet Reproduction + Deep Parameter Interpolation

Context for any Claude Code session working in this repo.

**What this is (two phases):**
- Phase A: training End-to-End Variational Network (VarNet) MRI reconstruction baselines on the fastMRI **brain** multicoil dataset (knee until 2026-09-18), on UW-Madison CHTC (HTCondor). This is baseline/infra work.
- Phase B, the actual contribution: extending VarNet with Deep Parameter Interpolation (DPI) — from Park et al., "Deep Parameter Interpolation for Scalar Conditioning" (CVPR 2026) — to condition the network on MRI acceleration rate. DPI's paper only covers diffusion/flow-matching image generation; applying it to VarNet/MRI is new work, not a documented extension.

Part of ongoing work with Prof. Kamilov's Computational Imaging Group.

**Current phase:** see `ROADMAP.md` for the live checklist — check it before assuming what's done.
Measured results live in `RESULTS.md`.

**Read `ROADMAP.md` as the first action of a session, and update it as the last one.**
It is the handoff document and the only context that survives a cleared conversation.
Do not reconstruct state from `git log` or the filesystem while an unread roadmap sits
there. On 2026-09-29 a stale copy cost a whole session: it claimed nothing had run on
CHTC and that Phase B had no code, when DPI had been implemented on 09-19 and the
project had already moved to brain data.

**Key facts to keep in mind:**
- Compute: UW-Madison CHTC, HTCondor scheduler (NOT Slurm), access point `ap2001.chtc.wisc.edu`
- Proven container pattern (reused from a prior CHTC project, not generic CHTC docs): `universe = container`, a custom Docker image pushed to Docker Hub, referenced as `container_image = docker://<user>/<image>:latest` — not Apptainer
- Code base: `facebookresearch/fastMRI`, training entrypoint `fastmri_examples/varnet/train_varnet_demo.py`
- Data (since 2026-09-18): the fastMRI **brain** multicoil set, all 20 NYU v2.0 tarballs (~1.37 TB compressed: `brain_multicoil_train_batch_0..9`, `brain_multicoil_val_batch_0..2`, test, fully-sampled test, DICOM), in the group staging directory `/staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/` (2.5 TB, **10,000-item cap: never extract there**), transferred and SHA256-verified by `verification/prepare_brain_staging.sh`. `train.sub` defaults to batch 0 of train (455 volumes) and val (460 volumes); one batch per split is what fits job scratch (`request_disk` ~520 GB). Every Phase A/B model must train and validate on those same two batches; absolute numbers are not comparable to the paper. Data is pulled into job scratch via `transfer_input_files`, never read from `/staging` directly. The 2026-09 knee subsets (`knee_multicoil_{train,val}_subset` in personal staging `/staging/a/apryan3/fastmri/`, 100 GB quota) are superseded but reachable with `dataset=knee` macro overrides
- Training target set at the 2026-09-18 meeting: E2E VarNet on multicoil brain at acceleration rates 2, 4, 6, 8 (`training/` preset `model2`, one rate drawn per sample, no rate input), with the configuration audited against the paper (`training/README.md` "Configuration": 12 cascades, Adam 3e-4, batch 1, 50 epochs; the demo script's 8 cascades / 1e-3 were dropped) and the mask generation audited (`training/README.md` "Masks": `equispaced_fraction`, uniform rate draw per slice). The hypothesis for Phase B (explicit rate scalar beats the blind joint model) is written up in `plan.md` "Expected gain"
- Supervisor asks of 2026-09-24 (Phase C in `ROADMAP.md`): per-rate results next meeting; a DPI variant grid (`plan.md` "DPI variant grid"); and NVIDIA's NV-Raw2insights-MRI (SDUM, arXiv:2512.17137) as a baseline at matched parameters, identical input setup, implementation checked against their code. Chosen: their 12-cascade Restormer at widths 64/128, single CSM, scalar DC, conditioning on = 29.93M (`plan.md` "Matched-parameter SDUM baseline"). Planned in a `sdum/` sibling of `dpi/` as a third `build_model` hook; not implemented yet
- **Input identity rule (supervisor, 2026-10-01):** every model in a comparison must see byte-identical inputs, meaning the same volumes, the same per-volume mask seed, and (once noise is added) the same seeded noise. The check is the zero-filled input metrics: per-volume `zf_ssim` / `zf_psnr` / `zf_nmse` must be **exactly equal** (not "close") between each treatment and its baseline. If they differ, the comparison is invalid; do not report it. Exception, found 2026-10-01: `zf_nmse`, and `zf_psnr` in the last float64 bit, differ across execute-node CPUs (`np.linalg.norm` BLAS), so across hosts compare `zf_ssim` and `zf_mse` byte-for-byte. Check this before any paired result goes into `RESULTS.md`
- Meeting of 2026-10-01 (Phase D in `ROADMAP.md`): new grid at rates **2, 6, 10**: specialists per rate, mixed blind, and DPI with a **fixed** linear λ(R) = (R−2)/8 (λ(6) = 0.5), first with the data's original noise, then with added measurement noise. Report training and validation. Measurement noise today: none is added anywhere; the only noise is the scanner's own, in both input and target. This matches the E2E VarNet paper, which adds no measurement noise either. Its only added noise is σ=0.02 output dithering for display (supplement 6.1), which is not measurement noise
- Observability: PyTorch Lightning's `MriModule` already logs val loss/NMSE/SSIM/PSNR + sample images to TensorBoard by default with zero config; Weights & Biases can be wired in like the prior project if wanted (needs a freshly rotated API key, injected as an env var, never hardcoded in a `.sub` file)
- Known VarNet reproduction caveats vs. the original paper: variable `center_fractions` instead of fixed center lines, joint multi-rate training instead of separate models, effective batch size 1 (one GPU) vs the leaderboard's 32, one NYU batch per split instead of the full set, no explicit acceleration-rate conditioning signal today (relevant directly to Phase B)

**Git: NEVER commit or push code yourself.** Do not run `git commit`, `git push`, `git add`, or anything else that changes the repository state, under any circumstances, even if it seems implied by the task. The user commits and pushes by hand. When a change is ready, PROVIDE a ready-to-paste `git add`/`git commit -m "..."` command with a suggested message and stop there. Read-only git commands (`git status`, `git diff`, `git log`) are fine.

**Security:** a prior CHTC project in this user's `Research/` folder has a Weights & Biases API key committed in git history (`inpainting.sub`) in a public repo. Never replicate that pattern here — no secrets in `.sub` files.

**When helping with this project:**
- Use HTCondor submit-file conventions, not Slurm/`sbatch`
- Don't assume the dataset is downloaded yet — check `ROADMAP.md`'s checklist state first
- Flag any metric comparisons against the original VarNet paper with the caveats above
- Phase B work has no reference implementation to check against — be explicit about what's untested/novel vs. established
