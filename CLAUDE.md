# Project: fastMRI E2E VarNet Reproduction + Deep Parameter Interpolation

Context for any Claude Code session working in this repo.

**What this is (two phases):**
- Phase A: reproducing End-to-End Variational Network (VarNet) MRI reconstruction on the fastMRI knee dataset, on UW-Madison CHTC (HTCondor). This is baseline/infra work.
- Phase B, the actual contribution: extending VarNet with Deep Parameter Interpolation (DPI) — from Park et al., "Deep Parameter Interpolation for Scalar Conditioning" (CVPR 2026) — to condition the network on MRI acceleration rate. DPI's paper only covers diffusion/flow-matching image generation; applying it to VarNet/MRI is new work, not a documented extension.

Part of ongoing work with Prof. Kamilov's Computational Imaging Group.

**Current phase:** see `ROADMAP.md` for the live checklist — check it before assuming what's done.

**Key facts to keep in mind:**
- Compute: UW-Madison CHTC, HTCondor scheduler (NOT Slurm), access point `ap2001.chtc.wisc.edu`
- Proven container pattern (reused from a prior CHTC project, not generic CHTC docs): `universe = container`, a custom Docker image pushed to Docker Hub, referenced as `container_image = docker://<user>/<image>:latest` — not Apptainer
- Code base: `facebookresearch/fastMRI`, training entrypoint `fastmri_examples/varnet/train_varnet_demo.py`
- Data: fastMRI knee dataset, multicoil challenge, gated access from https://fastmri.med.nyu.edu/ — approved week of 2026-09-08. It lives on CHTC at `/staging/a/apryan3/fastmri/` (personal staging, sharded by the netid's first letter; the group directory `/staging/groups/kamilov_group/Kamilov-SciAI-datasets` is not readable by this account as of 2026-09-10). Staging quota is 100 GB (confirmed 2026-09-10); the full val tarball is 100.7 GB and the train split is five ~91 GB batches (`knee_multicoil_train_batch_0..4.tar.xz`), so neither full split fits and the user has chosen NOT to request a quota increase except as a last resort. Instead `/staging` is repacked into two subsets (`training/README.md` section 3b, decided 2026-09-11): `knee_multicoil_val_subset.tar` (20 volumes) and `knee_multicoil_train_subset.tar.xz` (the first ~65 GB of `train_batch_0`, ~120 volumes, streamed with an HTTP range request, never fully downloaded). `train.sub` defaults to these. All Phase A/B models must train on the same subsets; absolute numbers are not comparable to the paper. Data is pulled into job scratch via `transfer_input_files`, never read from `/staging` directly
- Observability: PyTorch Lightning's `MriModule` already logs val loss/NMSE/SSIM/PSNR + sample images to TensorBoard by default with zero config; Weights & Biases can be wired in like the prior project if wanted (needs a freshly rotated API key, injected as an env var, never hardcoded in a `.sub` file)
- Known VarNet reproduction caveats vs. the original paper: variable `center_fractions` instead of fixed center lines, joint 4x/8x acceleration training instead of separate models, no explicit acceleration-rate conditioning signal today (relevant directly to Phase B)

**Git: NEVER commit or push code yourself.** Do not run `git commit`, `git push`, `git add`, or anything else that changes the repository state, under any circumstances, even if it seems implied by the task. The user commits and pushes by hand. When a change is ready, PROVIDE a ready-to-paste `git add`/`git commit -m "..."` command with a suggested message and stop there. Read-only git commands (`git status`, `git diff`, `git log`) are fine.

**Security:** a prior CHTC project in this user's `Research/` folder has a Weights & Biases API key committed in git history (`inpainting.sub`) in a public repo. Never replicate that pattern here — no secrets in `.sub` files.

**When helping with this project:**
- Use HTCondor submit-file conventions, not Slurm/`sbatch`
- Don't assume the dataset is downloaded yet — check `ROADMAP.md`'s checklist state first
- Flag any metric comparisons against the original VarNet paper with the caveats above
- Phase B work has no reference implementation to check against — be explicit about what's untested/novel vs. established
