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
- Data: fastMRI knee dataset, multicoil challenge, gated access from https://fastmri.med.nyu.edu/ — approved week of 2026-09-08. It lives on CHTC at `/staging/a/apryan3/fastmri/` (personal staging, sharded by the netid's first letter; the group directory `/staging/groups/kamilov_group/Kamilov-SciAI-datasets` is not readable by this account as of 2026-09-10). It stays packed as NYU's `.tar.xz` and is pulled into job scratch via `transfer_input_files`, not bundled into the submit directory the way the smaller CIFAR-10 prior project did. Default staging quota is 100 GB / 1000 items: the 93.8 GB val split fits, `multicoil_train` (~931 GB unpacked) does not
- Observability: PyTorch Lightning's `MriModule` already logs val loss/NMSE/SSIM/PSNR + sample images to TensorBoard by default with zero config; Weights & Biases can be wired in like the prior project if wanted (needs a freshly rotated API key, injected as an env var, never hardcoded in a `.sub` file)
- Known VarNet reproduction caveats vs. the original paper: variable `center_fractions` instead of fixed center lines, joint 4x/8x acceleration training instead of separate models, no explicit acceleration-rate conditioning signal today (relevant directly to Phase B)

**Security:** a prior CHTC project in this user's `Research/` folder has a Weights & Biases API key committed in git history (`inpainting.sub`) in a public repo. Never replicate that pattern here — no secrets in `.sub` files.

**When helping with this project:**
- Use HTCondor submit-file conventions, not Slurm/`sbatch`
- Don't assume the dataset is downloaded yet — check `ROADMAP.md`'s checklist state first
- Flag any metric comparisons against the original VarNet paper with the caveats above
- Phase B work has no reference implementation to check against — be explicit about what's untested/novel vs. established
