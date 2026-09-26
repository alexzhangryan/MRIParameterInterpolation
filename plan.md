# Plan: Acceleration-rate conditioning for E2E VarNet via Deep Parameter Interpolation (DPI)

## Context

VarNet in `fastMRI` trains jointly on 4x/8x with no explicit acceleration signal; the mask is the only cue. Phase B of this project (see `ROADMAP.md`) conditions VarNet on the scalar acceleration rate R using DPI (Park et al., CVPR 2026, `parameter_interpolation/`): every parameter tensor gets a second learnable copy, and the forward pass uses `W(R) = λ(R)·W + (1-λ(R))·W_copy` with a learnable monotone λ. DPI is architecture-agnostic, so it avoids the norm-layer/MLP surgery a diffusion-style timestep embedding would need. Two variants are requested: the exact DPI mechanism (duplicate everything) and a light variant that duplicates only a few modules.

Everything is new work: DPI has only been applied to diffusion/flow generation, never to an unrolled MRI reconstruction network.

**Decisions made in this plan (stated, not asked, so implementation can proceed):**
- Fork into `fastmri_examples/dpi_varnet/` (repo precedent: `feature_varnet`). `fastmri/` core is untouched, so the baseline, its tests, and its checkpoints are preserved verbatim.
- Scalar source = the **nominal** acceleration recorded from the mask function (exact 4.0 / 8.0), with a padding- and center-corrected estimate from the mask as a fallback for the test split. Reason: the naive `W / mask.sum()` is biased (measured 4.29 / 8.55 for nominal 4 / 8 because `apply_mask` zeroes the padded columns after the mask is drawn).
- Light variant = `io` scope (first ConvBlock + final 1x1 conv + `dc_weight` per cascade). VarNet has **no `nn.Linear` layers**; only `Conv2d`, `ConvTranspose2d`, and the `dc_weight` scalar carry parameters (InstanceNorm is `affine=False`). Scope is a flag so `shallow` and `dc` are one-word ablations.
- λ table indexed on **log2(R)** over `[accel_min=1, accel_max=16]` with 1000 bins (R=4 → bin 500, R=8 → bin 750), so each doubling gets equal table resolution.
- Mixed-acceleration batches are split per sample in the Lightning module (batch_size=1 is the fastMRI default, so this costs nothing in practice) and the model contract stays "one λ per call", exactly like DPI.

## Reference: DPI mechanism to reproduce

From `parameter_interpolation/guided_diffusion/nn_ours.py:38-64` (`OurConv2d`) and `models/ADM/flow_adm_ours.py:421-446` (`NormalizedSoftmaxApprox`):
- `weight_copy = nn.Parameter(weight.clone().detach())` in `__init__` → both copies identical at init → DPI net == base net at init for any R.
- forward: `w = a*weight + (1-a)*weight_copy`, single `F.conv2d`. Weight-space, not output-space.
- λ: `theta = Parameter(randn(N))`, `lam = cumsum(softmax(theta))`, `lam[0] = 0`; index by `round(t*(N-1))`. Strictly monotone, in [0,1]; λ=0 → `weight_copy` anchor, λ=1 → `weight` anchor. Init ≈ linear.
- One 0-dim λ per batch (training forces a shared t per batch).
- Optimizer: separate AdamW group for `theta` (100x backbone lr, absolute 1e-3), no other regularizer.
- All parameterized layers duplicated (conv, conv-transpose, norm affine, linear). No partial variant exists in the repo.

## Files to create (all new; nothing in `fastmri/` changes)

> **Implemented 2026-09-19 in `dpi/` at the repo root, not in
> `fastmri_examples/dpi_varnet/` as sketched below.** `fastMRI/` is a pinned
> submodule this project never modifies, and a sibling directory keeps the
> DPI code next to `training/` and `verification/`, sharing their `.env`,
> their image and their job executable. The layout collapsed to five modules
> (`dpi_varnet.py` holds the lambda table and the layers as well as the
> model); `--dpi_scope` was dropped in favour of the paper's own choice of
> duplicating every learnable tensor, with `--no_dpi_sens` kept as the one
> ablation. The scalar source, the log-spaced lambda table and the separate
> phi learning rate are as designed below. See `dpi/README.md`.

```
fastmri_examples/dpi_varnet/
  __init__.py
  dpi_layers.py            LambdaTable, DPIConv2d, DPIConvTranspose2d, DPISequential,
                           DPIConvBlock, DPITransposeConvBlock, DPIUnet, unet_dpi_flags
  dpi_varnet.py            DPINormUnet, DPISensitivityModel, DPIVarNetBlock, DPIVarNet,
                           load_baseline_state_dict, dpi_param_summary
  dpi_transforms.py        DPIVarNetSample, DPIVarNetDataTransform, Recording*MaskFunc,
                           create_recording_mask_for_mask_type, estimate_acceleration_from_mask
  dpi_varnet_module.py     DPIVarNetModule(VarNetModule)
  train_dpi_varnet_demo.py copy of fastmri_examples/varnet/train_varnet_demo.py + flags
  README.md                usage, scope table, related work
tests/test_dpi_varnet.py         model/transform unit tests (reuse tests/conftest.py fixtures)
tests/test_dpi_varnet_module.py  fast_dev_run trainer test
```

Mirror the baseline files exactly: `fastmri/models/unet.py`, `fastmri/models/varnet.py`, `fastmri/pl_modules/varnet_module.py`, `fastmri/data/transforms.py:392-512`.

## 1. `dpi_layers.py`

**`LambdaTable(length=1000, accel_min=1.0, accel_max=16.0, spacing="log")`**
- `theta = nn.Parameter(torch.randn(length))`.
- `table()`: `lam = cumsum(softmax(theta))`; return `torch.cat([lam.new_zeros(1), lam[1:]])` (no in-place write, scriptable).
- `index(accel)`: `t = log2(accel/accel_min) / log2(accel_max/accel_min)` (or linear), clamp to [0,1], `round(t*(length-1)).long()`.
- `forward(accel: Tensor(B,)) -> Tensor(B,)` = `table()[index(accel)]`.

**`DPIConv2d(nn.Conv2d)` / `DPIConvTranspose2d(nn.ConvTranspose2d)`** with a `dpi: bool` ctor flag:
- `dpi=True`: `weight_copy`/`bias_copy` Parameters cloned from the base; `dpi=False`: `register_parameter("weight_copy", None)` so the layer is a plain conv that accepts and ignores `lam`. One class per layer type gives every layer the same `forward(x, lam)` signature, which is what keeps `ModuleList`/`zip` iteration scriptable in the light scopes.
- forward: local refinement `wc = self.weight_copy; if wc is not None: w = lam*w + (1-lam)*wc`, then `self._conv_forward(x, w, b)` / `F.conv_transpose2d(...)`.

**Blocks, state-dict-key compatible with the baseline** (`layers.0.weight`, `layers.4.weight`, `up_conv.{n-1}.0.layers.*`, `up_conv.{n-1}.1.weight/bias`):
- `DPIConvBlock`: `self.layers = nn.ModuleList([DPIConv2d, InstanceNorm2d, LeakyReLU(0.2, inplace=True), Dropout2d, DPIConv2d, InstanceNorm2d, LeakyReLU, Dropout2d])`; forward indexes with literal ints and passes `lam` only to indices 0 and 4.
- `DPITransposeConvBlock`: same with `[DPIConvTranspose2d, IN, LReLU]`.
- `DPISequential(nn.Sequential)`: `forward(x, lam): for m in self: x = m(x, lam)`; used for the last up level `DPISequential(DPIConvBlock, DPIConv2d(k=1, bias=True))`.

**`DPIUnet(in_chans, out_chans, chans, num_pool_layers, drop_prob, scope)`**: same construction as `Unet`, each layer's `dpi` flag from `unet_dpi_flags(scope, n)` (up lists are deepest-first, so U-Net level 0 is index `n-1`):

| scope | down | bottleneck | up_transpose / up_conv | final 1x1 | dc_weight |
|---|---|---|---|---|---|
| full | all | yes | all | yes | yes |
| shallow | levels 0,1 | no | levels 0,1 | yes | yes |
| io (light default) | level 0 only | no | none | yes | yes |
| dc | none | no | none | no | yes |
| none (ablation) | none | no | none | no | no |

`forward(image, lam)` is `Unet.forward` with `lam` threaded to every block.

## 2. `dpi_varnet.py`

- `DPINormUnet` = copy of `NormUnet` with `DPIUnet`, `forward(x, lam)`. Separate class because `fastmri/models/adaptive_varnet.py` imports `NormUnet`.
- `DPISensitivityModel` = copy of `SensitivityModel`, `forward(masked_kspace, mask, num_low_frequencies, lam)`.
- `DPIVarNetBlock(model, dpi_dc)`: `dc_weight` + optional `dc_weight_copy`; interpolates `dc` and passes `lam` to `self.model`.
- `DPIVarNet(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4, mask_center=True, dpi_scope="full", dpi_sens=True, lambda_length=1000, accel_min=1.0, accel_max=16.0, lambda_spacing="log")`:
  - attribute names `sens_net`, `cascades`, `cascades.k.model.unet`, `cascades.k.dc_weight` identical to `VarNet`, so every baseline key exists verbatim; new keys are only `*_copy` and `lambda_table.theta`.
  - `forward(masked_kspace, mask, num_low_frequencies: Optional[int] = None, acceleration: Optional[Tensor] = None)`: raise if `acceleration` is None or not all-equal; `lam = self.lambda_table(accel[:1])[0]` (0-dim, differentiable w.r.t. theta); then the baseline loop with `lam` threaded through.
- `load_baseline_state_dict(dpi_model, state_dict, strip_prefix="varnet.", init_copies=True)`: strip Lightning prefix, `load_state_dict(strict=False)`, assert no unexpected keys and every missing key ends in `_copy` or is `lambda_table.theta`, then copy `weight -> weight_copy`. This is the warm-start path (both anchors start from a trained baseline, as Shared LoRA does with its frozen backbone).
- `dpi_param_summary(model)` → `{base, copies, theta}` counts.

**Expected extra parameters** (measured on the real `VarNet`; +1000 theta included):

| scope | extra / cascade | extra in sens U-Net | total extra, 12 casc | % of 29.94M | total extra, 8 casc (demo default) |
|---|---|---|---|---|---|
| full | 2,454,339 | 484,898 | 29,937,966 | 100% | 20,120,610 |
| shallow | 77,475 | 15,394 | 946,094 | 3.2% | 636,194 |
| io | 3,279 | 738 | 41,086 | 0.14% | 27,970 |
| dc | 1 | 0 | 1,012 | 0.003% | 1,008 |

Justification for `io` as the light default: the first ConvBlock is the only layer that reads the aliased coil-combined image directly, whose artifact statistics are what changes with R; the final 1x1 projection scales the regularizer's output; and `dc_weight` is the data-consistency step size, the knob a hand-tuned unrolled solver would change per rate (Ada-MoDL conditions exactly this). `shallow` adds level-1 layers if `io` proves too weak.

## 3. `dpi_transforms.py` — the scalar source

- `_AccelRecorder` mixin overriding only `choose_acceleration()` (`fastmri/data/subsample.py:199`), which runs inside the `temp_seed` block in `MaskFunc.__call__`, so the recorded value is exactly the one used (deterministic per volume when `use_seed=True`). Module-level subclasses `RecordingRandomMaskFunc`, `RecordingEquiSpacedMaskFunc`, `RecordingEquispacedMaskFractionFunc`, `RecordingMagicMaskFunc`, `RecordingMagicMaskFractionFunc` (picklable for spawn workers; keep `.rng` so `data_module.worker_init_fn` still reseeds `transform.mask_func.rng`). `create_recording_mask_for_mask_type(...)` mirrors `create_mask_for_mask_type` (`subsample.py:478`).
- `DPIVarNetSample(NamedTuple)`: the 8 `VarNetSample` fields + `acceleration: float` (default collate → `(B,)` float64 tensor; cast to k-space dtype in the module).
- `DPIVarNetDataTransform(VarNetDataTransform).__call__`: `sample = super().__call__(...)` (both original branches untouched), then `acceleration` resolved in order: recorder value → `attrs["acceleration"]` (test/challenge h5 only) → `estimate_acceleration_from_mask(mask, padding_left, padding_right, num_low_frequencies)` using `W / (n_lf + (mask[win].sum() - n_lf) * W / W_win)`, deriving `n_lf` from the contiguous center run when it is 0 (same logic as `SensitivityModel.get_pad_and_num_low_freqs`).

## 4. `dpi_varnet_module.py` — `DPIVarNetModule(VarNetModule)`

- `__init__(... baseline args ..., dpi_scope="full", dpi_sens=True, lambda_lr=1e-2, lambda_length=1000, accel_min=1.0, accel_max=16.0, log_accelerations=(2,4,8,16), **kwargs)`: call `super().__init__`, then replace `self.varnet = DPIVarNet(...)`. Verify `hparams` captured the new args (re-call `save_hyperparameters()` in the subclass if not).
- `forward(masked_kspace, mask, num_low_frequencies, acceleration)`: fast path when all accelerations equal; otherwise loop per sample (slice `num_low_frequencies[i:i+1]` when it is a tensor, because `transforms.batched_mask_center` requires batch-1 or full-batch indices) and `torch.cat`. Exact: InstanceNorm, the hand-rolled norm, FFTs, and SSIM are per-sample; dropout is 0.
- `training_step/validation_step/test_step`: copies of the parent with `batch.acceleration` passed; `validation_step` also returns `"acceleration"`.
- `validation_step_end`: `MriModule.validation_step_end` (`mri_module.py:143-149`) rebuilds the dict with 5 keys and drops extras, so override to re-attach `acceleration`.
- `validation_epoch_end`: call super, then log `lambda/R{R}` for `log_accelerations` and `val_loss_accel{R}` grouped by rounded acceleration (`sync_dist=True`).
- `configure_optimizers`: two Adam groups, `"lambda_table" in name` → `lambda_lr`, wd 0; rest → `lr`, `weight_decay`; same `StepLR`. Default `lambda_lr=1e-2` (10x the demo's 1e-3). DPI's 100x ratio was relative to a 1e-5 backbone, i.e. absolute 1e-3; theta lives at scale ~1 under a 1000-way softmax, so 1e-1 thrashes and 1e-3 barely moves. Sweep {1e-3, 1e-2, 1e-1}.
- `add_model_specific_args`: parent args + `--dpi_scope {full,shallow,io,dc,none}`, `--no_dpi_sens`, `--lambda_lr`, `--lambda_length`, `--accel_min`, `--accel_max`. Re-declare `--sens_chans` as `type=int` (`varnet_module.py:194` has `type=float`, a pre-existing quirk) via `conflict_handler="resolve"`.

## 5. `train_dpi_varnet_demo.py`

Copy of `fastmri_examples/varnet/train_varnet_demo.py` with: `create_recording_mask_for_mask_type`, `DPIVarNetDataTransform` for train/val/test, `DPIVarNetModule(..., dpi_scope=..., dpi_sens=..., lambda_lr=..., lambda_length=..., accel_min=..., accel_max=...)`, defaults `--accelerations 4 8 --center_fractions 0.08 0.04`, `default_root_dir = log_path / "dpi_varnet" / f"dpi_{scope}"`, optional `--init_from_baseline PATH` calling `load_baseline_state_dict` before `fit`, and `path_config = Path(__file__).parents[2] / "fastmri_dirs.yaml"`. Run as `python -m fastmri_examples.dpi_varnet.train_dpi_varnet_demo` from the repo root.

## 6. Tests

`tests/test_dpi_varnet.py` (small configs `num_cascades=2, chans=4/8, pools=2, sens_chans=4, sens_pools=2`; inputs via `create_input` + `apply_mask` exactly as `tests/test_models.py::test_varnet`):
1. **Equivalence at init** over scope x R in {1,4,8,16,32}: load `VarNet().state_dict()` into `DPIVarNet` via `load_baseline_state_dict(strip_prefix="")`, outputs `allclose`; missing keys are only `*_copy`/`lambda_table.theta`, no unexpected keys.
2. **λ properties**: table non-decreasing, `[0]==0`, `[-1]` allclose 1 (fp32 softmax sums to `1-1e-7`, so not `==`); index clamps outside range; log spacing puts R=4 at bin 500 and R=8 at 750.
3. **Gradients**: at init `theta.grad` is exactly zero (since `W == W_copy`, `dL/dλ = <g, W - W_copy> = 0`; document this as expected DPI behavior), `weight_copy.grad` nonzero for R < accel_max; after perturbing all `*_copy` params, `theta.grad`, `weight.grad`, `weight_copy.grad` all nonzero.
4. **Param counts** per scope match the table above on the full-size config; `none` equals `VarNet`.
5. **Forward shape** parametrized like `test_varnet` for each scope.
6. **Mixed batch**: `DPIVarNetModule.forward` with `acceleration=[4,8]` equals concatenated single-sample calls.
7. **Transform**: recording mask returns `acceleration in {4.0, 8.0}` equal to the recorder; plain `RandomMaskFunc` falls to the estimator within 25% of nominal; branch B uses `attrs["acceleration"]`; first 8 fields equal the baseline `VarNetDataTransform` output on the same seed; `mask_func.rng` exists.
8. **Scripting**: `torch.jit.script(DPIVarNet(...))` per scope and scripted == eager. Patterns were verified on torch 1.12.1; if a newer torch rejects one, mark only this test `xfail(strict=False)` with the error in the reason.

`tests/test_dpi_varnet_module.py`: copy of `tests/test_modules.py::build_varnet_args` / `test_varnet_trainer` with the DPI transform/module, `fast_dev_run=True`, parametrized over scope {full, io}; assert `hparams["dpi_scope"]` and `lambda_table.theta.grad is not None` after fit.

## 7. Risks

- **TorchScript**: keep `**kwargs`, `isinstance` on modules, `functional_call`, and non-literal `ModuleList` indices out of forwards; batch splitting lives in the unscripted Lightning module.
- **Zero theta gradient at init** is inherent to DPI (identical copies). Copies diverge because W gets `λ·g` and W_copy gets `(1-λ)·g` with λ differing between 4x and 8x batches. Watch `lambda/R4`, `lambda/R8`; if flat, raise `lambda_lr` or use `--init_from_baseline`.
- **DDP**: all params participate in every forward for every scope (theta gets a full gradient through the softmax normalizer), so no unused-parameter issue.
- **Memory**: `full` doubles parameter + Adam-state memory (~60M params), negligible against activations at batch 1.
- **In-place LeakyReLU**: no new hazard; the interpolated weight is a fresh tensor.
- **Environment**: use the user's fresh `Fall26Research` conda env. It needs `pip install -e .` in `fastMRI/` plus `pytorch-lightning>=1.9,<2` (the repo's `validation_epoch_end` / `Trainer.from_argparse_args` hooks predate Lightning 2) and `pytest`. The old `Desktop/Research` directory is reference-only for the CHTC/W&B recipe and is otherwise irrelevant.

## 8. Verification

1. `pytest tests/test_dpi_varnet.py tests/test_dpi_varnet_module.py -v`.
2. `pytest tests/test_models.py tests/test_transforms.py tests/test_modules.py` unchanged and green; `git status` shows only new files.
3. `black --check`, `flake8`, `isort --check-only`, `mypy fastmri` (CI gates in `.github/workflows/build-and-test.yml`).
4. Smoke run on the mock or a 10-file subset: `python -m fastmri_examples.dpi_varnet.train_dpi_varnet_demo --dpi_scope io --fast_dev_run 1 --gpus 0`; then `--dpi_scope none` must track the baseline script's `validation_loss` to seed noise.
5. First real experiment grid on CHTC: baseline joint (Phase A), `dpi_full`, `dpi_io`, per-rate baselines. Free extra ablation: `--accel_min 4 --accel_max 8` makes λ(4)=0 and λ(8)=1 exactly, i.e. two fully independent parameter sets inside one model, a per-rate ceiling with shared data loading.

## Design notes and alternatives considered

- **Why no `nn.Linear`**: none exist in VarNet. The parameterized leaves are 3x3 `Conv2d` (no bias), `ConvTranspose2d` (no bias), one 1x1 `Conv2d` with bias per U-Net, and `dc_weight`. InstanceNorm has no affine params, so there is nothing to FiLM without adding parameters.
- **Ada-MoDL-style light variant** (MLP → per-channel feature scales + DC λ) was considered and rejected for the light variant because it adds new parameter types rather than duplicating existing ones, which is the DPI story; it is a good later baseline.
- **Cheap "constant map" baseline** for later: concatenate `R·1(H,W)` as an extra input channel to each NormUnet (the DPI paper's t-map baseline), ~10 lines.
- **Nominal vs realized R**: chosen nominal (exact, matches the paper's definition, discrete indices as in DPI); the corrected estimator is kept as fallback and could be swapped in with one line for a "true undersampling ratio" ablation.

## Related work: scalar / acceleration-rate conditioning in MRI reconstruction

| Work | How the acceleration rate enters the network | Notes |
|---|---|---|
| **Shared LoRA** — Zhao, Gong, Li, Xu, [arXiv 2609.06338](https://arxiv.org/html/2609.06338) (Sep 2026) | Scalar AF → MLP GateNet (1→64→64→D) → per-layer gate on LoRA residuals added to a frozen SHFormer; final layer initialised to unit gates. | Closest precedent: explicit scalar-AF conditioning of a recon net. Baselines are jointly trained adapters *without* conditioning; no per-rate ceiling. IXI 4/8/16x, fastMRI knee 8/16/32x; generalises to unseen 3/5/10x. |
| **SDUM** — Wang et al. (JHU/NVIDIA), [arXiv 2512.17137](https://arxiv.org/html/2512.17137) (Dec 2025) | Sinusoidal embedding of (mask type, acceleration, modality) + cascade index → MLPs → additive spatially-broadcast bias in every Restormer block of an unrolled model. | Ablation: +0.38 dB PSNR from conditioning. The timestep-embedding-style approach the user rejected as architecture-heavy. |
| **Ada-MoDL** — Pramanik, Bhave, Sajib, Sharma, Jacob, [arXiv 2304.11238](https://arxiv.org/abs/2304.11238) (2023) | MLP from a condition vector (field strength, acceleration, contrast) → per-channel CNN feature scales + data-consistency λ in unrolled MoDL. | Unrolled-network precedent for conditioning both regulariser and DC weight; motivates interpolating `dc_weight`. |
| **HyperRecon** — Wang, Dalca, Sabuncu, [MELBA 2022](https://arxiv.org/abs/2202.11009) / MICCAI-W 2021 | Hypernetwork generates all main-network weights from the regularisation-weight scalar. | Conditions on λ_reg, not AF, but is the weight-space family DPI belongs to (DPI ≈ hypernetwork restricted to a line segment). |
| **ULF-ZS-SSL** — van Straten et al., [MRM 96(3) 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13327438/) | Sinusoidal embedding of the unrolled iteration index added to feature maps. | Scalar = cascade index, not AF; faster convergence, little quality gain. |
| **Feature VarNet** (in this repo, `fastmri_examples/feature_varnet/`) | `acceleration` is a constructor int used structurally for block attention; the train script collapses the AF list to its mean. | Confirms nobody in the repo has plumbed a per-sample AF to the model. |
| E2E VarNet (Sriram et al. 2020) | none; separate models per rate in the paper, joint 4x/8x with no conditioning in the repo default. | The gap this work targets. |

Takeaways: no one has applied weight-space interpolation to AF conditioning; Shared LoRA is the closest cousin and a natural extra baseline; Ada-MoDL supports conditioning `dc_weight`; SDUM's additive-bias embedding is the cheap "MLP embedding" baseline analogous to DPI's own paper baselines.

## Expected gain from explicit rate conditioning (hypothesis, 2026-09-18)

Question from the 2026-09-18 meeting: what do we expect to gain by giving VarNet the acceleration rate R as a scalar, against the "blind" joint model (`training/` model2, one network trained on 2x/4x/6x/8x with no rate input)? The hypothesis on the table is that the explicit scalar is better than the blind approach. This section says why that should be true, where it should show up, how large it plausibly is, and what result would falsify it. Nothing here has been measured yet.

**The blind model is not blind to R; it is blind to which weights to use.** The mask is an input to every cascade's data-consistency term and, through `num_low_frequencies`, to the sensitivity-map net, so R is inferable from the input (mask density, aliasing strength in the coil-combined image). What the blind model cannot do is change its parameters with R: one set of U-Net weights per cascade and one `dc_weight` per cascade must serve every rate. The aliasing at 2x is mild and near noise-like; at 8x it is strong, structured, and comes with a quarter of the ACS lines, so the sensitivity maps are worse too. A single regulariser is a compromise across those regimes. This is exactly the situation DPI was built for in diffusion: the noise level is also visible in the input there, yet conditioning helps, because the optimal denoiser changes with the noise level and one weight set cannot be optimal at every level. Acceleration rate is the MRI analogue of noise level, which is the whole premise of the Phase B design.

**Where the gain should appear, in order of confidence.**

1. **At the ends of the rate range, 8x first.** The joint compromise costs most where the regimes differ most. Expect the largest per-rate SSIM / PSNR delta at 8x (hardest, fewest ACS lines), a smaller one at 2x (where the blind model likely over-regularises), and the least at 4x / 6x, which sit near the compromise point.
2. **In `dc_weight`.** In a hand-tuned unrolled solver the data-consistency step size is retuned per rate; joint training learns one value. `--dpi_scope dc` (1 extra parameter per cascade) isolates this and is the cheapest possible test of the hypothesis. Ada-MoDL conditions exactly this knob.
3. **In the sensitivity-map net.** The ACS width spans a factor of four across the rate list (0.16 to 0.04 of k-space). `--no_dpi_sens` versus default is the ablation.
4. **On rates not seen in training** (3x, 5x, 10x). The blind model has nothing to hold on to; DPI's monotone λ(R) gives a principled interpolation and a testable extrapolation. Shared LoRA reports generalisation to unseen rates from scalar conditioning. This is a secondary claim, not the headline.

**Plausible size.** Anchors from the literature, none of them VarNet: SDUM's ablation credits its (mask, rate, modality) embedding with about +0.4 dB PSNR in an unrolled network; Shared LoRA's scalar gating beats its unconditioned adapters by a few tenths of a dB; DPI's own paper reports FID gains over unconditioned and embedding-conditioned diffusion baselines. For VarNet on brain, a reasonable prior is a few thousandths of SSIM and 0.2-0.5 dB PSNR at 8x, less at 4x, against the blind model, with per-rate models (one network per R) as the ceiling. Two consequences: the effect is of the same order as seed-to-seed variance (`VERIFICATION.md` section 6 requires measuring σ before any comparison, and anything under ~2σ is not reportable from one run pair), and it must be read per rate on paired volumes; the validation mixture Lightning logs (`training/README.md`, "Masks") averages the four rates and will hide it.

**What would falsify it, and what each outcome means.**

| Outcome (per-rate, paired over volumes, beyond 2σ) | Reading |
|---|---|
| per-rate ceiling ≈ blind | The blind model already infers R well enough. No conditioning method can help on these rates; the contribution shifts to unseen-rate generalisation or to parameter efficiency, or the rate range must be widened (2x to 16x) to open a gap. |
| per-rate ≫ blind, DPI ≈ blind | Weight interpolation along one 1-D path is too restrictive for this network, or λ did not move (check `lambda/R{2,4,6,8}` in W&B; `plan.md` section 7 explains the zero-gradient start). Escalate scope (`io` → `shallow` → `full`), raise `lambda_lr`, or warm-start from the blind checkpoint. |
| per-rate ≫ blind, DPI ≈ per-rate | The hypothesis holds and DPI recovers the ceiling with one model. The remaining question is whether a cheaper conditioner (constant-map channel, MLP embedding) does the same; those are the baselines to run next. |
| DPI > per-rate | Suspect leakage or a mask / scalar mismatch before believing it (per-rate models see only their own rate's data, so DPI gaining from shared data is possible but should be small). |

**Two things that make the measurement honest.** First, `equispaced_fraction` masks: the realised rate equals the nominal R, so the scalar the network is conditioned on is the acceleration the data actually has (under the leaderboard's uncorrected `equispaced`, "4x" is ~3.2x and the label would be wrong). Second, every model in the comparison (blind, per-rate, DPI, and any embedding baseline) trains on the same `brain_multicoil_train_batch_0` and is scored on the same `brain_multicoil_val_batch_0` volumes with the same mask seed, so the paired test compares reconstructions of the same slices under the same masks.

## DPI variant grid (2026-09-24)

Requested by the supervisor on 2026-09-24 ("check several DPI variants").
The implementation in `dpi/` kept only the paper's own choice (every
learnable tensor duplicated) plus `--no_dpi_sens`; the scope flag designed in
sections 1 and 2 above is the first thing to re-add, because three of the
variants below are scopes. Parameter overheads are the section 2 table
(measured on the real 12-cascade VarNet).

| Variant | Flag(s) | Extra parameters | What it tests | Priority |
|---|---|---|---|---|
| full, from scratch | default | +100% (29.94M) | the paper's mechanism as-is; **already the first DPI run** | running |
| full, warm start | `INIT=<blind ckpt>` | +100% | the paper's actual recipe (both sets start from a trained model); also the fix if lambda does not move from scratch | 1 |
| io (light) | `--dpi_scope io` | +0.14% (41k) | first ConvBlock + final 1x1 + `dc_weight`: does conditioning the layers that touch the aliased image suffice? | 2 |
| dc | `--dpi_scope dc` | +0.003% (1,012) | only the data-consistency step size varies with R; the cheapest possible test of the hypothesis (Ada-MoDL conditions exactly this) | 3 |
| full, no sens | `--no_dpi_sens` | +98.4% | is the sensitivity-map net's share of the gain material? ACS width spans 4x across the rate list | 4 |
| shallow | `--dpi_scope shallow` | +3.2% (946k) | the step between io and full; run only if io is clearly below full | 5 |
| independent sets | `--accel_min 4 --accel_max 8` | +100% | lambda(4)=0, lambda(8)=1 exactly: two disjoint models inside one, i.e. the per-rate ceiling with shared data loading (2x and 6x then clamp to the endpoints, so read those columns as extrapolation) | 6 |
| lambda lr | `--lambda_lr 1e-2` on full | 0 | sensitivity of the schedule; only if `lambda/R4`, `lambda/R6` are flat at 1e-3 | as needed |
| linear spacing | `--lambda_spacing linear` | 0 | whether log2(R) indexing matters; low value, last | 7 |
| none (control) | `--dpi_scope none` | 0 | must track the blind baseline to seed noise; a free sanity check of the driver, not a result | with block D |

Every variant is a model2-sized run (brain batch 0, rates 2/4/6/8, 12
cascades, 50 epochs, one GPU) and is scored per rate through
`verification/` (ROADMAP Phase C, block A). The reading rules are the
"Expected gain" table above: the interesting outcomes are `io` or `dc`
matching `full` (the gain is cheap) and `full` matching the independent-sets
ceiling (one interpolation path is enough).

## Matched-parameter SDUM baseline (NV-Raw2insights-MRI), 2026-09-24

Requested by the supervisor on 2026-09-24: add
[NVIDIA-Medtech/NV-Raw2insights-MRI](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI)
to the baselines, pick an architecture with a parameter count close to
ours, keep the input setup identical, and check the implementation against
their code. The repo is the release of SDUM (Wang et al.,
[arXiv:2512.17137](https://arxiv.org/abs/2512.17137)), already in the
related-work table above as the embedding-style conditioning we did not
take. Code read on 2026-09-24 (clone at `main`, Apache-2.0 code, NVIDIA Open
Model License weights).

### What their model is, from the code

`scripts/models/latent_recon.py::create_mri_recon_model` builds
`Cascaded_SkipConnected_MRI_Recon`, a list of `num_cascades` copies of
`restormer_mri` (`scripts/models/restormer/restormer.py`). Each cascade:

- a coil-sensitivity U-Net (`CoilSensitivityModel_DCAE`, complex BasicUNet,
  features 12/24/48/96/192, instance norm, 1.10M parameters) run on the ACS
  image; **per cascade** unless `use_single_csm`, in which case only cascade
  0 has one and passes the maps on (the fastMRI VarNet pattern);
- sensitivity reduce to one complex image, z-scored (`complex_zscore`, the
  transform does this per sample), then a two-level Restormer: 3x3 stem,
  `num_blocks` transformer blocks per level (MDTA channel attention + gated
  depthwise FFN), PixelUnshuffle down / PixelShuffle up, `num_refinement`
  blocks, 3x3 output conv; residual add to the cascade input;
- soft data consistency in k-space, either a scalar `dc_weight` (VarNet's) or
  a learnable 768x768 per-mask-type `dc_weight_map`
  (`use_dc_weight_map`);
- **universal conditioning** (`time_cond`, `label_cond`): sinusoidal
  embedding of the cascade index, plus a sinusoidal embedding of one
  integer label `mask_idx*|M|*|A| + acc_idx*|A| + acq_idx` (mask family,
  acceleration, acquisition), each through a 2-layer MLP, summed, and added
  as a spatially broadcast bias inside every block's FFN. With one mask
  family and one acquisition the label is just the **index** of the rate in
  the config list (0..3 for 2/4/6/8), scaled by 10 before the sinusoid. An
  acceleration not in the list gets index -1. This matters for the
  unseen-rate probe: DPI's lambda(R) is a function of R, theirs is a lookup;
- optional cascade skips (`enable_cas_skips`, deep features passed between
  cascades), gradient checkpointing per cascade, bf16 autocast.

Released configurations (`configs/`): `small` T=6, `base` T=18, `large`
T=34, all at channels 256/512, blocks 3/6, heads 1/2, refinement 2,
`mlp_ratio` 3, LayerNorm, `num_frames` 5 (cine temporal window), per-cascade
CSM, mask-specific `dc_weight_map`. Note that the **released `small` config
has conditioning off** (`time_cond`/`label_cond` absent, default False);
only `base` turns it on. Training: SSIM loss (`ssim_l1` for small), Muon
(base) or AdamW at 1e-5 with cosine decay and weight decay 1e-3, batch 1 per
GPU with adaptive micro-batching, bf16, 160 epochs of post-training on
CMRxRecon; the paper's brain results are for a model pretrained on
CMRxRecon 2024/2025 plus fastMRI brain jointly. Their `calmetric` scores
per slice after 99.5-percentile normalisation of both images, which is the
CMRxRecon convention and not fastMRI's; we do not use it.

### Parameter counts and the pick

Counted on 2026-09-24 by instantiating their unmodified code
(`create_mri_recon_model`) under the fastMRI setting: `num_frames` 1 (static
2D, so the stem takes 2 channels, not 10), one mask family, one acquisition
label. The released numbers (230M/760M/1.4B) are the cine setting with 5
frames and three mask families, which is why the first row is below 230M.

| Configuration | Cascades | Widths | CSM | DC | Conditioning | Total | of which CSM / emb |
|---|---|---|---|---|---|---|---|
| their `small`, as released | 6 | 256/512 | per cascade | 768x768 map | off | 207.2M | 6.6M / 0 |
| their T=1 (paper Table D.1 row, 42.2M in the cine setting) | 1 | 256/512 | 1 | map | off | 34.5M | 1.1M / 0 |
| same, conditioning on | 1 | 256/512 | 1 | map | on | 38.8M | 1.1M / 4.2M |
| **SDUM-12, the pick** | **12** | **64/128** | **single** | **scalar** | **on** | **29.93M** | 1.1M / 3.2M |
| SDUM-12, conditioning off (the blind twin) | 12 | 64/128 | single | scalar | off | 26.7M | 1.1M / 0 |
| SDUM-12, per-cascade CSM | 12 | 64/128 | per cascade | scalar | on | 42.0M | 13.2M / 3.2M |
| SDUM-12 at 80/160, per-cascade CSM | 12 | 80/160 | per cascade | scalar | on | 57.8M | 13.2M / 5.0M |
| SDUM-6 at 96/192 | 6 | 96/192 | single | scalar | on | 33.0M | 1.1M / 3.6M |
| SDUM-12 at 64/128 with their DC map | 12 | 64/128 | single | map | on | 37.0M | 1.1M / 3.2M |

Reference: VarNet 29,936,966; DPI full 59,874,932.

**Pick: SDUM-12 at widths 64/128, single CSM, scalar DC, conditioning on
(29.93M, within 0.03% of VarNet).** Reasons, in order: same unroll depth as
VarNet (12 cascades), so the comparison is regulariser-vs-regulariser rather
than depth-vs-width; same sensitivity-map arrangement as VarNet (one
estimator feeding every cascade) and a scalar DC weight, so the only
architectural differences are the Restormer block and the conditioning;
everything else is a config value in their code (`channels`, `num_cascades`,
`use_single_csm`, `use_dc_weight_map`, `time_cond`, `label_cond`), so the
port changes no model code. Its conditioning-off twin (26.7M) is the second
row: it isolates their embedding-style conditioning at fixed architecture,
which is the same +0.38 dB ablation their paper reports (Table 3f) and the
class of baseline DPI's own paper compares against. The paper-native T=1
(34.5M / 38.8M) is the fallback if a single-cascade comparison is wanted;
it is the smallest configuration the authors themselves report, but one
unrolled step against twelve is not the comparison the paper needs.

If the DPI comparison should be at DPI's count rather than VarNet's, the
80/160 per-cascade-CSM row (57.8M) is the match for 59.9M; it is a second
run, not a replacement.

### Input setup: what "identical" means here, and what is deliberately not

Enforced the way `dpi/` enforces it: `sdum/train_sdum.py` calls
`training/train_wandb.py::cli_main` with only `build_model` replaced, so
everything below the model is the baseline's code path, not a copy.

| Kept identical (baseline's) | Their code does instead | Decision |
|---|---|---|
| data: `brain_multicoil_train_batch_0` / `val_batch_0`, fastMRI h5 | CMRxRecon `.mat` + JSON descriptors; their `FastMRIReader` exists but is not wired into `train.py` | ours |
| masks: fastMRI `EquispacedMaskFractionFunc`, rates 2/4/6/8, centre fractions 0.16/0.08/0.0533/0.04, one draw per slice in training, filename-seeded per volume at validation | their `EquispacedKspaceMask` is the same fraction-corrected formula, but drawn by MONAI's RNG, so the per-volume masks would differ | ours, so every model sees the same mask on the same slice |
| scalar fed to the model: nominal R from the recording mask function | `acc_factor` from the mask filename | ours; mapped to their `acc_idx` in the wrapper |
| loss: fastMRI `SSIMLoss` on the centre-cropped RSS, `data_range = attrs["max"]` | SSIM (base) / SSIM+L1 (small), `data_range` from `max` attr when present, on RSS after their post-processing | ours; same loss as the VarNet and DPI runs |
| metrics: `fastmri.evaluate` per volume via `verification/` | `calmetric`, per slice after 99.5-percentile normalisation | ours |
| optimiser: Adam 3e-4, x0.1 at epoch 40, batch 1, 50 epochs, fp32, no weight decay, no augmentation | Muon or AdamW 1e-5 cosine, wd 1e-3, bf16, adaptive micro-batch, k-space augmentations (off in `small`), 160 epochs | ours, **stated deviation**: the comparison is "same training recipe as VarNet", and Muon is theirs to tune. A Muon run is an optional extra if SDUM-12 looks undertrained at epoch 50 |
| in-model normalisation | complex z-score of the coil-combined image per sample (in their transform) | keep theirs inside the wrapper: it is part of the architecture, as `NormUnet`'s normalisation is part of VarNet |
| ACS extraction for the sensitivity net | `get_acs_region` grows a box from the mask centre; fastMRI uses `num_low_frequencies` | keep theirs, and assert on real masks that both give the same ACS width (they should: the centre block is contiguous) |
| gradient checkpointing per cascade, cascade skips | on | keep on: neither changes the function, only memory and the skip path (skips are part of the architecture) |

### Correctness check against their code (the supervisor's condition)

1. **Key/shape equality.** Instantiate the port at their released `small`
   configuration (T=6, 256/512, 5 frames, 3 mask types) and assert the
   `state_dict` keys and shapes equal the `nv_raw2insights_mri_small`
   checkpoint's (auto-downloaded from Hugging Face by their code).
2. **Output equality on their example.** Load those weights into their
   unmodified `scripts/inference.py` path and into the port, run the example
   case that ships in `example/` through both, assert `allclose` at fp32.
   This is the test that the port is their network and not a rewrite of it.
3. **Config sweep.** Assert the port's parameter count equals theirs for the
   pick and for the blind twin (table above), from the same config dict.
4. **Input parity on fastMRI.** One brain slice through the wrapper: the ACS
   width their `get_acs_region` finds equals `num_low_frequencies` from the
   recording mask function; the label index equals the position of R in the
   rate list; and the output shape and centre crop match what
   `VarNetModule` hands the loss.
5. **Smoke.** `make smoke` / `make job-smoke` on synthetic phantoms, as
   `dpi/` does, before any GPU time.

### What it needs

- `sdum/` sibling of `dpi/`: the two ported modules (with their Apache
  headers and a pointer to the commit), `sdum_module.py` (a `VarNetModule`
  subclass with the wrapper as `self.varnet`, `acceleration` threaded as in
  `DPIVarNetModule`), `train_sdum.py`, `train.sub`, `Makefile`, tests.
- `monai`, `timm`, `einops` in `training/Dockerfile` and
  `verification/Dockerfile`; new image tag.
- Memory: MDTA attention is over channels, so cost is linear in pixels; at
  widths 64/128, batch 1, 640x320 with checkpointing, it fits the 24 GB
  slots the VarNet run uses. Measure on the short test job before the real
  submit.
