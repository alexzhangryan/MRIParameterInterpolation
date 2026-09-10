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
