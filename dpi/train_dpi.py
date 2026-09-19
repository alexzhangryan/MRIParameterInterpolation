#!/usr/bin/env python
"""
train_dpi.py: training driver for DPI VarNet on CHTC.

Deliberately thin. Everything that is not the model itself -- the W&B
preflight and logger, seeding, the data module, ModelCheckpoint, the
auto-resume-from-output/checkpoints contract, the trainer -- is
`../training/train_wandb.py`, imported and called with two hooks swapped:

    build_model       -> DPIVarNetModule instead of VarNetModule
    build_transforms  -> recording mask functions and DPIVarNetDataTransform

So a DPI run and the baseline `mixed acceleration brain` run differ in the
network and in nothing else. Every other flag, default and behaviour is
literally the same code path, which is what makes the two comparable.

    python train_dpi.py --data_path data/fastmri --default_root_dir output \
        --run_name dpi-mixed-acceleration-brain --accelerations 2 4 6 8 \
        --center_fractions 0.16 0.08 0.0533 0.04 --mask_type equispaced_fraction \
        --max_epochs 50 --gpus 1

    --init_from_baseline runs/brain/model2/<Cluster>/checkpoints/last.ckpt
        start both parameter sets from the trained baseline (the paper warm
        starts from a pretrained model). Without it, from scratch.
"""

import sys
from pathlib import Path

import torch

# Inside a job every transferred file lands in one flat directory; from a
# checkout, ../training has to be on the path.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import train_wandb
except ImportError:  # pragma: no cover - exercised only outside the job
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    import train_wandb

from dpi_module import DPIVarNetModule
from dpi_transforms import (
    DPIVarNetDataTransform,
    create_recording_mask_for_mask_type,
)
from dpi_varnet import load_baseline_state_dict


def build_dpi_transforms(args):
    """`train_wandb.build_transforms` with the recording mask function.

    Same mask family, same lists, same seeding as the baseline: the training
    transform draws a fresh rate per slice (`use_seed=False`) and the
    validation transform is seeded from the filename, so each validation
    volume is locked to one rate. Only the recording is added.
    """
    mask = create_recording_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = DPIVarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = DPIVarNetDataTransform(mask_func=mask)
    test_transform = DPIVarNetDataTransform()
    return train_transform, val_transform, test_transform


def build_dpi_model(args):
    model = DPIVarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=int(args.sens_chans),
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        dpi_sens=args.dpi_sens,
        lambda_lr=args.lambda_lr,
        lambda_length=args.lambda_length,
        accel_min=args.accel_min,
        accel_max=args.accel_max,
        lambda_spacing=args.lambda_spacing,
        log_accelerations=tuple(args.accelerations),
    )

    if args.init_from_baseline:
        path = Path(args.init_from_baseline)
        if not path.is_file():
            raise SystemExit(f"--init_from_baseline: no such file {path}")
        ckpt = torch.load(str(path), map_location="cpu")
        info = load_baseline_state_dict(model.varnet, ckpt, strip_prefix="varnet.")
        print(
            f"[train_dpi] warm start from {path.name}: {info['loaded']} baseline tensors "
            f"loaded into both parameter sets"
        )
    else:
        print("[train_dpi] from scratch (both parameter sets randomly initialised, identical)")

    return model


def build_args():
    parser = train_wandb.build_parser(module_cls=DPIVarNetModule)
    # The rate list of the run this is compared against. Everything else
    # (12 cascades, Adam 3e-4, batch 1, 50 epochs, equispaced_fraction, seed
    # 42) is already train_wandb's default, i.e. the baseline's.
    parser.set_defaults(
        accelerations=[2, 4, 6, 8],
        center_fractions=[0.16, 0.08, 0.0533, 0.04],
        run_name="dpi-varnet",
    )
    return train_wandb.build_args(parser)


if __name__ == "__main__":
    train_wandb.cli_main(
        build_args(),
        build_model=build_dpi_model,
        build_transforms=build_dpi_transforms,
    )
