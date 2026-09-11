#!/usr/bin/env python
"""
train_wandb.py: E2E VarNet training driver for CHTC.

A thin wrapper around the `fastmri` package's own VarNetModule and
FastMriDataModule, the same objects fastmri_examples/varnet/train_varnet_demo.py
builds. Nothing in the fastMRI/ submodule is modified or imported by path; the
package is used as installed in the image (/opt/fastMRI, pinned commit).

Differences from train_varnet_demo.py, all of them needed for a job that runs
in an HTCondor scratch directory and can be evicted at any moment:
  * --data_path and --default_root_dir come from the command line, not from a
    fastmri_dirs.yaml next to the script (there is none inside the job).
  * A Weights & Biases logger is attached when a key is available, with a
    stable run id so a resumed job continues the same W&B run.
  * ModelCheckpoint keeps `last.ckpt` in addition to the best-by-metric file,
    so a resume picks up the most recent epoch, not the most recent improvement.
  * If --resume_from_checkpoint is not given, the newest checkpoint already in
    <default_root_dir>/checkpoints is used. This is what makes an evicted job
    (whose output/ HTCondor brought back) continue instead of restarting.

Model 1:  --accelerations 4        --center_fractions 0.08
Model 2:  --accelerations 2 4 6 8  --center_fractions 0.16 0.08 0.0533 0.04
The mask function pairs the two lists elementwise and draws one pair uniformly
at random per training sample (fastmri.data.subsample.MaskFunc), so model 2 is
a single network trained jointly on four rates with no explicit rate signal.
"""

import os
import pathlib
from argparse import ArgumentParser
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule


def latest_checkpoint(checkpoint_dir: pathlib.Path) -> Optional[str]:
    last = checkpoint_dir / "last.ckpt"
    if last.is_file():
        return str(last)
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None


def build_logger(args, root: pathlib.Path):
    if not args.wandb:
        return True  # Lightning's default TensorBoardLogger under default_root_dir
    return WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        id=args.wandb_id or args.run_name,
        resume="allow",
        save_dir=str(root),
        log_model=False,
    )


def cli_main(args):
    pl.seed_everything(args.seed, workers=True)

    if len(args.center_fractions) != len(args.accelerations):
        raise SystemExit(
            f"--center_fractions ({args.center_fractions}) and --accelerations "
            f"({args.accelerations}) must have the same length; they are paired elementwise"
        )

    root = pathlib.Path(args.default_root_dir)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------
    # data
    # ------------
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()

    use_ddp = args.strategy is not None and str(args.strategy).startswith("ddp")
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=use_ddp,
    )

    # ------------
    # model
    # ------------
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=int(args.sens_chans),
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=1,
            save_last=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]
    logger = build_logger(args, root)

    resume = args.resume_from_checkpoint or latest_checkpoint(checkpoint_dir)
    print(f"[train_wandb] run_name={args.run_name} accelerations={args.accelerations} "
          f"center_fractions={args.center_fractions} mask_type={args.mask_type}")
    print(f"[train_wandb] checkpoints -> {checkpoint_dir}")
    print(f"[train_wandb] resume from: {resume or 'nothing (fresh run)'}")

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(root),
        resume_from_checkpoint=None,
    )
    if args.wandb and trainer.is_global_zero:
        logger.log_hyperparams(
            {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
        )

    if args.mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=resume)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module, ckpt_path=resume)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])

    parser.add_argument("--mode", default="train", choices=("train", "test"))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--run_name", default="varnet", type=str,
        help="W&B run name and, unless --wandb_id is given, the W&B run id. Keep it the "
             "same across resubmissions of one training run so W&B continues the run.",
    )

    # data transform params (paired elementwise, see module docstring)
    parser.add_argument(
        "--mask_type", choices=("random", "equispaced_fraction"),
        default="equispaced_fraction", type=str,
    )
    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int)

    # W&B
    parser.add_argument("--no_wandb", dest="wandb", action="store_false",
                        help="disable Weights & Biases; TensorBoard logs only")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "fastmri-varnet-train"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_id", default=None, help="override the W&B run id (default: --run_name)")

    # data module: --data_path, --challenge, --batch_size, --num_workers, --sample_rate, ...
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(challenge="multicoil", batch_size=1, test_path=None)

    # model: --num_cascades, --chans, --lr, ...  (paper / demo defaults below)
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=8,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
    )

    # trainer: --gpus, --max_epochs, --default_root_dir, --resume_from_checkpoint, ...
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1,
        replace_sampler_ddp=False,  # required for fastMRI's per-volume dispatch during val
        deterministic=True,
        max_epochs=50,
        default_root_dir="output",
    )

    args = parser.parse_args()
    if args.data_path is None:
        parser.error("--data_path is required (directory containing multicoil_train/ and multicoil_val/)")
    return args


if __name__ == "__main__":
    cli_main(build_args())
