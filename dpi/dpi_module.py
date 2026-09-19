"""
dpi_module.py: the Lightning module for DPI VarNet.

A subclass of fastMRI's `VarNetModule`, so every training and validation
mechanic of the baseline run is inherited unchanged: SSIM loss, the
validation metrics and image logging of `MriModule`, the checkpoint
contract. Three things differ, and only these:

  1. `self.varnet` is a `DPIVarNet` instead of a `VarNet`;
  2. each step passes `batch.acceleration` to it;
  3. `configure_optimizers` puts the interpolation vector phi in its own
     parameter group with its own learning rate, as the paper does
     (section 4.1: "we assign a separate learning rate of 1e-3 to the
     interpolation coefficients phi").

Everything else about the optimiser is the baseline's: Adam, lr 3e-4,
StepLR x0.1 at epoch 40, no weight decay. The paper used AdamW at 1e-5 with
weight decay 0.05 for its diffusion backbones; matching the fastMRI baseline
matters more here, since the whole point is a like-for-like comparison
against the model2 run.
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from pytorch_lightning.loggers import WandbLogger

from fastmri.data import transforms
from fastmri.pl_modules import VarNetModule

from dpi_varnet import DPIVarNet, dpi_param_summary

# The W&B-safe image logging lives in ../training/train_wandb.py. Inside a
# job every transferred file sits in one flat directory, so the plain import
# works there; from a checkout, training/ has to be on the path first.
try:
    from train_wandb import WandbSafeVarNetModule as _BaseModule
except ImportError:  # pragma: no cover - exercised only outside the job
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    from train_wandb import WandbSafeVarNetModule as _BaseModule


class DPIVarNetModule(_BaseModule):
    """VarNetModule whose network is conditioned on the acceleration rate."""

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        dpi_sens: bool = True,
        lambda_lr: float = 1e-3,
        lambda_length: int = 1000,
        accel_min: float = 2.0,
        accel_max: float = 8.0,
        lambda_spacing: str = "log",
        log_accelerations: tuple = (2, 4, 6, 8),
        **kwargs,
    ):
        super().__init__(
            num_cascades=num_cascades,
            pools=pools,
            chans=chans,
            sens_pools=sens_pools,
            sens_chans=sens_chans,
            lr=lr,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            weight_decay=weight_decay,
            **kwargs,
        )
        # re-capture: the parent's save_hyperparameters() ran against the
        # parent's signature and would drop every dpi_* argument, which the
        # checkpoint needs to rebuild this model.
        self.save_hyperparameters()

        self.lambda_lr = lambda_lr
        self.log_accelerations = tuple(log_accelerations)

        self.varnet = DPIVarNet(
            num_cascades=num_cascades,
            sens_chans=int(sens_chans),
            sens_pools=sens_pools,
            chans=chans,
            pools=pools,
            dpi_sens=dpi_sens,
            lambda_length=lambda_length,
            accel_min=accel_min,
            accel_max=accel_max,
            lambda_spacing=lambda_spacing,
        )

    # -- forward ----------------------------------------------------------
    def forward(self, masked_kspace, mask, num_low_frequencies, acceleration):
        """One DPI forward per distinct acceleration in the batch.

        DPI interpolates the parameters once per forward call, so a batch
        mixing rates has to be split. At the fastMRI default batch_size=1
        the fast path always applies; the loop is exact rather than an
        approximation, because every operation in VarNet is per-sample
        (InstanceNorm, the hand-rolled norm, the FFTs) and dropout is 0.
        """
        accel = torch.as_tensor(acceleration).reshape(-1)
        if accel.numel() == 1 or bool(torch.all(accel == accel[0])):
            return self.varnet(masked_kspace, mask, num_low_frequencies, accel[:1])

        outputs = []
        for i in range(accel.numel()):
            nlf = num_low_frequencies
            if torch.is_tensor(nlf) and nlf.numel() > 1:
                nlf = nlf[i : i + 1]
            # A mask with batch dim 1 is shared across the batch by
            # broadcasting (what apply_mask returns for a whole batch);
            # slicing it would hand cascade i an empty tensor.
            one_mask = mask if mask.shape[0] == 1 else mask[i : i + 1]
            outputs.append(
                self.varnet(
                    masked_kspace[i : i + 1],
                    one_mask,
                    nlf,
                    accel[i : i + 1],
                )
            )
        return torch.cat(outputs, dim=0)

    # -- steps: the parent's, with acceleration passed through ------------
    def training_step(self, batch, batch_idx):
        output = self(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.acceleration
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.acceleration
        )
        target, output = transforms.center_crop_to_smallest(batch.target, output)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            ),
        }

    def test_step(self, batch, batch_idx):
        output = self(
            batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.acceleration
        )
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size
        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    # -- what lambda is doing ---------------------------------------------
    def validation_epoch_end(self, val_logs):
        super().validation_epoch_end(val_logs)
        # lambda(R) at each trained rate. These are the numbers that say
        # whether the conditioning is doing anything: at init they sit on a
        # near-straight line, and a run where they never move means phi is
        # not learning (raise --lambda_lr, or warm-start from the baseline).
        with torch.no_grad():
            table = self.varnet.lambda_table
            accel = torch.tensor(
                [float(a) for a in self.log_accelerations], device=self.device
            )
            lam = table(accel)
        for rate, value in zip(self.log_accelerations, lam.tolist()):
            self.log(f"lambda/R{rate}", value, sync_dist=True)

    # -- phi gets its own learning rate -----------------------------------
    def configure_optimizers(self):
        lambda_params, backbone_params = [], []
        for name, param in self.named_parameters():
            (lambda_params if "lambda_table" in name else backbone_params).append(param)

        optim = torch.optim.Adam(
            [
                {
                    "params": backbone_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {"params": lambda_params, "lr": self.lambda_lr, "weight_decay": 0.0},
            ]
        )
        # StepLR scales every group, so phi's rate decays with the backbone's
        # at epoch 40. That is a choice, not the paper's: the paper holds phi
        # at a constant 1e-3 throughout.
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]

    def on_fit_start(self):
        summary = dpi_param_summary(self.varnet)
        self.print(
            f"[dpi] parameters: base {summary['base']:,} + copies {summary['copies']:,} "
            f"+ phi {summary['phi']:,} = {summary['total']:,}"
        )

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = VarNetModule.add_model_specific_args(parent_parser)
        parser.add_argument(
            "--no_dpi_sens",
            dest="dpi_sens",
            action="store_false",
            help="leave the sensitivity-map U-Net with a single parameter set (ablation)",
        )
        parser.add_argument(
            "--lambda_lr",
            default=1e-3,
            type=float,
            help="learning rate for the interpolation vector phi (paper section 4.1: 1e-3)",
        )
        parser.add_argument(
            "--lambda_length",
            default=1000,
            type=int,
            help="entries in the lambda table (paper: S = 1000)",
        )
        parser.add_argument(
            "--accel_min",
            default=2.0,
            type=float,
            help="acceleration mapped to lambda = 0, i.e. the first parameter set",
        )
        parser.add_argument(
            "--accel_max",
            default=8.0,
            type=float,
            help="acceleration mapped to lambda = 1, i.e. the second parameter set",
        )
        parser.add_argument(
            "--lambda_spacing",
            default="log",
            choices=("log", "linear"),
            help="how acceleration maps onto the table index",
        )
        parser.add_argument(
            "--init_from_baseline",
            default=None,
            type=str,
            help="Lightning checkpoint of a trained baseline VarNet to start both "
                 "parameter sets from (the paper's warm start). Default: from scratch.",
        )
        return parser
