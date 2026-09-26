#!/usr/bin/env python
"""
make_tiny_ckpt.py: toy Lightning checkpoints of both kinds, for the harness
smoke test (`make smoke-ckpt`).

Writes, from untrained modules at a toy size, in the format
../training/train_wandb.py and ../dpi/train_dpi.py actually produce
(Lightning's `state_dict` plus `hyper_parameters`):

    <out>/tiny_varnet.ckpt    VarNetModule      (the baseline's checkpoint)
    <out>/tiny_dpi.ckpt       DPIVarNetModule   (a ../dpi checkpoint)

They exist so verify_varnet.py's checkpoint path can be exercised without a
trained model: the `varnet.` subtree selection (a module checkpoint also
holds `loss.w`, which a strict load into the bare network must not see), the
DPI detection, the lambda settings read from hyper_parameters, and the rate
argument to the forward. Numbers scored with these are meaningless.
"""

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("ckpttest"))
    p.add_argument("--num_cascades", type=int, default=2)
    p.add_argument("--chans", type=int, default=4)
    p.add_argument("--sens_chans", type=int, default=4)
    p.add_argument("--pools", type=int, default=2)
    p.add_argument("--sens_pools", type=int, default=2)
    p.add_argument("--lambda_length", type=int, default=16)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    for d in ("training", "dpi"):  # train_wandb.py, dpi_module.py / dpi_varnet.py
        sys.path.insert(0, str(here.parent / d))
    import pytorch_lightning as pl
    from fastmri.pl_modules import VarNetModule
    from dpi_module import DPIVarNetModule

    torch.manual_seed(0)
    arch = dict(num_cascades=args.num_cascades, pools=args.pools, chans=args.chans,
                sens_pools=args.sens_pools, sens_chans=args.sens_chans)
    modules = (
        ("tiny_varnet.ckpt", VarNetModule(**arch)),
        ("tiny_dpi.ckpt", DPIVarNetModule(**arch, lambda_length=args.lambda_length)),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, module in modules:
        ckpt = {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
        }
        path = args.out / name
        torch.save(ckpt, path)
        non_net = sorted(k for k in ckpt["state_dict"] if not k.startswith("varnet."))
        print(f"wrote {path}: {len(ckpt['state_dict'])} tensors "
              f"({len(non_net)} outside varnet.*: {non_net}), "
              f"hyper_parameters {sorted(ckpt['hyper_parameters'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
