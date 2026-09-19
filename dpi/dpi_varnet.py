"""
dpi_varnet.py: End-to-End VarNet with Deep Parameter Interpolation (DPI).

DPI (Park, McCann, Garcia-Cardona, Wohlberg, Kamilov, "Deep Parameter
Interpolation for Scalar Conditioning", arXiv:2511.21028 / CVPR 2026)
conditions a network on a scalar s by keeping TWO learnable parameter sets
and running the network on their interpolation, eq. (2) of the paper:

    g(theta0, theta1, x, s) = f([1 - lambda(s)] theta0 + lambda(s) theta1, x)

with lambda monotone, lambda(s_min) = 0, lambda(s_max) = 1, and, section 3.2,

    lambda(s_i) = sum_{j <= i} softmax(phi)_j,    phi in R^S learnable.

Here the scalar is the MRI acceleration rate R and f is fastMRI's VarNet
(fastmri/models/varnet.py + unet.py at commit 91f2df4). Every learnable
tensor of VarNet is duplicated, exactly as the paper does for DRUNet and
ADM: the 3x3 and 1x1 convolution weights (and the one bias, in the final
1x1 conv of each U-Net), the transposed-convolution weights, and each
cascade's data-consistency weight. VarNet's InstanceNorm layers have no
affine parameters, so there is nothing there to duplicate. The result is
one network with 2x the parameters plus the S-entry phi.

Naming follows the reference implementation (parameter_interpolation/
guided_diffusion/nn_ours.py): `weight` is theta1 (the lambda = 1 end,
s_max = the highest acceleration) and `weight_copy` is theta0 (lambda = 0,
s_min). Both start as identical copies, so at initialisation the DPI network
computes exactly what the baseline VarNet with the same `weight` tensors
computes, for every R. Every baseline state-dict key exists here verbatim;
the only new keys are `*_copy` and `lambda_table.phi`.

Nothing in fastMRI/ is modified; the classes below are mirrors of the
baseline ones with `lam` threaded through every forward.

Untested against any published DPI-on-MRI result: none exists. The unit
tests in test_dpi.py are the evidence that the mechanics are right.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms


# ---------------------------------------------------------------------------
# the interpolation function lambda(s)
# ---------------------------------------------------------------------------
class LambdaTable(nn.Module):
    """lambda(R): a learnable, strictly monotone map from acceleration to [0, 1].

    Paper section 3.2: phi in R^S, p = softmax(phi), lambda_i = sum_{j<=i} p_j.
    Index 0 is pinned to 0 (as in the reference code, y_internal[0] = 0), and
    the last index is sum(p) = 1, so lambda(accel_min) = 0 and
    lambda(accel_max) = 1 exactly; accelerations outside the range clamp to
    the ends.

    The table index for an acceleration R is round(t * (S - 1)) with
    t = log2(R / accel_min) / log2(accel_max / accel_min) ("log" spacing:
    every doubling of R gets the same number of bins, so 2x, 4x, 8x sit at
    t = 0, 0.5, 1) or the linear equivalent.

    The cumulative sum is written as a matmul with a lower-triangular ones
    matrix rather than torch.cumsum: same numbers, but matmul has a
    deterministic CUDA kernel, and training runs with
    Trainer(deterministic=True).
    """

    def __init__(
        self,
        length: int = 1000,
        accel_min: float = 2.0,
        accel_max: float = 8.0,
        spacing: str = "log",
    ):
        super().__init__()
        if length < 2:
            raise ValueError("lambda table needs at least 2 entries")
        if not accel_max > accel_min > 0:
            raise ValueError("need 0 < accel_min < accel_max")
        if spacing not in ("log", "linear"):
            raise ValueError("spacing must be 'log' or 'linear'")
        self.length = length
        self.accel_min = float(accel_min)
        self.accel_max = float(accel_max)
        self.spacing = spacing
        # phi: the learnable vector of section 3.2; randn init as in the
        # reference code (NormalizedSoftmaxApprox), which makes softmax(phi)
        # nearly uniform and lambda nearly linear at the start.
        self.phi = nn.Parameter(torch.randn(length))
        self.register_buffer(
            "cum", torch.tril(torch.ones(length, length)), persistent=False
        )

    def table(self) -> torch.Tensor:
        """The whole lambda table, shape (length,), non-decreasing, 0 to 1."""
        p = F.softmax(self.phi, dim=0)
        lam = self.cum @ p  # lam[i] = sum_{j<=i} p_j
        # pin lambda(s_min) = 0 without an in-place write on a grad tensor
        return torch.cat([lam.new_zeros(1), lam[1:]])

    def index(self, accel: torch.Tensor) -> torch.Tensor:
        """Table index for each acceleration, clamped to the table range."""
        accel = accel.to(dtype=self.phi.dtype)
        if self.spacing == "log":
            t = torch.log2(accel / self.accel_min) / math.log2(
                self.accel_max / self.accel_min
            )
        else:
            t = (accel - self.accel_min) / (self.accel_max - self.accel_min)
        t = t.clamp(0.0, 1.0)
        return torch.round(t * (self.length - 1)).long()

    def forward(self, accel: torch.Tensor) -> torch.Tensor:
        """lambda(R) for a 1-D tensor of accelerations, shape (B,).

        Differentiable with respect to phi (an index into the table), not
        with respect to R, which is data.
        """
        return self.table()[self.index(accel)]


# ---------------------------------------------------------------------------
# the duplicated layers
# ---------------------------------------------------------------------------
class DPIConv2d(nn.Conv2d):
    """Conv2d holding two weight sets, used at lam*weight + (1-lam)*weight_copy.

    Mirrors OurConv2d in the reference implementation. `dpi=False` keeps the
    plain conv (no copy) but the same forward signature, so a scope where
    only some layers are duplicated still iterates uniformly.
    """

    def __init__(self, *args, dpi: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if dpi:
            self.weight_copy = nn.Parameter(self.weight.detach().clone())
            if self.bias is not None:
                self.bias_copy = nn.Parameter(self.bias.detach().clone())
            else:
                self.register_parameter("bias_copy", None)
        else:
            self.register_parameter("weight_copy", None)
            self.register_parameter("bias_copy", None)

    def forward(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.weight_copy is not None:
            weight = lam * weight + (1.0 - lam) * self.weight_copy
        bias = self.bias
        if bias is not None and self.bias_copy is not None:
            bias = lam * bias + (1.0 - lam) * self.bias_copy
        return self._conv_forward(x, weight, bias)


class DPIConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d with the same two-parameter-set treatment."""

    def __init__(self, *args, dpi: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if dpi:
            self.weight_copy = nn.Parameter(self.weight.detach().clone())
            if self.bias is not None:
                self.bias_copy = nn.Parameter(self.bias.detach().clone())
            else:
                self.register_parameter("bias_copy", None)
        else:
            self.register_parameter("weight_copy", None)
            self.register_parameter("bias_copy", None)

    def forward(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.weight_copy is not None:
            weight = lam * weight + (1.0 - lam) * self.weight_copy
        bias = self.bias
        if bias is not None and self.bias_copy is not None:
            bias = lam * bias + (1.0 - lam) * self.bias_copy
        return F.conv_transpose2d(
            x,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


# ---------------------------------------------------------------------------
# the U-Net, mirroring fastmri/models/unet.py
# ---------------------------------------------------------------------------
class DPIConvBlock(nn.Module):
    """fastmri.models.unet.ConvBlock with lam threaded through.

    `layers` is a ModuleList, not a Sequential, because the two convolutions
    need `lam` while InstanceNorm / LeakyReLU / Dropout do not. The indices
    are the baseline's (conv at 0 and 4), so every state-dict key
    (`layers.0.weight`, `layers.4.weight`) is unchanged.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, dpi: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers = nn.ModuleList(
            [
                DPIConv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False, dpi=dpi),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
                DPIConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False, dpi=dpi),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(drop_prob),
            ]
        )

    def forward(self, image: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        out = self.layers[0](image, lam)
        out = self.layers[3](self.layers[2](self.layers[1](out)))
        out = self.layers[4](out, lam)
        return self.layers[7](self.layers[6](self.layers[5](out)))


class DPITransposeConvBlock(nn.Module):
    """fastmri.models.unet.TransposeConvBlock with lam threaded through."""

    def __init__(self, in_chans: int, out_chans: int, dpi: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.ModuleList(
            [
                DPIConvTranspose2d(
                    in_chans, out_chans, kernel_size=2, stride=2, bias=False, dpi=dpi
                ),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ]
        )

    def forward(self, image: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        out = self.layers[0](image, lam)
        return self.layers[2](self.layers[1](out))


class DPIUnetHead(nn.ModuleList):
    """The baseline's `nn.Sequential(ConvBlock, Conv2d(1x1))` last up level.

    A ModuleList names its children `0` and `1` exactly as Sequential does,
    so the state-dict keys (`up_conv.{n-1}.0.layers.*`,
    `up_conv.{n-1}.1.weight`, `.1.bias`) are the baseline's verbatim. The
    1x1 conv is the only layer in VarNet with a bias, so it is also the only
    place `bias_copy` appears.
    """

    def __init__(self, in_chans: int, out_chans: int, final_chans: int, drop_prob: float,
                 dpi: bool = True):
        super().__init__(
            [
                DPIConvBlock(in_chans, out_chans, drop_prob, dpi=dpi),
                DPIConv2d(out_chans, final_chans, kernel_size=1, stride=1, dpi=dpi),
            ]
        )

    def forward(self, image: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return self[1](self[0](image, lam), lam)


class DPIUnet(nn.Module):
    """fastmri.models.unet.Unet with lam threaded through every block.

    Structure, attribute names and forward arithmetic are the baseline's;
    only the `lam` argument is added.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        dpi: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [DPIConvBlock(in_chans, chans, drop_prob, dpi=dpi)]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(DPIConvBlock(ch, ch * 2, drop_prob, dpi=dpi))
            ch *= 2
        self.conv = DPIConvBlock(ch, ch * 2, drop_prob, dpi=dpi)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(DPITransposeConvBlock(ch * 2, ch, dpi=dpi))
            self.up_conv.append(DPIConvBlock(ch * 2, ch, drop_prob, dpi=dpi))
            ch //= 2

        self.up_transpose_conv.append(DPITransposeConvBlock(ch * 2, ch, dpi=dpi))
        self.up_conv.append(DPIUnetHead(ch * 2, ch, self.out_chans, drop_prob, dpi=dpi))

    def forward(self, image: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        for layer in self.down_sample_layers:
            output = layer(output, lam)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output, lam)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output, lam)

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output, lam)

        return output


# ---------------------------------------------------------------------------
# VarNet, mirroring fastmri/models/varnet.py
# ---------------------------------------------------------------------------
class DPINormUnet(nn.Module):
    """fastmri.models.varnet.NormUnet with lam threaded through."""

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        dpi: bool = True,
    ):
        super().__init__()
        self.unet = DPIUnet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
            dpi=dpi,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x, lam)

        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class DPISensitivityModel(nn.Module):
    """fastmri.models.varnet.SensitivityModel with lam threaded through.

    Duplicating this net matters for acceleration conditioning: the number of
    ACS lines it sees spans a factor of four across the 2x-8x rate list
    (center fractions 0.16 to 0.04), so the map-estimation problem it solves
    is genuinely different at each rate. `dpi_sens=False` on DPIVarNet leaves
    it as a single parameter set for the ablation.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
        dpi: bool = True,
    ):
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = DPINormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
            dpi=dpi,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        lam: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images, lam), batches)
        )


class DPIVarNetBlock(nn.Module):
    """fastmri.models.varnet.VarNetBlock with lam threaded through.

    `dc_weight` is the data-consistency step size, a single learnable scalar
    per cascade. It is duplicated like every other learnable tensor, which is
    also the parameter a hand-tuned unrolled solver would retune per rate.
    """

    def __init__(self, model: nn.Module, dpi: bool = True):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        if dpi:
            self.dc_weight_copy = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("dc_weight_copy", None)

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        dc_weight = self.dc_weight
        if self.dc_weight_copy is not None:
            dc_weight = lam * dc_weight + (1.0 - lam) * self.dc_weight_copy
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), lam), sens_maps
        )

        return current_kspace - soft_dc - model_term


class DPIVarNet(nn.Module):
    """VarNet conditioned on the acceleration rate by parameter interpolation.

    Same architecture, attribute names and arithmetic as
    fastmri.models.varnet.VarNet; the additions are the `lambda_table`, the
    `acceleration` forward argument, and a second copy of every learnable
    tensor.

    One lambda per forward call, as in the paper ("a single scalar per
    batch"). The Lightning module splits a batch whose samples have
    different accelerations and calls this once per sample; at the fastMRI
    default batch_size=1 that never happens.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        dpi_sens: bool = True,
        lambda_length: int = 1000,
        accel_min: float = 2.0,
        accel_max: float = 8.0,
        lambda_spacing: str = "log",
    ):
        super().__init__()

        self.lambda_table = LambdaTable(
            length=lambda_length,
            accel_min=accel_min,
            accel_max=accel_max,
            spacing=lambda_spacing,
        )
        self.sens_net = DPISensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
            dpi=dpi_sens,
        )
        self.cascades = nn.ModuleList(
            [
                DPIVarNetBlock(DPINormUnet(chans, pools), dpi=True)
                for _ in range(num_cascades)
            ]
        )

    def lam_for(self, acceleration: torch.Tensor) -> torch.Tensor:
        """The 0-dim lambda used by one forward call.

        Requires every sample in the batch to carry the same acceleration:
        DPI interpolates the parameters once per forward, so a batch with two
        rates has no single correct answer here. DPIVarNetModule.forward
        splits such a batch before calling this.
        """
        accel = acceleration.reshape(-1)
        if accel.numel() == 0:
            raise ValueError("acceleration is empty")
        if accel.numel() > 1 and not bool(torch.all(accel == accel[0])):
            raise ValueError(
                f"DPIVarNet needs one acceleration per forward call, got {accel.tolist()}; "
                "DPIVarNetModule.forward splits mixed batches"
            )
        return self.lambda_table(accel[:1])[0]

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
        acceleration: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if acceleration is None:
            raise ValueError(
                "DPIVarNet.forward needs the acceleration rate; use "
                "DPIVarNetDataTransform, which records it from the mask function"
            )
        lam = self.lam_for(acceleration)

        sens_maps = self.sens_net(masked_kspace, mask, lam, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, lam)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dpi_param_summary(model: nn.Module) -> dict:
    """Counts of base parameters, interpolation copies, and phi."""
    base = copies = phi = 0
    for name, p in model.named_parameters():
        if name.endswith("phi"):
            phi += p.numel()
        elif name.endswith("_copy"):
            copies += p.numel()
        else:
            base += p.numel()
    return {"base": base, "copies": copies, "phi": phi, "total": base + copies + phi}


def load_baseline_state_dict(
    dpi_model: DPIVarNet,
    state_dict: dict,
    strip_prefix: str = "varnet.",
    init_copies: bool = True,
) -> dict:
    """Load baseline VarNet weights into a DPIVarNet (the warm-start path).

    Accepts a raw VarNet state dict or a Lightning checkpoint's (whose keys
    carry the `varnet.` prefix that VarNetModule adds). Every baseline key
    must exist here verbatim; the only keys allowed to be missing are the
    `*_copy` duplicates and `lambda_table.phi`. With init_copies, each copy
    is then set to its base tensor, so the loaded network reproduces the
    baseline exactly at every R.
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if strip_prefix:
        # Select the varnet subtree, do not merely rename it: a VarNetModule
        # checkpoint also holds the SSIM loss buffer (`loss.w`) and the
        # metric states, which are not part of the network.
        state_dict = {
            k[len(strip_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(strip_prefix)
        }
        if not state_dict:
            raise RuntimeError(
                f"no keys starting with {strip_prefix!r} in the checkpoint; "
                "pass strip_prefix='' for a raw VarNet state dict"
            )

    result = dpi_model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint has keys the DPI model does not: {result.unexpected_keys[:8]}"
        )
    unexplained = [
        k
        for k in result.missing_keys
        if not (k.endswith("_copy") or k == "lambda_table.phi")
    ]
    if unexplained:
        raise RuntimeError(
            f"DPI model has parameters the checkpoint does not fill: {unexplained[:8]}"
        )

    if init_copies:
        with torch.no_grad():
            params = dict(dpi_model.named_parameters())
            for name, p in params.items():
                if name.endswith("_copy"):
                    base = params.get(name[: -len("_copy")])
                    if base is not None:
                        p.copy_(base)

    return {
        "loaded": len(state_dict),
        "missing": len(result.missing_keys),
        "copies_initialised": init_copies,
    }
