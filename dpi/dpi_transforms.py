"""
dpi_transforms.py: get the acceleration rate of each sample to the model.

The DPI VarNet is conditioned on the acceleration rate R, so R has to travel
with the sample. fastMRI's `VarNetSample` has no field for it and the mask
that reaches the network only encodes it implicitly, so this module adds

  * `Recording*MaskFunc`: fastMRI's five mask families, each overriding the
    one method that picks the rate, so the value actually used is recorded;
  * `DPIVarNetSample`: `VarNetSample` plus `acceleration`;
  * `DPIVarNetDataTransform`: `VarNetDataTransform` plus that field.

Why record rather than measure. `MaskFunc.choose_acceleration` draws one
(center_fraction, acceleration) pair uniformly per call, inside the
`temp_seed` block of `MaskFunc.__call__`, so the recorded value is exactly
the one that produced this sample's mask: nominal, exact, and identical to
the number the rate list names. Counting sampled columns instead is biased,
because `apply_mask` zeroes the padded edges of k-space *after* the mask is
drawn, so a nominal 4x mask measures as roughly 4.3x on knee data. The
estimator below exists only for the test split, which ships a stored mask
and no mask function.

Recording is safe under the DataLoader: each worker process gets its own
copy of the transform (and `worker_init_fn` reseeds its `rng`), and within a
worker `__call__` runs one sample at a time, so the value read back is
always the one just drawn.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from fastmri.data.subsample import (
    EquiSpacedMaskFunc,
    EquispacedMaskFractionFunc,
    MagicMaskFractionFunc,
    MagicMaskFunc,
    MaskFunc,
    RandomMaskFunc,
)
from fastmri.data.transforms import VarNetDataTransform, VarNetSample


class _AccelRecorderMixin:
    """Records the (center_fraction, acceleration) pair each call draws.

    Overrides the single method `MaskFunc.sample_mask` calls to pick a rate.
    Everything else about the mask family is untouched, so masks are
    bit-identical to the baseline's for the same seed.
    """

    last_acceleration: Optional[float] = None
    last_center_fraction: Optional[float] = None

    def choose_acceleration(self):
        center_fraction, acceleration = super().choose_acceleration()
        self.last_acceleration = float(acceleration)
        self.last_center_fraction = float(center_fraction)
        return center_fraction, acceleration


# Module-level subclasses, not classes built inside a function: DataLoader
# workers started with the "spawn" method have to pickle the transform, and a
# locally defined class cannot be pickled.
class RecordingRandomMaskFunc(_AccelRecorderMixin, RandomMaskFunc):
    pass


class RecordingEquiSpacedMaskFunc(_AccelRecorderMixin, EquiSpacedMaskFunc):
    pass


class RecordingEquispacedMaskFractionFunc(_AccelRecorderMixin, EquispacedMaskFractionFunc):
    pass


class RecordingMagicMaskFunc(_AccelRecorderMixin, MagicMaskFunc):
    pass


class RecordingMagicMaskFractionFunc(_AccelRecorderMixin, MagicMaskFractionFunc):
    pass


def create_recording_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """`fastmri.data.subsample.create_mask_for_mask_type` with recording."""
    if mask_type_str == "random":
        return RecordingRandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return RecordingEquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return RecordingEquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "magic":
        return RecordingMagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return RecordingMagicMaskFractionFunc(center_fractions, accelerations)
    else:
        raise ValueError(f"{mask_type_str} not supported")


def estimate_acceleration_from_mask(
    mask: torch.Tensor,
    padding_left: Optional[int] = None,
    padding_right: Optional[int] = None,
) -> float:
    """Fallback R for a sample whose mask was not drawn here (test split).

    Counts sampled columns inside the acquired region only, so the zeroed
    padding `apply_mask` applies does not inflate the estimate. Returns the
    ratio of acquired columns to sampled ones. Approximate by construction;
    the recorded nominal value is used whenever a mask function exists.
    """
    flat = mask.reshape(-1, mask.shape[-2], mask.shape[-1])[0, :, 0] if mask.dim() >= 3 else mask.reshape(-1)
    flat = flat.reshape(-1)
    lo = int(padding_left) if padding_left is not None else 0
    hi = int(padding_right) if padding_right is not None else flat.numel()
    lo = max(0, min(lo, flat.numel()))
    hi = max(lo + 1, min(hi, flat.numel()))
    window = flat[lo:hi]
    sampled = float(window.sum().item())
    if sampled <= 0:
        return 1.0
    return float(window.numel()) / sampled


class DPIVarNetSample(NamedTuple):
    """`fastmri.data.transforms.VarNetSample` plus the acceleration rate.

    The default DataLoader collate turns a batch of these into the same
    NamedTuple with batched fields, so `acceleration` arrives at the module
    as a float tensor of shape (B,).
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    acceleration: float


class DPIVarNetDataTransform(VarNetDataTransform):
    """`VarNetDataTransform` that also reports the acceleration rate.

    The eight baseline fields come from `super().__call__`, unchanged, so a
    DPI sample and a baseline sample built from the same seed carry
    bit-identical k-space, mask and target. Only the ninth field is new.

    R is resolved in this order: the value the recording mask function just
    drew; `attrs["acceleration"]`, which only the test and challenge files
    carry; the estimate from the mask itself.
    """

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> DPIVarNetSample:
        sample = super().__call__(kspace, mask, target, attrs, fname, slice_num)

        acceleration = None
        recorded = getattr(self.mask_func, "last_acceleration", None)
        if self.mask_func is not None and recorded is not None:
            acceleration = float(recorded)
        elif "acceleration" in attrs:
            acceleration = float(attrs["acceleration"])
        else:
            acceleration = estimate_acceleration_from_mask(
                sample.mask, attrs.get("padding_left"), attrs.get("padding_right")
            )

        return DPIVarNetSample(
            masked_kspace=sample.masked_kspace,
            mask=sample.mask,
            num_low_frequencies=sample.num_low_frequencies,
            target=sample.target,
            fname=sample.fname,
            slice_num=sample.slice_num,
            max_value=sample.max_value,
            crop_size=sample.crop_size,
            acceleration=acceleration,
        )
