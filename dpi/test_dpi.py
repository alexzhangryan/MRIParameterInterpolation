"""
test_dpi.py: the evidence that the DPI mechanics are right.

There is no published DPI-on-MRI result to check against, so these tests are
what stands between the implementation and a wrong multi-day run. They check
the properties the paper's equations imply, on a small model:

  1. at initialisation the DPI network equals the baseline VarNet, for every
     acceleration (both parameter sets are identical copies, so eq. (2)
     collapses to f(theta, x) whatever lambda is);
  2. lambda is monotone, starts at 0, ends at 1, and puts the rates where
     the log spacing says;
  3. every baseline state-dict key exists verbatim, and only `*_copy` and
     `lambda_table.phi` are new;
  4. the parameter count is 2x base + S;
  5. gradients: phi has exactly zero gradient while the copies are equal
     (inherent to DPI, not a bug) and a real one once they differ;
  6. once the copies differ, different accelerations give different outputs,
     which is the whole point;
  7. a mixed-acceleration batch equals the concatenation of single-sample
     calls;
  8. the recording transform reports the rate the mask function actually
     drew, and its other eight fields are bit-identical to the baseline
     transform's.

Run inside the training image:
    docker run --rm -v "$PWD/..":/work -w /work/dpi fastmri-train:local \
        python -m pytest test_dpi.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastmri.data import transforms as fastmri_transforms
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.models import VarNet

from dpi_transforms import (
    DPIVarNetDataTransform,
    create_recording_mask_for_mask_type,
)
from dpi_varnet import DPIVarNet, LambdaTable, dpi_param_summary, load_baseline_state_dict

SMALL = dict(num_cascades=2, sens_chans=4, sens_pools=2, chans=4, pools=2)
RATES = [2.0, 4.0, 6.0, 8.0]


def create_input(shape):
    x = np.arange(np.prod(shape)).reshape(shape)
    return torch.from_numpy(x).float()


def masked_input(shape=(1, 3, 32, 16, 2), center_fractions=(0.08,), accelerations=(4,)):
    """Masked k-space with one mask per sample, as fastMRI's own tests build it.

    Masking a whole batch in one call instead returns a single mask with
    batch dim 1, which broadcasts; both shapes occur, and
    test_mixed_batch_with_a_shared_mask covers the other one.
    """
    mask_func = create_mask_for_mask_type(
        "equispaced_fraction", list(center_fractions), list(accelerations)
    )
    x = create_input(shape)
    outputs, masks = [], []
    num_low_frequencies = None
    for i in range(x.shape[0]):
        output, mask, num_low_frequencies = fastmri_transforms.apply_mask(
            x[i : i + 1], mask_func, seed=123
        )
        outputs.append(output)
        masks.append(mask)
    return torch.cat(outputs), torch.cat(masks).byte(), num_low_frequencies


def accel(value, n=1):
    return torch.full((n,), float(value))


# ---------------------------------------------------------------- 1, 3, 4 --
def test_equals_baseline_at_init_for_every_rate():
    """Eq. (2) with theta0 == theta1 collapses to the baseline, for any lambda.

    Compared relatively, not absolutely: `lam*w + (1-lam)*w` is only
    bit-exact for lam of exactly 0 or 1, so at intermediate rates the two
    networks agree to float32 round-off (~1e-7 relative), not exactly. The
    next test pins the exact endpoints.
    """
    torch.manual_seed(0)
    baseline = VarNet(**SMALL)
    dpi = DPIVarNet(**SMALL)
    info = load_baseline_state_dict(dpi, baseline.state_dict(), strip_prefix="")
    assert info["copies_initialised"]

    kspace, mask, nlf = masked_input()
    with torch.no_grad():
        want = baseline(kspace, mask, nlf)
        scale = float(want.abs().max())
        for rate in RATES:
            got = dpi(kspace, mask, nlf, accel(rate))
            rel = float((want - got).abs().max()) / scale
            assert rel < 1e-5, f"R={rate}: relative difference {rel:.2e}"


def test_endpoints_are_bit_exact_against_the_baseline():
    """At lambda = 0 and lambda = 1 the interpolation is a no-op, so the two
    networks must agree exactly, not just to round-off."""
    torch.manual_seed(0)
    baseline = VarNet(**SMALL)
    dpi = DPIVarNet(**SMALL)
    load_baseline_state_dict(dpi, baseline.state_dict(), strip_prefix="")

    kspace, mask, nlf = masked_input()
    with torch.no_grad():
        want = baseline(kspace, mask, nlf)
        for rate in (2.0, 8.0):  # accel_min -> lambda 0, accel_max -> lambda 1
            assert torch.equal(want, dpi(kspace, mask, nlf, accel(rate))), f"R={rate}"


def test_warm_start_from_a_lightning_checkpoint():
    """The --init_from_baseline path: a VarNetModule checkpoint also carries
    the SSIM loss buffer, which is not part of the network and must not be
    mistaken for an unexpected key."""
    from fastmri.pl_modules import VarNetModule

    torch.manual_seed(1)
    baseline = VarNetModule(num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2)
    ckpt = {"state_dict": baseline.state_dict(), "epoch": 7}
    assert any(k == "loss.w" for k in ckpt["state_dict"]), "expected the SSIM buffer"

    dpi = DPIVarNet(**SMALL, lambda_length=64)
    load_baseline_state_dict(dpi, ckpt, strip_prefix="varnet.")

    loaded = dict(dpi.named_parameters())
    for name, param in baseline.varnet.named_parameters():
        assert torch.equal(param, loaded[name]), name
        assert torch.equal(param, loaded[name + "_copy"]), name + "_copy"


def test_warm_start_refuses_a_mismatched_architecture():
    from fastmri.pl_modules import VarNetModule

    baseline = VarNetModule(num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2)
    ckpt = {"state_dict": baseline.state_dict()}
    wrong = DPIVarNet(num_cascades=3, sens_chans=4, sens_pools=2, chans=4, pools=2)
    with pytest.raises(RuntimeError, match="does not fill"):
        load_baseline_state_dict(wrong, ckpt, strip_prefix="varnet.")


def test_state_dict_keys_are_a_superset_of_the_baseline():
    baseline_keys = set(VarNet(**SMALL).state_dict().keys())
    dpi_keys = set(DPIVarNet(**SMALL).state_dict().keys())

    assert baseline_keys <= dpi_keys, sorted(baseline_keys - dpi_keys)[:8]
    new = dpi_keys - baseline_keys
    unexplained = [k for k in new if not (k.endswith("_copy") or k == "lambda_table.phi")]
    assert not unexplained, unexplained[:8]
    assert "lambda_table.phi" in new


def test_parameter_count_is_twice_the_baseline_plus_phi():
    length = 64
    baseline_params = sum(p.numel() for p in VarNet(**SMALL).parameters())
    dpi = DPIVarNet(**SMALL, lambda_length=length)
    summary = dpi_param_summary(dpi)

    assert summary["base"] == baseline_params
    assert summary["copies"] == baseline_params  # every learnable tensor duplicated
    assert summary["phi"] == length
    assert summary["total"] == 2 * baseline_params + length


def test_no_dpi_sens_leaves_the_sensitivity_net_single():
    full = dpi_param_summary(DPIVarNet(**SMALL))
    ablated = dpi_param_summary(DPIVarNet(**SMALL, dpi_sens=False))

    sens_params = sum(p.numel() for p in VarNet(**SMALL).sens_net.parameters())
    assert full["copies"] - ablated["copies"] == sens_params
    assert ablated["base"] == full["base"]


# ------------------------------------------------------------------- 2 -----
def test_lambda_is_monotone_from_zero_to_one():
    torch.manual_seed(0)
    table = LambdaTable(length=128, accel_min=2.0, accel_max=8.0)
    lam = table.table()

    assert lam.shape == (128,)
    assert torch.all(lam[1:] >= lam[:-1] - 1e-12), "lambda must be non-decreasing"
    assert float(lam[0]) == 0.0
    # float32 softmax sums to 1 - 1e-7, so allclose rather than ==
    assert torch.allclose(lam[-1], torch.tensor(1.0), atol=1e-5)


def test_log_spacing_places_the_rates():
    table = LambdaTable(length=1001, accel_min=2.0, accel_max=8.0, spacing="log")
    idx = table.index(torch.tensor(RATES))

    assert idx.tolist() == [0, 500, 792, 1000]  # log2(R/2)/log2(4)
    # out of range clamps rather than indexing off the end
    assert int(table.index(torch.tensor([1.0]))[0]) == 0
    assert int(table.index(torch.tensor([64.0]))[0]) == 1000


def test_lambda_endpoints_are_the_two_parameter_sets():
    """Paper eq. (2): lambda(s_min) = 0 -> theta0 = weight_copy exactly."""
    torch.manual_seed(0)
    dpi = DPIVarNet(**SMALL, lambda_length=64)
    with torch.no_grad():
        for p in dpi.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # make the two sets differ
        lam_min = dpi.lambda_table(accel(2.0))[0]
        lam_max = dpi.lambda_table(accel(8.0))[0]
    assert float(lam_min) == pytest.approx(0.0, abs=1e-6)
    assert float(lam_max) == pytest.approx(1.0, abs=1e-4)


# ------------------------------------------------------------------- 5 -----
def test_phi_gradient_is_zero_while_the_copies_are_equal_and_nonzero_after():
    torch.manual_seed(0)
    dpi = DPIVarNet(**SMALL, lambda_length=64)
    kspace, mask, nlf = masked_input()

    dpi.zero_grad()
    dpi(kspace, mask, nlf, accel(4.0)).sum().backward()
    phi_grad = dpi.lambda_table.phi.grad
    assert phi_grad is not None
    # identical parameter sets => dL/dlambda = <g, theta1 - theta0> = 0.
    # This is inherent to DPI at initialisation, not a wiring bug; the sets
    # separate on the first optimiser step because theta1 receives lambda*g
    # and theta0 receives (1-lambda)*g.
    assert torch.allclose(phi_grad, torch.zeros_like(phi_grad), atol=1e-9)

    with torch.no_grad():
        for name, p in dpi.named_parameters():
            if name.endswith("_copy"):
                p.add_(torch.randn_like(p) * 0.1)

    dpi.zero_grad()
    dpi(kspace, mask, nlf, accel(4.0)).sum().backward()
    assert dpi.lambda_table.phi.grad.abs().sum() > 0


def test_both_parameter_sets_receive_gradient():
    """Both ends of the interpolation must train, at lambda in (0, 1).

    Note cascades.0.dc_weight is excluded: the first cascade starts from
    kspace_pred == masked_kspace, so its soft data-consistency term is
    identically zero and that one parameter gets no gradient. That is the
    baseline VarNet's behaviour too (measured, not assumed), not a DPI
    artefact, so the check uses cascade 1.
    """
    torch.manual_seed(0)
    dpi = DPIVarNet(**SMALL, lambda_length=64)
    kspace, mask, nlf = masked_input()

    dpi.zero_grad()
    dpi(kspace, mask, nlf, accel(4.0)).sum().backward()

    params = dict(dpi.named_parameters())
    for name in (
        "cascades.1.dc_weight",
        "cascades.1.model.unet.down_sample_layers.0.layers.0.weight",
        "sens_net.norm_unet.unet.down_sample_layers.0.layers.0.weight",
    ):
        for key in (name, name + "_copy"):
            grad = params[key].grad
            assert grad is not None, key
            assert float(grad.abs().sum()) > 0, key


# ------------------------------------------------------------------- 6 -----
def test_different_rates_give_different_outputs_once_the_sets_differ():
    torch.manual_seed(0)
    dpi = DPIVarNet(**SMALL, lambda_length=64)
    with torch.no_grad():
        for name, p in dpi.named_parameters():
            if name.endswith("_copy"):
                p.add_(torch.randn_like(p) * 0.1)

    kspace, mask, nlf = masked_input()
    with torch.no_grad():
        out2 = dpi(kspace, mask, nlf, accel(2.0))
        out8 = dpi(kspace, mask, nlf, accel(8.0))
    assert not torch.allclose(out2, out8, atol=1e-6)


def test_forward_without_acceleration_is_an_error():
    dpi = DPIVarNet(**SMALL, lambda_length=64)
    kspace, mask, nlf = masked_input()
    with pytest.raises(ValueError, match="acceleration"):
        dpi(kspace, mask, nlf)


def test_mixed_batch_is_rejected_by_the_model():
    dpi = DPIVarNet(**SMALL, lambda_length=64)
    kspace, mask, nlf = masked_input(shape=(2, 3, 32, 16, 2))
    with pytest.raises(ValueError, match="one acceleration per forward"):
        dpi(kspace, mask, nlf, torch.tensor([4.0, 8.0]))


# ------------------------------------------------------------------- 7 -----
def test_module_splits_a_mixed_batch_into_single_sample_calls():
    from dpi_module import DPIVarNetModule

    torch.manual_seed(0)
    module = DPIVarNetModule(
        num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2, lambda_length=64
    )
    module.eval()
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("_copy"):
                p.add_(torch.randn_like(p) * 0.1)

    kspace, mask, nlf = masked_input(shape=(2, 3, 32, 16, 2))
    rates = torch.tensor([4.0, 8.0])
    with torch.no_grad():
        mixed = module(kspace, mask, nlf, rates)
        one = module(kspace[0:1], mask[0:1], nlf, rates[0:1])
        two = module(kspace[1:2], mask[1:2], nlf, rates[1:2])

    assert torch.allclose(mixed, torch.cat([one, two]), atol=1e-6)


def test_mixed_batch_with_a_shared_mask():
    """apply_mask on a whole batch returns one mask with batch dim 1."""
    from dpi_module import DPIVarNetModule

    torch.manual_seed(0)
    module = DPIVarNetModule(
        num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2, lambda_length=64
    )
    module.eval()

    mask_func = create_mask_for_mask_type("equispaced_fraction", [0.08], [4])
    kspace, mask, nlf = fastmri_transforms.apply_mask(
        create_input((2, 3, 32, 16, 2)), mask_func, seed=123
    )
    assert mask.shape[0] == 1, "this test is about the broadcast-mask shape"

    with torch.no_grad():
        out = module(kspace, mask.byte(), nlf, torch.tensor([4.0, 8.0]))
    assert out.shape[0] == 2


def test_optimizer_puts_phi_in_its_own_group():
    from dpi_module import DPIVarNetModule

    module = DPIVarNetModule(
        num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2,
        lambda_length=64, lr=3e-4, lambda_lr=1e-3,
    )
    (optim,), _ = module.configure_optimizers()

    assert len(optim.param_groups) == 2
    backbone, lam = optim.param_groups
    assert backbone["lr"] == pytest.approx(3e-4)
    assert lam["lr"] == pytest.approx(1e-3)
    assert len(lam["params"]) == 1
    assert lam["params"][0].numel() == 64
    n_all = sum(p.numel() for p in module.parameters())
    n_groups = sum(p.numel() for g in optim.param_groups for p in g["params"])
    assert n_all == n_groups, "every parameter must be in exactly one group"


# ------------------------------------------------------------------- 8 -----
@pytest.mark.parametrize("rates", [[4], [2, 4, 6, 8]])
def test_transform_records_the_rate_the_mask_function_drew(rates):
    fractions = {2: 0.16, 4: 0.08, 6: 0.0533, 8: 0.04}
    center_fractions = [fractions[r] for r in rates]

    dpi_mask = create_recording_mask_for_mask_type(
        "equispaced_fraction", center_fractions, rates
    )
    base_mask = create_mask_for_mask_type(
        "equispaced_fraction", center_fractions, rates
    )
    dpi_transform = DPIVarNetDataTransform(mask_func=dpi_mask)
    base_transform = VarNetDataTransform(mask_func=base_mask)

    kspace = create_input((3, 32, 16)).numpy() + 1j * create_input((3, 32, 16)).numpy()
    target = create_input((16, 16)).numpy()
    attrs = {"max": 1.0, "padding_left": 0, "padding_right": 16,
             "recon_size": (16, 16, 1)}

    for fname in ("file1000.h5", "file1001.h5", "file1002.h5"):
        dpi_sample = dpi_transform(kspace, None, target, attrs, fname, 0)
        base_sample = base_transform(kspace, None, target, attrs, fname, 0)

        assert dpi_sample.acceleration in [float(r) for r in rates]
        assert dpi_sample.acceleration == float(dpi_mask.last_acceleration)
        # the eight baseline fields must be untouched
        assert torch.equal(dpi_sample.masked_kspace, base_sample.masked_kspace)
        assert torch.equal(dpi_sample.mask, base_sample.mask)
        assert dpi_sample.num_low_frequencies == base_sample.num_low_frequencies
        assert torch.equal(dpi_sample.target, base_sample.target)


def test_seeded_validation_transform_locks_a_volume_to_one_rate():
    mask = create_recording_mask_for_mask_type(
        "equispaced_fraction", [0.16, 0.08, 0.0533, 0.04], [2, 4, 6, 8]
    )
    transform = DPIVarNetDataTransform(mask_func=mask)  # use_seed=True

    kspace = create_input((3, 32, 16)).numpy() + 1j * create_input((3, 32, 16)).numpy()
    target = create_input((16, 16)).numpy()
    attrs = {"max": 1.0, "padding_left": 0, "padding_right": 16,
             "recon_size": (16, 16, 1)}

    rates = {
        transform(kspace, None, target, attrs, "file1000.h5", s).acceleration
        for s in range(5)
    }
    assert len(rates) == 1, f"one volume drew several rates: {rates}"


# ------------------------------------------------------------- dpi_scope ---
# The narrower scopes. Each is a variant of the grid in plan.md ("DPI variant
# grid"); the parameter counts below come from that document's table, derived
# by hand from the architecture, so these tests are a real oracle rather than
# a recording of whatever the code happens to do.
PAPER_CONFIG = dict(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4)

# (extra per cascade, extra in the sensitivity U-Net), plan.md section 2.
SCOPE_EXTRA = {
    "full": (2454339, 484898),
    "shallow": (77475, 15394),
    "io": (3279, 738),
    "dc": (1, 0),
    "none": (0, 0),
}


@pytest.mark.parametrize("scope", ["full", "shallow", "io", "dc", "none"])
def test_parameter_counts_per_scope(scope):
    """Every scope duplicates exactly the tensors plan.md says it does.

    Checked on the real 12-cascade configuration, because the overheads are
    what justify the variants: if `io` is not actually 0.14%, the result it is
    meant to support is not the result it reports.
    """
    per_cascade, in_sens = SCOPE_EXTRA[scope]
    length = 1000
    model = DPIVarNet(**PAPER_CONFIG, dpi_scope=scope, lambda_length=length)
    summary = dpi_param_summary(model)

    baseline_params = sum(p.numel() for p in VarNet(**PAPER_CONFIG).parameters())
    assert summary["base"] == baseline_params

    assert summary["copies"] == 12 * per_cascade + in_sens
    # phi is a parameter only when there is something for lambda to
    # interpolate; the "none" control must match the baseline exactly.
    assert summary["phi"] == (0 if scope == "none" else length)
    if scope == "none":
        assert summary["total"] == baseline_params


@pytest.mark.parametrize("scope", ["full", "shallow", "io", "dc", "none"])
def test_scope_equals_baseline_at_init_for_every_rate(scope):
    """Test 1 (equivalence at initialisation) holds for every scope.

    A scope only decides which tensors have a copy; while each copy equals
    its base tensor, eq. (2) collapses to the baseline whatever lambda is.
    """
    torch.manual_seed(0)
    baseline = VarNet(**SMALL)
    dpi = DPIVarNet(**SMALL, dpi_scope=scope, lambda_length=64)
    load_baseline_state_dict(dpi, baseline.state_dict(), strip_prefix="")

    kspace, mask, nlf = masked_input()
    with torch.no_grad():
        want = baseline(kspace, mask, nlf)
        scale = float(want.abs().max())
        for rate in RATES:
            got = dpi(kspace, mask, nlf, accel(rate))
            rel = float((want - got).abs().max()) / scale
            assert rel < 1e-5, f"scope={scope} R={rate}: relative difference {rel:.2e}"


@pytest.mark.parametrize("scope", ["full", "shallow", "io", "dc", "none"])
def test_scope_state_dict_stays_baseline_compatible(scope):
    """Narrowing the scope must not rename or drop a baseline tensor.

    This is what keeps one verification harness and one warm-start path
    working for every variant: the baseline keys are always present verbatim,
    and the only additions are the copies this scope created plus phi.
    """
    baseline_keys = set(VarNet(**SMALL).state_dict().keys())
    dpi_keys = set(
        DPIVarNet(**SMALL, dpi_scope=scope, lambda_length=64).state_dict().keys()
    )

    assert baseline_keys <= dpi_keys, sorted(baseline_keys - dpi_keys)[:8]
    new = dpi_keys - baseline_keys
    unexplained = [k for k in new if not (k.endswith("_copy") or k == "lambda_table.phi")]
    assert not unexplained, unexplained[:8]
    # phi stays in the state dict even as a buffer under "none": it is how
    # verify_varnet.is_dpi_state recognises a DPI checkpoint at all.
    assert "lambda_table.phi" in new


def test_io_scope_duplicates_exactly_the_intended_layers():
    """The `io` scope is a specific claim about which layers matter, so name them.

    First ConvBlock of each U-Net, the final 1x1 projection, and dc_weight --
    and nothing else. The head ConvBlock sits behind that projection and is
    deliberately NOT duplicated, which is the one case where a scope splits an
    nn.Sequential the baseline treats as one unit.
    """
    model = DPIVarNet(**SMALL, dpi_scope="io", lambda_length=64)
    copies = sorted(n for n, _ in model.named_parameters() if n.endswith("_copy"))

    cascade0 = [n for n in copies if n.startswith("cascades.0.")]
    assert cascade0 == [
        "cascades.0.dc_weight_copy",
        "cascades.0.model.unet.down_sample_layers.0.layers.0.weight_copy",
        "cascades.0.model.unet.down_sample_layers.0.layers.4.weight_copy",
        "cascades.0.model.unet.up_conv.1.1.bias_copy",
        "cascades.0.model.unet.up_conv.1.1.weight_copy",
    ]
    # up_conv.{n-1}.0 is the head ConvBlock: present in the model, no copy.
    assert "cascades.0.model.unet.up_conv.1.0.layers.0.weight_copy" not in copies
    assert any(n.startswith("sens_net.") for n in copies), "sens net follows the scope"


def test_dc_scope_duplicates_only_the_step_sizes():
    model = DPIVarNet(**SMALL, dpi_scope="dc", lambda_length=64)
    copies = sorted(n for n, _ in model.named_parameters() if n.endswith("_copy"))
    assert copies == ["cascades.0.dc_weight_copy", "cascades.1.dc_weight_copy"]


def test_none_scope_is_the_baseline_network():
    """The control: identical to VarNet in parameters, and rate-blind.

    phi becomes a buffer, so there is no parameter without a gradient (which
    DDP rejects) and the parameter count matches the baseline exactly. lambda
    is still computed; it simply multiplies nothing.
    """
    model = DPIVarNet(**SMALL, dpi_scope="none", lambda_length=64)
    assert not [n for n, _ in model.named_parameters() if n.endswith("_copy")]
    assert "lambda_table.phi" not in dict(model.named_parameters())
    assert "lambda_table.phi" in dict(model.named_buffers())

    baseline_params = sum(p.numel() for p in VarNet(**SMALL).parameters())
    assert sum(p.numel() for p in model.parameters()) == baseline_params

    # rate-blind by construction: no copy anywhere, so lambda cannot change
    # the output even once training has moved the weights.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    kspace, mask, nlf = masked_input()
    with torch.no_grad():
        assert torch.equal(
            model(kspace, mask, nlf, accel(2.0)), model(kspace, mask, nlf, accel(8.0))
        )


@pytest.mark.parametrize("scope", ["full", "shallow", "io", "dc"])
def test_every_copy_a_scope_creates_receives_gradient(scope):
    """No scope may create a parameter that never trains.

    cascades.0.dc_weight is the documented exception (the first cascade data
    consistency term is identically zero, in the baseline too), so that one
    name is skipped.
    """
    torch.manual_seed(0)
    dpi = DPIVarNet(**SMALL, dpi_scope=scope, lambda_length=64)
    # Separate the two sets first: at init the copies are equal, which is the
    # one situation where phi legitimately has a zero gradient.
    with torch.no_grad():
        for name, p in dpi.named_parameters():
            if name.endswith("_copy"):
                p.add_(torch.randn_like(p) * 0.1)

    kspace, mask, nlf = masked_input()
    dpi.zero_grad()
    dpi(kspace, mask, nlf, accel(4.0)).sum().backward()

    assert float(dpi.lambda_table.phi.grad.abs().sum()) > 0, "phi must train"
    for name, p in dpi.named_parameters():
        if not name.endswith("_copy") or name == "cascades.0.dc_weight_copy":
            continue
        assert p.grad is not None, name
        assert float(p.grad.abs().sum()) > 0, name


@pytest.mark.parametrize("scope", ["shallow", "io", "dc", "none"])
def test_narrow_scopes_are_cheaper_than_full(scope):
    """Sanity ordering, so a mis-specified flag table cannot pass unnoticed."""
    full = dpi_param_summary(DPIVarNet(**SMALL, lambda_length=64))["copies"]
    narrow = dpi_param_summary(
        DPIVarNet(**SMALL, dpi_scope=scope, lambda_length=64)
    )["copies"]
    assert narrow < full


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError, match="dpi_scope"):
        DPIVarNet(**SMALL, dpi_scope="everything", lambda_length=64)


def test_scope_survives_a_checkpoint_round_trip():
    """A scoped checkpoint must rebuild as the same network.

    This is the contract verification/verify_varnet.py depends on: it reads
    dpi_scope out of hyper_parameters (defaulting to "full" for checkpoints
    written before the flag existed) and rebuilds DPIVarNet from it, then
    loads the state dict strictly.
    """
    torch.manual_seed(0)
    original = DPIVarNet(**SMALL, dpi_scope="io", lambda_length=64)
    state = original.state_dict()

    rebuilt = DPIVarNet(**SMALL, dpi_scope="io", lambda_length=64)
    rebuilt.load_state_dict(state, strict=True)

    # the wrong scope must fail loudly rather than silently drop the copies
    with pytest.raises(RuntimeError):
        DPIVarNet(**SMALL, dpi_scope="full", lambda_length=64).load_state_dict(
            state, strict=True
        )


# ------------------------------------------- fixed lambda (2026-10-01 meeting) --
FIXED = dict(lambda_mode="fixed", lambda_spacing="linear", accel_min=2.0, accel_max=10.0)


def test_fixed_lambda_is_exactly_the_linear_position():
    """lambda_mode="fixed" over 2..10: lambda(2, 6, 10) = 0, 0.5, 1 exactly, no table rounding."""
    model = DPIVarNet(**SMALL, lambda_length=1000, **FIXED)
    lam = model.lambda_table(torch.tensor([2.0, 6.0, 10.0, 4.0, 12.0, 1.0]))
    assert lam.tolist() == [0.0, 0.5, 1.0, 0.25, 1.0, 0.0]
    # the learned table at the same spacing rounds to its grid: the reason fixed bypasses it
    learned = LambdaTable(length=1000, accel_min=2.0, accel_max=10.0, spacing="linear")
    with torch.no_grad():
        learned.phi.zero_()  # uniform softmax: the table is i / (S - 1)
    assert learned(torch.tensor([6.0])).item() != 0.5


def test_fixed_lambda_learns_nothing():
    """No phi parameter, no lambda optimiser group, and the count is full DPI minus phi."""
    from dpi_module import DPIVarNetModule

    fixed = DPIVarNet(**SMALL, lambda_length=64, **FIXED)
    learned = DPIVarNet(**SMALL, lambda_length=64)
    assert "phi" not in dict(fixed.lambda_table.named_parameters())
    assert "lambda_table.phi" in fixed.state_dict(), "the verifier detects DPI by this key"
    assert dpi_param_summary(fixed)["phi"] == 0
    assert dpi_param_summary(fixed)["total"] == dpi_param_summary(learned)["total"] - 64

    module = DPIVarNetModule(
        num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2,
        lambda_length=64, lr=3e-4, lambda_lr=1e-3, log_accelerations=(2, 6, 10), **FIXED,
    )
    (optim,), _ = module.configure_optimizers()
    assert len(optim.param_groups) == 1
    assert module.hparams["lambda_mode"] == "fixed"


def test_fixed_lambda_interpolates_the_two_sets():
    """At R6 (lambda 0.5) the forward equals the network built from the averaged parameters."""
    torch.manual_seed(0)
    model = DPIVarNet(**SMALL, lambda_length=64, **FIXED).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("_copy"):
                p.add_(torch.randn_like(p) * 0.1)
    mk, mask, nlf = masked_input(accelerations=(6,), center_fractions=(0.0533,))
    with torch.no_grad():
        out6 = model(mk, mask, nlf, accel(6))
        out2 = model(mk, mask, nlf, accel(2))
        out10 = model(mk, mask, nlf, accel(10))
    assert not torch.allclose(out6, out2) and not torch.allclose(out6, out10)
    # rebuild with every pair averaged and lambda pinned to an endpoint: same output
    avg = DPIVarNet(**SMALL, lambda_length=64, **FIXED).eval()
    state = model.state_dict()
    for k in list(state):
        if k.endswith("_copy"):
            base = k[: -len("_copy")]
            mean = 0.5 * (state[base] + state[k])
            state[base], state[k] = mean.clone(), mean.clone()
    avg.load_state_dict(state)
    with torch.no_grad():
        assert torch.allclose(avg(mk, mask, nlf, accel(2)), out6, atol=1e-5)


def test_fixed_lambda_survives_a_checkpoint_round_trip_through_the_module():
    from dpi_module import DPIVarNetModule

    module = DPIVarNetModule(num_cascades=2, chans=4, sens_chans=4, pools=2, sens_pools=2,
                             lambda_length=64, **FIXED)
    hp = dict(module.hparams)
    rebuilt = DPIVarNet(**SMALL, lambda_length=hp["lambda_length"], accel_min=hp["accel_min"],
                        accel_max=hp["accel_max"], lambda_spacing=hp["lambda_spacing"],
                        lambda_mode=hp["lambda_mode"])
    rebuilt.load_state_dict(module.varnet.state_dict(), strict=True)
    assert rebuilt.lambda_table(torch.tensor([6.0])).item() == 0.5


def test_unknown_lambda_mode_is_rejected():
    with pytest.raises(ValueError):
        DPIVarNet(**SMALL, lambda_mode="frozen")
