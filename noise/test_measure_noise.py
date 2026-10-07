"""
Tests for measure_noise.py's estimator on synthetic k-space where the true
noise level is known. numpy only (no h5py / fastmri), so it runs anywhere:

  python3 -m pytest noise/test_measure_noise.py -q      # or
  python3 noise/test_measure_noise.py
"""

import math

import numpy as np

from measure_noise import chi_mean, ifft2c, slice_noise


def fft2c(x):
    ax = (-2, -1)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def phantom(C=8, H=320, W=160, seed=0):
    """Multicoil object confined to the central half of the readout (2x oversampling)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    obj = np.exp(-(((yy - H / 2) / (0.17 * H)) ** 2 + ((xx - W / 2) / (0.35 * W)) ** 2) ** 2)
    obj[np.abs(yy - H / 2) > 0.24 * H] = 0
    coils = []
    for c in range(C):
        cy, cx = H / 2 + 0.2 * H * math.cos(2 * math.pi * c / C), W / 2 + 0.4 * W * math.sin(2 * math.pi * c / C)
        prof = np.exp(-(((yy - cy) / (0.5 * H)) ** 2 + ((xx - cx) / (0.6 * W)) ** 2))
        coils.append(100.0 * obj * prof * np.exp(1j * rng.uniform(-np.pi, np.pi)))
    return np.stack(coils), rng


def noisy_kspace(sigma_k, pad=10, C=8, H=320, W=160, seed=0):
    x, rng = phantom(C, H, W, seed)
    k = fft2c(x)
    k = k + sigma_k * (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape))
    k[:, :, :pad] = 0  # fastMRI zero-pads the phase-encode edges
    k[:, :, W - pad:] = 0
    return k.astype(np.complex64), x


def test_recovers_sigma_k_with_zero_padding():
    for sigma in (0.05, 0.5, 2.0):
        k, _ = noisy_kspace(sigma)
        st = slice_noise(k)
        assert abs(st["sigma_k"] / sigma - 1) < 0.03, (sigma, st["sigma_k"])
        assert st["n_acq_cols"] == 140 and st["n_acq_rows"] == 320


def test_image_sigma_scaled_by_acquired_fraction():
    sigma, W, pad = 1.0, 160, 20
    k, _ = noisy_kspace(sigma, pad=pad, W=W)
    st = slice_noise(k)
    expected = sigma * math.sqrt((W - 2 * pad) / W)  # ortho ifft over the PE axis
    assert abs(st["sigma_img"] / expected - 1) < 0.03


def test_snr_matches_true_signal_energy():
    sigma = 0.5
    k, x = noisy_kspace(sigma, pad=0)
    clean = fft2c(x)
    true_db = 10 * math.log10(np.sum(np.abs(clean) ** 2) / (clean.size * 2 * sigma ** 2))
    st = slice_noise(k)
    assert abs(st["snr_db_full"] - true_db) < 0.3, (st["snr_db_full"], true_db)


def test_mask_snr_and_zero_filled_sigma():
    sigma, W = 1.0, 160
    k, _ = noisy_kspace(sigma, pad=0, W=W)
    m = np.zeros(W, bool)
    m[::4] = True
    m[W // 2 - 6: W // 2 + 6] = True
    st = slice_noise(k, masks={4: m})
    assert abs(st["sigma_img_zf_R4"] / (sigma * math.sqrt(m.sum() / W)) - 1) < 0.03
    assert st["snr_db_R4"] > st["snr_db_full"]  # masks keep the energetic centre


def test_uncorrelated_coils_and_target_floor():
    sigma = 1.0
    k, _ = noisy_kspace(sigma, pad=0)
    st0 = slice_noise(k)
    assert st0["noise_corr_mean"] < 0.05
    # a target built the fastMRI way: RSS of the full image, central half of the readout
    x = ifft2c(k)
    rss = np.sqrt(np.sum(np.abs(x) ** 2, axis=0))
    H, W = rss.shape
    tgt = rss[H // 4: 3 * H // 4, (W - 128) // 2:(W - 128) // 2 + 128]
    st = slice_noise(k, target=tgt, target_max=float(tgt.max()))
    assert st["rss_vs_target_relerr"] < 1e-5
    assert abs(st["target_floor_ratio"] - 1) < 0.08, st["target_floor_ratio"]
    assert 0 < st["sigma_img_rel_target_max"] < 1


def test_correlated_coils_detected():
    C, H, W = 4, 320, 160
    x, rng = phantom(C, H, W)
    n = rng.standard_normal((C, H, W)) + 1j * rng.standard_normal((C, H, W))
    mix = np.eye(C) + 0.6  # strong common component
    n = np.tensordot(mix, n, axes=1)
    st = slice_noise((fft2c(x) + fft2c(n)).astype(np.complex64))
    assert st["noise_corr_mean"] > 0.3


def test_chi_mean():
    rng = np.random.default_rng(1)
    for dof in (2, 16, 40):
        emp = np.sqrt((rng.standard_normal((200000, dof)) ** 2).sum(1)).mean()
        assert abs(emp / chi_mean(dof) - 1) < 0.005


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
