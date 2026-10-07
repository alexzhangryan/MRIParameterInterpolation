# noise/: the measurement noise already in the training and validation data

Added 2026-10-01 to answer "what is the noise level in training and testing?"
before emailing the supervisor.

**Background.** Nothing in this repo, and nothing in the E2E VarNet paper
(arXiv:2004.06688), adds measurement noise. The paper's eq. 1-2 name ε and
never quantify it. Its only added noise is σ = 0.02 output dithering for
display (supplement 6.1). So the noise in every experiment so far is the
scanner's thermal noise. It is in the input k-space and in the
fully-sampled target, and it is the same in train and val. fastMRI ships no
noise prescan, so this directory **estimates** it from the data, on exactly
the two batches every model uses (`brain_multicoil_{train,val}_batch_0`).

## How it is measured

`measure_noise.py`, per slice, per coil:

1. `x_c = ifft2c(k_c)` in fastMRI's convention (centred, orthonormal), so the
   image is on the scale of `reconstruction_rss` and of the VarNet output. The
   job checks this: `rss_vs_target_relerr` compares RSS(ifft(k)) with the
   stored target.
2. fastMRI raw data keeps the 2x readout oversampling. The target is the
   central half of the readout, so rows with |row − centre|/H in
   **[0.30, 0.42)** are air. The outermost rows (≥ 0.42) are left out because
   the receiver's anti-aliasing filter attenuates noise there;
   `sigma_outer_over_band` shows how much.
3. σ = 1.4826·MAD of the real and imaginary parts there (robust to stray
   signal). σ is **per real/imaginary component**; complex noise variance is
   2σ².
4. k-space σ = image σ · sqrt(H·W / n_acquired). The zero-padded
   phase-encode columns carry no noise.

What comes out, per slice, then as per-volume medians, then as split
summaries (overall and by contrast):

| column | meaning |
|---|---|
| `sigma_img`, `sigma_k` | σ per coil, image units / k-space units (mean over coils; `_min`/`_max` per coil) |
| `sigma_img_rel_target_max` | σ with images normalised so the target max = 1: the scale an "add σ = 0.01" proposal would be in, and the scale of the paper's 0.02 dithering |
| `sigma_k_rel_rms` | noise RMS / k-space RMS, amplitude ratio |
| `snr_db_full` | measurement SNR = 10·log10(signal energy / noise energy) over all acquired samples |
| `snr_db_R{2,4,6,8,10}` | the same over the samples each rate's mask keeps (`equispaced_fraction`, center 0.32/R, the harness's per-volume seed). The noise per sample does not change with R; the SNR rises because masks keep the energetic centre |
| `sigma_img_zf_R*` | noise std in the zero-filled input image at that rate |
| `noise_corr_mean/max` | absolute coil-to-coil noise correlation (0 = independent coils, i.e. prewhitened or naturally decoupled) |
| `target_corner_mean_rel_max` | the background floor of the ground truth / its max: noise the network is trained to reproduce |
| `target_floor_ratio` | that floor ÷ the floor predicted from σ (χ with 2C dof). ≈1 means the corners are air and the estimate is consistent |
| `systemFieldStrength_T`, `systemModel`, `acquisition`, `h5_keys`, `header_noise_tags` | scanner, contrast, and proof of whether any noise data or noise header field exists |

**Tripwires.** If `sigma_img_std_vs_mad` is far above 1 or `target_floor_ratio`
is far from 1, the band is not pure air for those volumes. Look at them before
quoting σ. On the synthetic smoke data (no oversampling, no noise) both fire,
as they should.

## Running it

```
make test          # estimator on synthetic k-space with known σ (numpy only)
make job-smoke     # run_noise.sh in the job image on the synthetic tarball (Docker)

# on ap2001, in ~/MRIParameterInterpolation/noise
make submit-test   # 3 volumes per split, minutes after transfer
make submit        # both splits, one CPU job each: ~100 GB transfer + one xz pass each
make summary       # markdown table, train vs val, for RESULTS.md / the email
```

CPU only, 4 cores, 16 GB, 140 GB disk per job. The tarball is streamed
through `xz | python` and never extracted. Results land in
`runs/<split>/<Cluster>/`.

## Caveats

- An estimate, not a calibration: there is no noise prescan. It assumes the
  oversampled margin is air and the noise is stationary across the image.
- The anti-aliasing filter makes the k-space noise not perfectly white along
  the readout. `sigma_k` is the equivalent white-noise level over the band.
- Per-rate SNR uses the validation masks (seeded by file name). Training
  redraws masks every epoch, which changes these numbers only marginally.
