#!/usr/bin/env python
"""
measure_noise.py: measure the measurement noise already present in fastMRI
multicoil k-space, per slice, for one split.

Neither the E2E VarNet paper (arXiv:2004.06688, eq. 1-2 name epsilon and never
quantify it) nor our training code adds noise, so the noise in every
experiment so far is whatever the scanner recorded. fastMRI ships no noise
prescan, so it has to be estimated from the data.

Estimator (per coil, per slice):
  x_c = ifft2c(k_c)  with fastMRI's convention (centred, norm="ortho"), so the
  image is on the same scale as `reconstruction_rss` and the VarNet output.
  fastMRI brain/knee raw data keeps the 2x readout oversampling: the image is
  H rows tall (640 for most brain), the target is the central H/2 rows. The
  rows outside that crop are air. The outermost rows are skipped as well,
  because the receiver's anti-aliasing filter attenuates noise there.
  sigma_img_c = robust std (1.4826 * MAD) of the real and imaginary parts of
  x_c in that band. sigma is per real/imaginary component; complex noise
  variance is 2 sigma^2.
  Converted to k-space units: sigma_k_c = sigma_img_c * sqrt(H*W / n_acq),
  where n_acq counts the k-space samples that are not zero padding (fastMRI
  zero-pads the phase-encode edges; with an ortho FFT only acquired samples
  carry noise into the image).

Reported per slice (see COLUMNS): sigma in image, k-space and
target-normalised units; measurement SNR in dB for the full k-space and for
each acceleration's mask; coil noise correlation; the noise floor in the
target's background corners, with the floor predicted from sigma (chi
distribution with 2C degrees of freedom) as a consistency check; and a check
that RSS(ifft2c(k)) reproduces reconstruction_rss, which validates the scale.

  # on CHTC (run_noise.sh does this): stream the tarball, never extract it
  xz -dc -T0 brain_multicoil_val_batch_0.tar.xz | python measure_noise.py --split val --tar_stdin --out output
  # locally, on a directory of .h5 files
  python measure_noise.py --split val --data_path some/multicoil_val --out output
  # merge train + val outputs into the markdown tables for RESULTS.md / the email
  python measure_noise.py --summarize runs/*/output
"""

import argparse
import csv
import json
import math
import os
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_ACCELS = (2, 4, 6, 8, 10)
DEFAULT_CFS = (0.16, 0.08, 0.0533, 0.04, 0.032)  # 0.32 / R, as in training


# ---------------------------------------------------------------------------
# core estimator (numpy only, unit-tested in test_measure_noise.py)
# ---------------------------------------------------------------------------
def ifft2c(k: np.ndarray) -> np.ndarray:
    """fastmri.ifft2c on a complex array: centred, orthonormal, last two axes."""
    ax = (-2, -1)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=ax), axes=ax, norm="ortho"), axes=ax)


def robust_sigma(v: np.ndarray) -> float:
    """1.4826 * MAD: the std of a Gaussian, insensitive to a minority of outliers."""
    v = v.ravel()
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def band_rows(H: int, inner: float, outer: float) -> np.ndarray:
    """Row indices with |row - centre| / H in [inner, outer): both margins."""
    d = np.abs(np.arange(H) + 0.5 - H / 2) / H
    return np.nonzero((d >= inner) & (d < outer))[0]


def acquired_extent(k: np.ndarray):
    """(rows, cols) that hold any non-zero sample across coils: excludes zero padding."""
    nz = np.abs(k).max(axis=0) > 0
    rows = np.nonzero(nz.any(axis=1))[0]
    cols = np.nonzero(nz.any(axis=0))[0]
    return rows, cols


def chi_mean(dof: int) -> float:
    """E[chi_dof] = sqrt(2) Gamma((dof+1)/2) / Gamma(dof/2)."""
    return math.sqrt(2.0) * math.exp(math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2))


def center_crop(x: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = x.shape[-2:]
    if h > H or w > W:
        return None
    t, l = (H - h) // 2, (W - w) // 2
    return x[..., t:t + h, l:l + w]


def slice_noise(
    k: np.ndarray,
    target: np.ndarray = None,
    target_max: float = None,
    masks: dict = None,
    band=(0.30, 0.42),
    outer_band=(0.42, 0.49),
    corner: int = 16,
    max_corr_pixels: int = 20000,
) -> dict:
    """All noise statistics for one slice. k: complex (C, H, W) raw k-space.

    band: |row - centre|/H range used for sigma. The target crop is
    |.| < 0.25, so 0.30-0.42 sits in the air outside it and away from the
    anti-aliasing roll-off at 0.5. outer_band is only a diagnostic: the ratio
    sigma(outer)/sigma(band) shows whether the receiver filter bites there.
    masks: {R: boolean column mask of length W}, already zeroed in padding.
    """
    C, H, W = k.shape
    x = ifft2c(k)
    rows = band_rows(H, *band)
    orows = band_rows(H, *outer_band)
    acq_r, acq_c = acquired_extent(k)
    n_acq = len(acq_r) * len(acq_c)
    to_k = math.sqrt(H * W / n_acq)

    bg = x[:, rows, :]  # (C, nb, W)
    sig_img = np.array([robust_sigma(np.concatenate([bg[c].real.ravel(), bg[c].imag.ravel()])) for c in range(C)])
    sig_std = np.array([float(np.sqrt(0.5 * np.mean(np.abs(bg[c] - bg[c].mean()) ** 2))) for c in range(C)])
    sig_outer = np.array([
        robust_sigma(np.concatenate([x[c, orows].real.ravel(), x[c, orows].imag.ravel()])) for c in range(C)
    ])
    sig_k = sig_img * to_k

    # coil noise correlation from the same background pixels
    B = bg.reshape(C, -1)
    if B.shape[1] > max_corr_pixels:
        B = B[:, np.linspace(0, B.shape[1] - 1, max_corr_pixels).astype(int)]
    B = B - B.mean(axis=1, keepdims=True)
    cov = (B @ B.conj().T) / B.shape[1]
    d = np.sqrt(np.real(np.diag(cov)))
    corr = np.abs(cov / np.outer(d, d))
    off = corr[~np.eye(C, dtype=bool)] if C > 1 else np.array([0.0])

    # measurement SNR: signal energy over expected noise energy on acquired samples
    kk = k[:, acq_r][:, :, acq_c]
    e_y = float(np.sum(np.abs(kk) ** 2))
    e_n = float(n_acq * np.sum(2 * sig_k ** 2))
    rms_k = math.sqrt(e_y / (C * n_acq))

    out = dict(
        coils=C, H=H, W=W, n_acq_rows=len(acq_r), n_acq_cols=len(acq_c),
        sigma_img=float(np.mean(sig_img)), sigma_img_min=float(sig_img.min()), sigma_img_max=float(sig_img.max()),
        sigma_img_std_vs_mad=float(np.mean(sig_std) / np.mean(sig_img)),
        sigma_outer_over_band=float(np.mean(sig_outer) / np.mean(sig_img)),
        sigma_k=float(np.mean(sig_k)),
        sigma_k_rel_rms=float(math.sqrt(2) * np.mean(sig_k) / rms_k),
        sigma_k_rel_max=float(np.mean(sig_k) / np.abs(kk).max()),
        snr_db_full=10 * math.log10(max(e_y - e_n, 1e-30) / e_n),
        noise_corr_mean=float(off.mean()), noise_corr_max=float(off.max()),
    )

    rss = np.sqrt(np.sum(np.abs(x) ** 2, axis=0))
    if target is not None:
        th, tw = target.shape[-2:]
        crop = center_crop(rss, th, tw)
        out["rss_vs_target_relerr"] = (
            float(np.linalg.norm(crop - target) / np.linalg.norm(target)) if crop is not None else float("nan")
        )
        cs = [target[:corner, :corner], target[:corner, -corner:], target[-corner:, :corner], target[-corner:, -corner:]]
        cm = np.concatenate([c.ravel() for c in cs])
        floor_pred = float(np.mean(sig_img)) * chi_mean(2 * C)
        out.update(
            target_corner_mean=float(cm.mean()), target_corner_std=float(cm.std()),
            target_floor_pred=floor_pred, target_floor_ratio=float(cm.mean() / floor_pred),
        )
    if target_max:
        out["sigma_img_rel_target_max"] = float(np.mean(sig_img) / target_max)
        if target is not None:
            out["target_corner_mean_rel_max"] = out["target_corner_mean"] / target_max

    for R, m in (masks or {}).items():
        cols = np.nonzero(m)[0]
        cols = np.intersect1d(cols, acq_c)
        km = k[:, acq_r][:, :, cols]
        ey = float(np.sum(np.abs(km) ** 2))
        en = float(len(acq_r) * len(cols) * np.sum(2 * sig_k ** 2))
        out[f"snr_db_R{R}"] = 10 * math.log10(max(ey - en, 1e-30) / en)
        # zero-filled image noise std for this mask (what the network's input carries)
        out[f"sigma_img_zf_R{R}"] = float(np.mean(sig_k) * math.sqrt(len(acq_r) * len(cols) / (H * W)))
    return out


# ---------------------------------------------------------------------------
# file handling
# ---------------------------------------------------------------------------
def header_info(xml_bytes) -> dict:
    """Field strength, model, channel count, and any header element mentioning noise."""
    info = {}
    try:
        root = ET.fromstring(xml_bytes)
    except Exception as e:  # pragma: no cover - malformed header
        return {"header_error": str(e)}
    noise_tags = set()
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("systemFieldStrength_T", "systemModel", "systemVendor", "receiverChannels") and el.text:
            info[tag] = el.text.strip()
        if "noise" in tag.lower():
            noise_tags.add(tag)
        for a in el.attrib:
            if "noise" in a.lower():
                noise_tags.add(f"{tag}@{a}")
    info["header_noise_tags"] = ";".join(sorted(noise_tags))
    return info


def make_masks(fname: str, W: int, pad_l: int, pad_r: int, accels, cfs):
    """Column masks as the verification harness draws them (seed from the file name)."""
    try:
        from fastmri.data.subsample import EquispacedMaskFractionFunc
    except ImportError:
        return {}
    seed = tuple(map(ord, fname))
    out = {}
    for R, cf in zip(accels, cfs):
        mf = EquispacedMaskFractionFunc(center_fractions=[cf], accelerations=[R])
        res = mf((1, 1, W, 2), offset=None, seed=seed)
        m = (res[0] if isinstance(res, tuple) else res).numpy().reshape(-1).astype(bool)
        m[:pad_l] = False
        m[pad_r:] = False
        out[R] = m
    return out


def process_file(path: Path, fname: str, split: str, args) -> list:
    import h5py

    rows = []
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        attrs = dict(f.attrs)
        hdr = header_info(f["ismrmrd_header"][()]) if "ismrmrd_header" in f else {}
        kspace = f["kspace"]
        target = f["reconstruction_rss"] if "reconstruction_rss" in f else None
        n_sl, _, _, W = kspace.shape
        pad_l = int(attrs.get("padding_left", 0))
        pad_r = int(attrs.get("padding_right", W))
        masks = make_masks(fname, W, pad_l, pad_r, args.accelerations, args.center_fractions)
        tmax = float(attrs["max"]) if "max" in attrs else None
        for s in range(n_sl):
            st = slice_noise(
                kspace[s], target[s] if target is not None else None, tmax, masks,
                band=(args.band_inner, args.band_outer),
            )
            rows.append(dict(
                split=split, fname=fname, slice=s, n_slices=n_sl,
                acquisition=str(attrs.get("acquisition", "")), h5_keys=";".join(keys),
                **hdr, **st,
            ))
    return rows


def iter_inputs(args):
    """Yield (local path, file name). From stdin a member is spooled to scratch and deleted after."""
    if args.data_path:
        for p in sorted(Path(args.data_path).glob("*.h5")):
            if not p.name.startswith("._"):
                yield p, p.name, False
        return
    tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
    tmpdir = Path(args.tmp_dir or tempfile.gettempdir())
    for m in tf:
        name = os.path.basename(m.name)
        if not m.isfile() or not name.endswith(".h5") or name.startswith("._"):
            continue
        tmp = tmpdir / name
        with tf.extractfile(m) as src, open(tmp, "wb") as dst:
            while True:
                b = src.read(1 << 24)
                if not b:
                    break
                dst.write(b)
        yield tmp, name, True


def write_csv(path: Path, rows: list):
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def numeric_cols(rows):
    return [c for c in rows[0] if isinstance(rows[0][c], (int, float)) and c not in ("slice", "n_slices")]


def per_volume(rows):
    vols = {}
    for r in rows:
        vols.setdefault(r["fname"], []).append(r)
    out = []
    for fname, rs in vols.items():
        v = {k: rs[0][k] for k in ("split", "fname", "n_slices", "acquisition", "systemFieldStrength_T",
                                    "systemModel", "receiverChannels", "header_noise_tags", "h5_keys") if k in rs[0]}
        for c in numeric_cols(rs):
            vals = np.array([float(r[c]) for r in rs if c in r and r[c] != ""], float)
            v[c] = float(np.nanmedian(vals)) if vals.size else float("nan")
        out.append(v)
    return out


def summary(vols: list) -> dict:
    def stats(vals):
        a = np.array(vals, float)
        a = a[np.isfinite(a)]
        if not a.size:
            return None
        return dict(n=int(a.size), median=float(np.median(a)), mean=float(a.mean()),
                    p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))

    num = [c for c in vols[0] if isinstance(vols[0][c], float)]
    s = {"volumes": len(vols), "slices": int(sum(v["n_slices"] for v in vols)), "all": {}, "by_acquisition": {}}
    for c in num:
        s["all"][c] = stats([v[c] for v in vols])
    for acq in sorted({v.get("acquisition", "") for v in vols}):
        sub = [v for v in vols if v.get("acquisition", "") == acq]
        s["by_acquisition"][acq] = {"volumes": len(sub), **{c: stats([v[c] for v in sub]) for c in num}}
    for key in ("systemFieldStrength_T", "systemModel", "h5_keys", "header_noise_tags"):
        cnt = {}
        for v in vols:
            cnt[str(v.get(key, ""))] = cnt.get(str(v.get(key, "")), 0) + 1
        s[f"count_{key}"] = cnt
    return s


# ---------------------------------------------------------------------------
# summarize: train + val side by side, markdown
# ---------------------------------------------------------------------------
HEADLINE = [
    ("sigma_img", "σ per coil, image domain (ortho ifft, target units)", "{:.3g}"),
    ("sigma_k", "σ per coil, k-space units", "{:.3g}"),
    ("sigma_img_rel_target_max", "σ / target max (images normalised to max = 1)", "{:.2e}"),
    ("sigma_k_rel_rms", "noise RMS / k-space RMS (amplitude)", "{:.3g}"),
    ("snr_db_full", "measurement SNR, full k-space (dB)", "{:.1f}"),
    ("snr_db_R2", "measurement SNR, R2 mask (dB)", "{:.1f}"),
    ("snr_db_R4", "measurement SNR, R4 mask (dB)", "{:.1f}"),
    ("snr_db_R6", "measurement SNR, R6 mask (dB)", "{:.1f}"),
    ("snr_db_R8", "measurement SNR, R8 mask (dB)", "{:.1f}"),
    ("snr_db_R10", "measurement SNR, R10 mask (dB)", "{:.1f}"),
    ("noise_corr_mean", "mean abs. coil noise correlation", "{:.3f}"),
    ("target_corner_mean_rel_max", "target background floor / target max", "{:.2e}"),
    ("target_floor_ratio", "target floor: measured / predicted from σ", "{:.2f}"),
    ("rss_vs_target_relerr", "RSS(ifft(k)) vs reconstruction_rss, rel. error", "{:.1e}"),
    ("sigma_outer_over_band", "diagnostic: σ(outer rows) / σ(band)", "{:.2f}"),
    ("sigma_img_std_vs_mad", "diagnostic: plain std / robust σ", "{:.2f}"),
]


def summarize(dirs):
    sums = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*_summary.json")):
            s = json.loads(p.read_text())
            sums[s["split"]] = s
    splits = [s for s in ("train", "val") if s in sums] + [s for s in sums if s not in ("train", "val")]
    if not splits:
        sys.exit("no *_summary.json found")
    fmt = lambda st, f: "—" if not st else f"{f.format(st['median'])} [{f.format(st['p5'])}, {f.format(st['p95'])}]"
    print("Per-volume medians over slices; table cells are the median over volumes [5th, 95th percentile].\n")
    print("| quantity | " + " | ".join(f"{s} ({sums[s]['volumes']} vol, {sums[s]['slices']} sl)" for s in splits) + " |")
    print("|---|" + "---|" * len(splits))
    for key, label, f in HEADLINE:
        print(f"| {label} | " + " | ".join(fmt(sums[s]["all"].get(key), f) for s in splits) + " |")
    for s in splits:
        print(f"\n**{s}, by contrast** (median σ / target max, median full-k-space SNR dB):\n")
        print("| acquisition | volumes | σ / target max | SNR full (dB) |")
        print("|---|---|---|---|")
        for acq, g in sums[s]["by_acquisition"].items():
            a, b = g.get("sigma_img_rel_target_max"), g.get("snr_db_full")
            print(f"| {acq or '?'} | {g['volumes']} | {a['median']:.2e} | {b['median']:.1f} |" if a and b else f"| {acq} | {g['volumes']} | — | — |")
        for key in ("systemFieldStrength_T", "h5_keys", "header_noise_tags"):
            print(f"\n{key}: {sums[s].get('count_' + key)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data_path", help="directory of .h5 files (instead of --tar_stdin)")
    ap.add_argument("--tar_stdin", action="store_true", help="read an uncompressed tar stream from stdin")
    ap.add_argument("--tmp_dir", default=None, help="where a streamed .h5 is spooled (default: $TMPDIR)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--accelerations", type=int, nargs="*", default=list(DEFAULT_ACCELS))
    ap.add_argument("--center_fractions", type=float, nargs="*", default=list(DEFAULT_CFS))
    ap.add_argument("--band_inner", type=float, default=0.30)
    ap.add_argument("--band_outer", type=float, default=0.42)
    ap.add_argument("--volume_limit", type=int, default=0)
    ap.add_argument("--summarize", nargs="+", metavar="DIR", help="merge *_summary.json files and print markdown")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return
    if len(args.accelerations) != len(args.center_fractions):
        sys.exit("--accelerations and --center_fractions must have the same length")
    if not (args.data_path or args.tar_stdin):
        sys.exit("give --data_path DIR or --tar_stdin")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for path, fname, spooled in iter_inputs(args):
        try:
            rows.extend(process_file(path, fname, args.split, args))
            n += 1
            print(f"[noise] {args.split} {n}: {fname} sigma_img={rows[-1]['sigma_img']:.3g} "
                  f"snr_full={rows[-1]['snr_db_full']:.1f} dB", flush=True)
        except Exception as e:
            print(f"[noise] ERROR {fname}: {e!r}", file=sys.stderr, flush=True)
        finally:
            if spooled:
                path.unlink(missing_ok=True)
        if n and n % 25 == 0:
            write_csv(out / f"{args.split}_slices.csv", rows)  # partial results survive an eviction
        if args.volume_limit and n >= args.volume_limit:
            break
    if not rows:
        sys.exit("no volumes processed")
    write_csv(out / f"{args.split}_slices.csv", rows)
    vols = per_volume(rows)
    write_csv(out / f"{args.split}_volumes.csv", vols)
    s = summary(vols)
    s["split"] = args.split
    s["band"] = [args.band_inner, args.band_outer]
    s["accelerations"] = args.accelerations
    s["center_fractions"] = args.center_fractions
    (out / f"{args.split}_summary.json").write_text(json.dumps(s, indent=1))
    print(f"[noise] done: {len(vols)} volumes, {len(rows)} slices -> {out}")


if __name__ == "__main__":
    main()
