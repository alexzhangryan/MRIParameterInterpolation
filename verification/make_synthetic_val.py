#!/usr/bin/env python
"""
make_synthetic_val.py: build a tiny fake `multicoil_val` directory with the
exact on-disk layout of fastMRI knee files (kspace, reconstruction_rss,
ismrmrd_header, attrs), so the verification harness and the CHTC job can be
smoke-tested end to end before the real 94 GB download lands.

The images are smooth random phantoms, not MRI. Numbers produced on this data
are meaningless. Its only purpose is to exercise file parsing, masking,
model I/O, metric code, CSV/JSON output, and W&B logging.

  python make_synthetic_val.py --out synthetic/multicoil_val --volumes 3 --slices 4
  tar -cf knee_multicoil_val_synthetic.tar -C synthetic multicoil_val
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

HEADER = """<?xml version="1.0" encoding="utf-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <encoding>
    <encodedSpace><matrixSize><x>{ex}</x><y>{ey}</y><z>1</z></matrixSize></encodedSpace>
    <reconSpace><matrixSize><x>{rx}</x><y>{ry}</y><z>1</z></matrixSize></reconSpace>
    <encodingLimits>
      <kspace_encoding_step_1><minimum>0</minimum><maximum>{kmax}</maximum><center>{kc}</center></kspace_encoding_step_1>
    </encodingLimits>
  </encoding>
</ismrmrdHeader>
"""


def phantom(rng, coils, H, W):
    """Smooth multi-coil image: a few Gaussian blobs times smooth coil profiles."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W), np.float32)
    for _ in range(6):
        cy, cx = rng.uniform(0.25, 0.75, 2) * (H, W)
        sy, sx = rng.uniform(0.05, 0.15, 2) * (H, W)
        img += rng.uniform(0.3, 1.0) * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    coil_imgs = []
    for c in range(coils):
        cy, cx = rng.uniform(0, 1, 2) * (H, W)
        prof = np.exp(-(((yy - cy) / (0.7 * H)) ** 2 + ((xx - cx) / (0.7 * W)) ** 2))
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi) * prof)
        coil_imgs.append((img * prof * phase).astype(np.complex64))
    return np.stack(coil_imgs)  # (C, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("synthetic/multicoil_val"))
    ap.add_argument("--volumes", type=int, default=3)
    ap.add_argument("--slices", type=int, default=4)
    ap.add_argument("--coils", type=int, default=4)
    ap.add_argument("--height", type=int, default=128, help="readout size (real knee: 640)")
    ap.add_argument("--width", type=int, default=96, help="phase-encode size (real knee: 368)")
    ap.add_argument("--recon", type=int, default=64, help="square recon size (real knee: 320)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    H, W = a.height, a.width
    for v in range(a.volumes):
        ks, rss = [], []
        for _ in range(a.slices):
            ci = phantom(rng, a.coils, H, W)
            k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ci, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
            ks.append(k.astype(np.complex64))
            r = np.sqrt((np.abs(ci) ** 2).sum(0))
            h0, w0 = (H - a.recon) // 2, (W - a.recon) // 2
            rss.append(r[h0:h0 + a.recon, w0:w0 + a.recon].astype(np.float32))
        ks, rss = np.stack(ks), np.stack(rss)
        fname = a.out / f"file{1000000 + v}.h5"
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("kspace", data=ks)
            hf.create_dataset("reconstruction_rss", data=rss)
            hf.create_dataset("ismrmrd_header", data=HEADER.format(
                ex=H, ey=W, rx=a.recon, ry=a.recon, kmax=W - 1, kc=W // 2).encode())
            hf.attrs["max"] = float(rss.max())
            hf.attrs["norm"] = float(np.linalg.norm(rss))
            hf.attrs["acquisition"] = "CORPD_FBK"
            hf.attrs["patient_id"] = f"synthetic{v}"
        print(f"wrote {fname}  kspace {ks.shape}  rss {rss.shape}")


if __name__ == "__main__":
    main()
