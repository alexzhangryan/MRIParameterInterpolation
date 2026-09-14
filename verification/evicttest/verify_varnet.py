#!/usr/bin/env python
"""
verify_varnet.py: standalone verification harness for the E2E VarNet baseline.

Implements Tier 0 and Tier 1 of VERIFICATION.md. It imports the `fastmri`
package but never modifies it, and it does not touch anything under fastMRI/.

Subcommands
-----------
  tier0   Environment, API, and metric-code invariants. No data, no GPU.
  tier1   Released-checkpoint evaluation on multicoil_val at one or more
          acceleration rates, with per-volume metrics, internal invariants,
          comparison against third-party reference values, and W&B logging.

Typical use
-----------
  python verify_varnet.py tier0 --fastmri_repo /opt/fastMRI --run_pytest
  python verify_varnet.py tier1 \
      --data_path /path/to/multicoil_val \
      --state_dict knee_leaderboard_state_dict.pt \
      --accelerations 4 --center_fractions 0.08 \
      --mask_type equispaced_fraction --output_dir output

  --state_dict accepts the released fastMRI state dict or a Lightning
  checkpoint written by ../training/train_wandb.py. The architecture
  (cascades, channels, pools) is read from the file, so a checkpoint trained
  with 8 cascades and the released 12-cascade model load the same way. The
  --num_cascades/--chans/... flags only matter with --random_init.

W&B
---
  Logs to Weights & Biases when WANDB_API_KEY is set in the environment.
  With no key it falls back to offline mode automatically and writes the run
  under <output_dir>/wandb for a later `wandb sync`. Pass --no_wandb to
  disable entirely. The key is never read from a file or an argument.
  As in training, the run id is --run_name (resume="allow"), so a
  resubmitted job continues the same W&B run; pass a new --run_name (or
  --wandb_id) for a genuinely new one.

Exit codes
----------
  0  every check passed or is in the "investigate" band
  1  at least one check failed, or an error
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

STATE_DICT_URL = (
    "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    "knee_leaderboard_state_dict.pt"
)

# Architecture of the released knee checkpoint, the CLI defaults. When a
# checkpoint is loaded the architecture is read from its state dict
# (infer_arch) and these are ignored; they only matter for --random_init,
# where a tiny model keeps the harness smoke test to minutes. Training
# (../training/train_wandb.py) uses 8 cascades, so a checkpoint from there
# differs from this in num_cascades only.
RELEASED_ARCH = dict(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)

# Third-party measurements of the released checkpoint. These are pipeline
# anchors, not paper numbers. Both sources evaluated a checkpoint that was
# trained on train+val, so they say nothing about generalisation. They are
# keyed on the fastMRI knee convention (random masks, 0.08 / 0.04).
# See VERIFICATION.md Section 2.
REFERENCE_VALUES = {
    ("random", 4): [
        dict(source="PromptMR README (released ckpt, own val-derived split)",
             ssim=0.9236, psnr=39.37, nmse=0.0053),
    ],
    ("random", 8): [
        dict(source="PromptMR README (released ckpt, own val-derived split)",
             ssim=0.8936, psnr=37.30, nmse=0.0087),
        dict(source="HUMUS-Net Table 2 (knee validation)",
             ssim=0.8908),
    ],
}

# (pass_below, fail_above). Between the two is "investigate".
# Judgment thresholds from VERIFICATION.md Section 4.4, not derived quantities.
THRESHOLDS = {
    "ssim": (0.010, 0.020),        # absolute
    "psnr": (0.5, 1.0),            # dB
    "nmse": (0.15, 0.30),          # relative
}

RESULTS_CSV_COLUMNS = [
    "run_id", "date", "git_sha", "tier", "model_source", "checkpoint_path",
    "split", "n_volumes", "mask_type", "center_fraction", "acceleration",
    "mask_seed", "train_seed", "ssim_mean", "ssim_std", "psnr_mean",
    "nmse_mean", "per_volume_csv", "notes",
]


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[verify {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verdict_for(delta: float, key: str) -> str:
    lo, hi = THRESHOLDS[key]
    if delta <= lo:
        return "pass"
    if delta <= hi:
        return "investigate"
    return "fail"


def worst(verdicts: Sequence[str]) -> str:
    order = {"pass": 0, "investigate": 1, "fail": 2, "error": 3}
    return max(verdicts, key=lambda v: order.get(v, 3)) if verdicts else "pass"


def git_sha_of(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def append_results_row(csv_path: Path, row: Dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULTS_CSV_COLUMNS})


# --------------------------------------------------------------------------
# W&B wrapper. Every call is a no-op when disabled, so the rest of the code
# never has to check.
# --------------------------------------------------------------------------

class WandB:
    def __init__(self, args, tier: str, run_name: str, config: Dict):
        self.run = None
        self.enabled = not args.no_wandb
        if not self.enabled:
            return
        try:
            import wandb  # noqa: WPS433
        except ImportError:
            log("wandb not installed, continuing without it")
            self.enabled = False
            return

        mode = os.environ.get("WANDB_MODE")
        if mode is None:
            mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
            if mode == "offline":
                log("WANDB_API_KEY not set, W&B running in offline mode "
                    "(sync later with `wandb sync <output_dir>/wandb/offline-run-*`)")
        os.makedirs(args.output_dir, exist_ok=True)
        self.wandb = wandb
        # A verification run is expensive (the real tier1 is an hour of GPU
        # time) and W&B is a convenience, not a result. Never let it take the
        # run down with it: a bad key, an expired login or a network blip
        # degrades to "no logging", not to a lost run.
        try:
            self.run = self._init(args, tier, run_name, config, mode)
        except Exception as e:  # noqa: BLE001
            log(f"wandb init failed ({e!r}); continuing without it")
            self.run = None
            self.enabled = False

    def _init(self, args, tier, run_name, config, mode):
        wandb = self.wandb
        # The run id is the run name (as in training/train_wandb.py) so a
        # resubmitted job continues the same W&B run. Ids allow only
        # [A-Za-z0-9_-]; a hostname-derived default name has dots.
        run_id = re.sub(r"[^A-Za-z0-9_-]", "-", args.wandb_id or run_name)
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            id=run_id,
            resume="allow",
            job_type=tier,
            tags=[tier, "e2e-varnet", "verification"],
            config=config,
            dir=str(args.output_dir),
            mode=mode,
            reinit=True,
        )

    def _guard(self, what, fn) -> None:
        if self.run is None:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log(f"wandb {what} failed ({e!r}); disabling W&B for the rest of this run")
            self.run = None
            self.enabled = False

    def log(self, data: Dict, step: Optional[int] = None) -> None:
        self._guard("log", lambda: self.run.log(data, step=step))

    def summary(self, data: Dict) -> None:
        def _do():
            for k, v in data.items():
                self.run.summary[k] = v
        self._guard("summary", _do)

    def table(self, name: str, columns: List[str], rows: List[List]) -> None:
        self._guard("table", lambda: self.run.log(
            {name: self.wandb.Table(columns=columns, data=rows)}))

    def images(self, name: str, images: Dict[str, np.ndarray]) -> None:
        self._guard("images", lambda: self.run.log(
            {f"{name}/{k}": self.wandb.Image(v) for k, v in images.items()}))

    def artifact_file(self, path: Path, name: str, type_: str) -> None:
        if not path.exists():
            return
        def _do():
            art = self.wandb.Artifact(name=name, type=type_)
            art.add_file(str(path))
            self.run.log_artifact(art)
        self._guard("artifact", _do)

    def finish(self) -> None:
        self._guard("finish", lambda: self.run.finish())


# --------------------------------------------------------------------------
# Tier 0
# --------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    status: str            # pass / fail / warn / skip
    detail: str = ""
    value: Dict = field(default_factory=dict)


def tier0(args) -> int:
    checks: List[Check] = []

    # name/status/detail are positional-only: a check that wants to record a
    # value called "name" (the GPU one does) would otherwise collide with the
    # parameter and raise TypeError, which the outer except turns into a
    # bogus framework_import failure.
    def add(name, status, detail="", /, **value):
        checks.append(Check(name, status, detail, dict(value)))
        log(f"{status.upper():6s} {name}: {detail}")

    # 0.1 framework API
    try:
        import torch
        import pytorch_lightning as pl
        v = pl.__version__
        ok = v.startswith("1.")
        add("pl_version_1x", "pass" if ok else "fail",
            f"pytorch_lightning {v} (repo requires 1.x)", version=v)
        has = hasattr(pl.Trainer, "from_argparse_args")
        add("pl_from_argparse_args", "pass" if has else "fail",
            "Trainer.from_argparse_args present" if has else "missing: every repo training script will fail")
        add("torch_version", "pass", f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}",
            version=torch.__version__, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            add("gpu", "pass", f"{torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}",
                device=torch.cuda.get_device_name(0))
        else:
            add("gpu", "warn", "no CUDA device visible (fine for tier0, tier1 will be slow)")
    except Exception as e:  # noqa: BLE001
        add("framework_import", "fail", repr(e))

    # 0.2 fastmri imports
    try:
        import fastmri  # noqa: F401
        from fastmri.models import VarNet  # noqa: F401
        from fastmri.data.transforms import VarNetDataTransform  # noqa: F401
        from fastmri.data.subsample import create_mask_for_mask_type  # noqa: F401
        from fastmri import evaluate as fe  # noqa: F401
        add("fastmri_import", "pass", f"fastmri {getattr(fastmri, '__version__', '?')}",
            version=getattr(fastmri, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        add("fastmri_import", "fail", repr(e))
        return finish_tier0(args, checks)

    # 0.3 metric identity: ground truth fed in as the prediction
    try:
        from fastmri.evaluate import mse, nmse, psnr, ssim
        rng = np.random.RandomState(0)
        gt = rng.rand(16, 320, 320).astype(np.float32)
        s, n, m, p = (float(np.squeeze(f(gt, gt))) for f in (ssim, nmse, mse, psnr))
        ok = abs(s - 1.0) < 1e-6 and n == 0.0 and m == 0.0 and (np.isinf(p) or p > 100)
        add("metric_identity", "pass" if ok else "fail",
            f"ssim={s:.8f} nmse={n} mse={m} psnr={p}", ssim=s, nmse=n, mse=m, psnr=p)
    except Exception as e:  # noqa: BLE001
        add("metric_identity", "fail", repr(e))

    # 0.4 mask conventions, and the documented centre-line deviation
    try:
        import torch
        from fastmri.data.subsample import create_mask_for_mask_type
        width = 368  # typical knee phase-encode width
        for mask_type, cf, R, paper_lines in [
            ("random", 0.08, 4, 30), ("random", 0.04, 8, 16),
            ("equispaced", 0.08, 4, 30), ("equispaced", 0.04, 8, 16),
        ]:
            mf = create_mask_for_mask_type(mask_type, [cf], [R])
            mask, nlf = mf((1, 640, width, 2), seed=0)
            frac = float(mask.float().mean())
            add(f"mask_{mask_type}_R{R}", "pass",
                f"center_fraction {cf} -> {int(nlf)} lines at width {width} "
                f"(paper fixed count {paper_lines}); sampled fraction {frac:.4f} vs 1/R={1/R:.4f}",
                num_low_freqs=int(nlf), sampled_fraction=frac, paper_lines=paper_lines)
    except Exception as e:  # noqa: BLE001
        add("mask_conventions", "fail", repr(e))

    # 0.5 optional: the repo's own test suite
    if args.run_pytest:
        repo = args.fastmri_repo
        if repo is None or not (Path(repo) / "tests").exists():
            add("repo_pytest", "skip", "pass --fastmri_repo pointing at a checkout with tests/")
        else:
            log("running fastMRI test suite (this takes a few minutes)")
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "tests", "-x", "-q", "-p", "no:cacheprovider"],
                cwd=str(repo), capture_output=True, text=True,
            )
            lines = (r.stdout.strip() or r.stderr.strip()).splitlines()
            summary = [ln for ln in lines if " passed" in ln or " failed" in ln or "error" in ln.lower()]
            tail = (summary[-1] if summary else (lines[-1] if lines else "")).strip("= ")
            add("repo_pytest", "pass" if r.returncode == 0 else "fail", tail, returncode=r.returncode)
            (Path(args.output_dir) / "pytest_output.txt").write_text(r.stdout + "\n" + r.stderr)
    else:
        add("repo_pytest", "skip", "not requested (--run_pytest)")

    return finish_tier0(args, checks)


def finish_tier0(args, checks: List[Check]) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall = "fail" if any(c.status == "fail" for c in checks) else "pass"
    report = {
        "tier": "tier0",
        "overall": overall,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "checks": [asdict(c) for c in checks],
    }
    (out / "tier0_report.json").write_text(json.dumps(report, indent=2, default=json_safe))
    log(f"tier0 overall: {overall.upper()}  (report: {out / 'tier0_report.json'})")

    wb = WandB(args, "tier0", args.run_name or f"tier0-{platform.node()}", {"tier": "tier0"})
    wb.table("tier0/checks", ["check", "status", "detail"],
             [[c.name, c.status, c.detail] for c in checks])
    wb.summary({"tier0/overall": overall,
                **{f"tier0/{c.name}": c.status for c in checks}})
    wb.artifact_file(out / "tier0_report.json", "tier0_report", "report")
    wb.finish()
    return 0 if overall == "pass" else 1


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------

def set_determinism(seed: int) -> None:
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download(url: str, dest: Path) -> None:
    import requests
    from tqdm import tqdm
    log(f"downloading {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="iB", unit_scale=True, desc="state_dict") as bar:
        for chunk in r.iter_content(8 * 1024 * 1024):
            fh.write(chunk)
            bar.update(len(chunk))


def infer_arch(state: Dict) -> Dict[str, int]:
    """Read the VarNet constructor arguments back out of a state dict.

    fastmri.models.varnet names things as
        cascades.<i>.model.unet.down_sample_layers.<j>.layers.0.weight
        sens_net.norm_unet.unet.down_sample_layers.<j>.layers.0.weight
    so the cascade count is the number of distinct <i>, the pool count the
    number of distinct <j>, and the channel count the first conv's output
    dim. This is what lets one harness score both the released 12-cascade
    checkpoint and an 8-cascade checkpoint from ../training.
    """
    def unet(prefix: str):
        pre = prefix + "down_sample_layers."
        levels = {int(k[len(pre):].split(".")[0]) for k in state if k.startswith(pre)}
        if not levels:
            raise KeyError(f"no '{pre}*' keys in state dict; is this a fastmri VarNet checkpoint?")
        chans = int(state[pre + "0.layers.0.weight"].shape[0])
        return len(levels), chans

    cascades = {int(k.split(".")[1]) for k in state if k.startswith("cascades.")}
    pools, chans = unet("cascades.0.model.unet.")
    sens_pools, sens_chans = unet("sens_net.norm_unet.unet.")
    return dict(num_cascades=len(cascades), pools=pools, chans=chans,
                sens_pools=sens_pools, sens_chans=sens_chans)


def load_state(sd_path: Path) -> Dict:
    """torch.load either the released state dict or a Lightning checkpoint."""
    import torch
    state = torch.load(str(sd_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:  # a Lightning checkpoint
        state = {k[len("varnet."):] if k.startswith("varnet.") else k: v
                 for k, v in state["state_dict"].items()}
    return state


def load_model(args, device):
    from fastmri.models import VarNet

    cli_arch = dict(num_cascades=args.num_cascades, pools=args.pools, chans=args.chans,
                    sens_pools=args.sens_pools, sens_chans=args.sens_chans)

    if args.random_init:
        log("--random_init: using an untrained model (harness smoke test only, numbers are meaningless)")
        arch, source = cli_arch, "random_init"
        model = VarNet(**arch)
    else:
        sd_path = Path(args.state_dict)
        if not sd_path.exists():
            if not args.download_state_dict:
                raise FileNotFoundError(
                    f"{sd_path} not found. Pass --download_state_dict or stage it via transfer_input_files.")
            download(STATE_DICT_URL, sd_path)
        state = load_state(sd_path)
        arch = infer_arch(state)
        if arch != cli_arch:
            log(f"architecture read from {sd_path.name}: {arch} (CLI flags {cli_arch} ignored)")
        model = VarNet(**arch)
        model.load_state_dict(state, strict=True)
        source = "pretrained"
        log(f"loaded {sd_path} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    return model.eval().to(device), arch, source


def build_dataset(args, mask_func, selected: List[str]):
    from fastmri.data import SliceDataset
    from fastmri.data.transforms import VarNetDataTransform

    selected_set = set(selected)
    # use_seed=True (the default): the mask is a deterministic function of the
    # filename, which is the fastMRI convention for validation and what other
    # groups' reported numbers were produced with.
    transform = VarNetDataTransform(mask_func=mask_func, use_seed=True)
    ds = SliceDataset(
        root=args.data_path,
        challenge="multicoil",
        transform=transform,
        raw_sample_filter=lambda s: s.fname.name in selected_set,
    )
    by_volume: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(ds.raw_samples):
        by_volume[s.fname.name].append(i)
    for k in by_volume:
        by_volume[k].sort(key=lambda i: ds.raw_samples[i].slice_ind)
    return ds, by_volume


def zero_filled_rss(masked_kspace):
    """Zero-filled root-sum-of-squares reconstruction, same input as the model."""
    import fastmri
    img = fastmri.ifft2c(masked_kspace)               # (B, C, H, W, 2)
    return fastmri.rss(fastmri.complex_abs(img), dim=1)  # (B, H, W)


def crop_like_evaluate(target: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replicates fastmri.evaluate: square crop both to the target's last dim."""
    from fastmri.data import transforms as T
    side = target.shape[-1]
    return (T.center_crop(target, (side, side)), T.center_crop(pred, (side, side)))


def run_volume(model, ds, indices: List[int], device, num_workers: int, zero_filled: bool = True):
    """Reconstruct one volume slice by slice. Returns (target, recon, zf) as (S, H, W) numpy."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from fastmri.data import transforms as T

    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, num_workers=num_workers)
    targets, recons, zfs, slice_ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            crop = tuple(int(c[0]) for c in batch.crop_size)
            mk = batch.masked_kspace.to(device)
            mask = batch.mask.to(device)
            # num_low_frequencies must move too: SensitivityModel does
            # `num_low_frequencies * torch.ones(..., device=mask.device)`, so a
            # CPU value here raises a device mismatch on GPU. Invisible on CPU.
            nlf = batch.num_low_frequencies
            if torch.is_tensor(nlf):
                nlf = nlf.to(device)
            # Same call as VarNetModule.forward, the signature the model was trained under.
            out = model(mk, mask, nlf).cpu()
            if out.shape[-1] < crop[1]:  # brain FLAIR 203 special case, kept for parity
                crop = (out.shape[-1], out.shape[-1])
            out = T.center_crop(out, crop)[0]
            recons.append(out.numpy())
            targets.append(batch.target[0].numpy())
            slice_ids.append(int(batch.slice_num[0]))
            if zero_filled:
                zf = zero_filled_rss(mk).cpu()
                zf = T.center_crop(zf, crop)[0]
                zfs.append(zf.numpy())
    order = np.argsort(slice_ids)
    stack = lambda xs: np.stack([xs[i] for i in order]) if xs else None  # noqa: E731
    return stack(targets), stack(recons), stack(zfs)


def volume_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    from fastmri.evaluate import mse, nmse, psnr, ssim
    t, p = crop_like_evaluate(target, pred)
    return dict(ssim=float(np.squeeze(ssim(t, p))), psnr=float(np.squeeze(psnr(t, p))),
                nmse=float(np.squeeze(nmse(t, p))), mse=float(np.squeeze(mse(t, p))))


def compare_to_reference(mask_type: str, R: int, agg: Dict[str, float]) -> List[Dict]:
    rows = []
    for ref in REFERENCE_VALUES.get((mask_type, R), []):
        row = dict(source=ref["source"])
        vs = []
        if "ssim" in ref:
            d = abs(agg["ssim_mean"] - ref["ssim"])
            row.update(ref_ssim=ref["ssim"], d_ssim=d, v_ssim=verdict_for(d, "ssim"))
            vs.append(row["v_ssim"])
        if "psnr" in ref:
            d = abs(agg["psnr_mean"] - ref["psnr"])
            row.update(ref_psnr=ref["psnr"], d_psnr=d, v_psnr=verdict_for(d, "psnr"))
            vs.append(row["v_psnr"])
        if "nmse" in ref:
            d = abs(agg["nmse_mean"] - ref["nmse"]) / ref["nmse"]
            row.update(ref_nmse=ref["nmse"], d_nmse_rel=d, v_nmse=verdict_for(d, "nmse"))
            vs.append(row["v_nmse"])
        row["verdict"] = worst(vs)
        rows.append(row)
    return rows


def log_reference_table(R: int, ref: Dict, agg: Dict) -> None:
    """Measured vs published side by side.

    The deltas alone are unreadable without knowing what the paper reported,
    and the tolerance that turned a delta into a verdict is in THRESHOLDS,
    three screens away. Print all four so a line can be judged on its own.
    """
    log("R{} vs {}: {}".format(R, ref["source"], ref["verdict"].upper()))
    log("    {:<10} {:>11} {:>11} {:>13} {:>16} {:>9}".format(
        "metric", "measured", "published", "difference", "tol pass/fail", "verdict"))
    if "ref_ssim" in ref:
        lo, hi = THRESHOLDS["ssim"]
        log("    {:<10} {:>11.4f} {:>11.4f} {:>13.4f} {:>16} {:>9}".format(
            "SSIM", agg["ssim_mean"], ref["ref_ssim"], ref["d_ssim"],
            "{:g} / {:g}".format(lo, hi), ref["v_ssim"]))
    if "ref_psnr" in ref:
        lo, hi = THRESHOLDS["psnr"]
        log("    {:<10} {:>11.2f} {:>11.2f} {:>13.2f} {:>16} {:>9}".format(
            "PSNR (dB)", agg["psnr_mean"], ref["ref_psnr"], ref["d_psnr"],
            "{:g} / {:g} dB".format(lo, hi), ref["v_psnr"]))
    if "ref_nmse" in ref:
        lo, hi = THRESHOLDS["nmse"]
        log("    {:<10} {:>11.4f} {:>11.4f} {:>13} {:>16} {:>9}".format(
            "NMSE", agg["nmse_mean"], ref["ref_nmse"],
            "{:.1f}% rel".format(100 * ref["d_nmse_rel"]),
            "{:g}% / {:g}%".format(100 * lo, 100 * hi), ref["v_nmse"]))


def to_uint8_image(x: np.ndarray, vmax: float) -> np.ndarray:
    x = np.clip(x / max(vmax, 1e-12), 0, 1)
    return (x * 255).astype(np.uint8)


def tier1(args) -> int:
    import torch
    from fastmri.data.subsample import create_mask_for_mask_type

    if len(args.accelerations) != len(args.center_fractions):
        raise ValueError("--accelerations and --center_fractions must have the same length "
                         "(they are paired elementwise, as in fastmri.data.subsample.MaskFunc)")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"device: {device}")

    data_path = Path(args.data_path)
    files = sorted(p.name for p in data_path.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files under {data_path}")
    if args.volume_limit:
        files = files[: args.volume_limit]
    log(f"{len(files)} volumes selected from {data_path}")

    model, arch, model_source = load_model(args, device)
    run_id = args.run_name or f"tier1-{model_source}-{time.strftime('%Y%m%d-%H%M%S')}"
    config = dict(vars(args), arch=arch, model_source=model_source, device=str(device),
                  n_volumes=len(files), fastmri_sha=git_sha_of(args.fastmri_repo),
                  container_image=os.environ.get("VERIFY_IMAGE", ""))
    wb = WandB(args, "tier1", run_id, config)

    per_rate: Dict[int, Dict] = {}
    all_verdicts: List[str] = []
    invariants: List[Check] = []

    def inv(name, status, detail="", **v):
        invariants.append(Check(name, status, detail, dict(v)))
        log(f"{status.upper():6s} invariant {name}: {detail}")

    for R, cf in zip(args.accelerations, args.center_fractions):
        log(f"=== acceleration {R}x, center_fraction {cf}, mask {args.mask_type} ===")
        mask_func = create_mask_for_mask_type(args.mask_type, [cf], [R])
        ds, by_volume = build_dataset(args, mask_func, files)
        if set(by_volume) != set(files):
            missing = sorted(set(files) - set(by_volume))
            inv(f"R{R}_volume_listing", "fail", f"{len(missing)} selected volumes not indexed: {missing[:3]}...")
            all_verdicts.append("fail")
            continue

        rows: List[Dict] = []
        det_records: List[Dict] = []
        t0 = time.time()
        for vi, fname in enumerate(files):
            idx = by_volume[fname]
            target, recon, zf = run_volume(model, ds, idx, device, args.num_workers)
            if target.shape != recon.shape:
                # evaluate.py would still square-crop, but a mismatch here is worth surfacing
                log(f"note: {fname} target {target.shape} vs recon {recon.shape} (will square-crop)")
            m = volume_metrics(target, recon)
            mz = volume_metrics(target, zf)
            row = dict(fname=fname, n_slices=int(target.shape[0]), acceleration=R, center_fraction=cf,
                       **m, **{f"zf_{k}": v for k, v in mz.items()})
            rows.append(row)

            if vi < args.determinism_volumes:
                _, recon2, _ = run_volume(model, ds, idx, device, args.num_workers, zero_filled=False)
                maxdiff = float(np.max(np.abs(recon - recon2)))
                m2 = volume_metrics(target, recon2)
                det_records.append(dict(fname=fname, max_abs_diff=maxdiff,
                                        ssim_diff=abs(m["ssim"] - m2["ssim"])))

            if vi < args.image_volumes:
                s = target.shape[0] // 2
                vmax = float(target[s].max())
                t_, r_ = crop_like_evaluate(target[s], recon[s])
                _, z_ = crop_like_evaluate(target[s], zf[s])
                wb.images(f"R{R}/{Path(fname).stem}", {
                    "target": to_uint8_image(t_, vmax),
                    "recon": to_uint8_image(r_, vmax),
                    "zero_filled": to_uint8_image(z_, vmax),
                    "abs_error_x5": to_uint8_image(np.abs(t_ - r_) * 5, vmax),
                })

            if (vi + 1) % args.log_every == 0 or vi + 1 == len(files):
                elapsed = time.time() - t0
                run_ssim = np.mean([r["ssim"] for r in rows])
                log(f"R{R} {vi+1}/{len(files)} volumes, running SSIM {run_ssim:.4f}, "
                    f"{elapsed/(vi+1):.1f}s/volume")
                wb.log({f"R{R}/progress_volumes": vi + 1, f"R{R}/running_ssim": run_ssim,
                        f"R{R}/sec_per_volume": elapsed / (vi + 1)})

        # per-volume CSV
        pv_csv = out / f"per_volume_R{R}.csv"
        with open(pv_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        agg = {}
        for k in ("ssim", "psnr", "nmse", "mse", "zf_ssim", "zf_psnr", "zf_nmse"):
            vals = np.array([r[k] for r in rows], dtype=np.float64)
            agg[f"{k}_mean"] = float(vals.mean())
            agg[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg["n_volumes"] = len(rows)
        agg["n_slices"] = int(sum(r["n_slices"] for r in rows))
        agg["seconds"] = time.time() - t0

        log("R{} measured over {} volumes / {} slices in {:.0f}s".format(
            R, agg["n_volumes"], agg["n_slices"], agg["seconds"]))
        log("    {:<10} {:>11} {:>11} {:>11}".format("metric", "model", "zero-filled", "model std"))
        log("    {:<10} {:>11.4f} {:>11.4f} {:>11.4f}".format(
            "SSIM", agg["ssim_mean"], agg["zf_ssim_mean"], agg["ssim_std"]))
        log("    {:<10} {:>11.2f} {:>11.2f} {:>11.2f}".format(
            "PSNR (dB)", agg["psnr_mean"], agg["zf_psnr_mean"], agg["psnr_std"]))
        log("    {:<10} {:>11.4f} {:>11.4f} {:>11.4f}".format(
            "NMSE", agg["nmse_mean"], agg["zf_nmse_mean"], agg["nmse_std"]))

        # reference comparison
        refs = compare_to_reference(args.mask_type, R, agg)
        for ref in refs:
            log_reference_table(R, ref, agg)
            all_verdicts.append(ref["verdict"])
        if refs and args.volume_limit:
            log("    NOTE: --volume_limit {} takes the FIRST N files by name, not a random".format(args.volume_limit))
            log("          sample. knee val is 50/50 CORPD/CORPDFS overall but the sorted head")
            log("          is not (the first 5 are 80% CORPDFS), and the two score differently,")
            log("          so the comparison above is indicative only -- not a scored verdict.")
        if not refs:
            log("R{}: no reference values for ({}, {}); recorded only".format(R, args.mask_type, R))

        # invariants for this rate
        if det_records:
            md = max(d["max_abs_diff"] for d in det_records)
            ok = md == 0.0 or md < args.determinism_tol
            inv(f"R{R}_determinism", "pass" if ok else "fail",
                f"max |recon - recon_rerun| = {md:.3e} over {len(det_records)} volumes (tol {args.determinism_tol:g})",
                max_abs_diff=md)
            all_verdicts.append("pass" if ok else "fail")
        margin = agg["ssim_mean"] - agg["zf_ssim_mean"]
        ok = margin > args.zf_min_margin
        inv(f"R{R}_beats_zero_filled", "pass" if ok else "fail",
            f"model SSIM - zero-filled SSIM = {margin:+.4f} (min {args.zf_min_margin})", margin=margin)
        all_verdicts.append("pass" if ok else "fail")
        inv(f"R{R}_volume_count", "pass" if agg["n_volumes"] == len(files) else "fail",
            f"{agg['n_volumes']} scored of {len(files)} selected")

        per_rate[R] = dict(center_fraction=cf, aggregate=agg, references=refs,
                           determinism=det_records, per_volume_csv=str(pv_csv))

        wb.log({f"R{R}/{k}": v for k, v in agg.items()})
        wb.table(f"R{R}/per_volume", list(rows[0].keys()), [list(r.values()) for r in rows])
        wb.artifact_file(pv_csv, f"per_volume_R{R}", "per_volume_metrics")
        append_results_row(out / "results.csv", dict(
            run_id=run_id, date=time.strftime("%Y-%m-%d"), git_sha=config["fastmri_sha"], tier="tier1",
            model_source=model_source, checkpoint_path=args.state_dict if model_source == "pretrained" else "",
            split=data_path.name, n_volumes=agg["n_volumes"], mask_type=args.mask_type,
            center_fraction=cf, acceleration=R, mask_seed="filename", train_seed="",
            ssim_mean=f"{agg['ssim_mean']:.6f}", ssim_std=f"{agg['ssim_std']:.6f}",
            psnr_mean=f"{agg['psnr_mean']:.4f}", nmse_mean=f"{agg['nmse_mean']:.6f}",
            per_volume_csv=pv_csv.name, notes=" | ".join(f"{r['source']}: {r['verdict']}" for r in refs),
        ))

    # cross-rate ordering
    if len(per_rate) > 1:
        Rs = sorted(per_rate)
        ss = [per_rate[R]["aggregate"]["ssim_mean"] for R in Rs]
        ok = all(a > b for a, b in zip(ss, ss[1:]))
        inv("ssim_monotone_in_R", "pass" if ok else "fail",
            "  ".join(f"R{R}={s:.4f}" for R, s in zip(Rs, ss)))
        all_verdicts.append("pass" if ok else "fail")

    overall = worst(all_verdicts)
    report = dict(
        tier="tier1", overall=overall, run_id=run_id, model_source=model_source, arch=arch,
        data_path=str(data_path), n_volumes=len(files), mask_type=args.mask_type, seed=args.seed,
        device=str(device), thresholds=THRESHOLDS, per_rate=per_rate,
        invariants=[asdict(c) for c in invariants], config=config,
    )
    (out / "tier1_report.json").write_text(json.dumps(report, indent=2, default=json_safe))
    wb.summary({"overall": overall,
                **{f"R{R}/ssim_mean": v["aggregate"]["ssim_mean"] for R, v in per_rate.items()},
                **{f"R{R}/verdict": worst([r["verdict"] for r in v["references"]] or ["pass"])
                   for R, v in per_rate.items()},
                **{f"invariant/{c.name}": c.status for c in invariants}})
    wb.table("tier1/reference_comparison",
             ["acceleration", "source", "verdict", "ssim", "ref_ssim", "psnr", "ref_psnr", "nmse", "ref_nmse"],
             [[R, r["source"], r["verdict"], v["aggregate"]["ssim_mean"], r.get("ref_ssim"),
               v["aggregate"]["psnr_mean"], r.get("ref_psnr"), v["aggregate"]["nmse_mean"], r.get("ref_nmse")]
              for R, v in per_rate.items() for r in v["references"]])
    wb.artifact_file(out / "tier1_report.json", "tier1_report", "report")
    wb.finish()

    log(f"tier1 overall: {overall.upper()}  (report: {out / 'tier1_report.json'})")
    if model_source == "random_init":
        log("reminder: --random_init numbers are meaningless, this was a harness smoke test")
    return 0 if overall in ("pass", "investigate") else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output_dir", type=Path, default=Path("output"))
    p.add_argument("--run_name", type=str, default=None,
                   help="W&B run name and id (unless --wandb_id is given) and results.csv run_id. "
                        "Keep it the same across resubmissions of one run so W&B continues the run.")
    p.add_argument("--wandb_id", type=str, default=None, help="override the W&B run id (default: --run_name)")
    p.add_argument("--fastmri_repo", type=Path, default=None,
                   help="fastMRI checkout, used for git sha and (tier0) --run_pytest")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "fastmri-varnet-verify"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("tier0", help="environment and metric-code invariants")
    add_common(p0)
    p0.add_argument("--run_pytest", action="store_true", help="also run the fastMRI repo test suite")

    p1 = sub.add_parser("tier1", help="released-checkpoint evaluation on multicoil_val")
    add_common(p1)
    p1.add_argument("--data_path", type=Path, required=True, help="directory of multicoil_val .h5 files")
    p1.add_argument("--state_dict", type=str, default="knee_leaderboard_state_dict.pt",
                    help="released fastMRI state dict or a Lightning .ckpt from ../training; "
                         "the architecture is read from the file")
    p1.add_argument("--download_state_dict", action="store_true",
                    help="download the released checkpoint if --state_dict is missing")
    # defaults match ../training/train_wandb.py (model 1). random + 4 8 is
    # the fastMRI knee convention the reference values are keyed on.
    p1.add_argument("--mask_type", choices=("random", "equispaced", "equispaced_fraction"),
                    default="equispaced_fraction")
    p1.add_argument("--accelerations", type=int, nargs="+", default=[4])
    p1.add_argument("--center_fractions", type=float, nargs="+", default=[0.08])
    p1.add_argument("--volume_limit", type=int, default=0, help="evaluate only the first N volumes (sorted)")
    p1.add_argument("--determinism_volumes", type=int, default=3,
                    help="re-run this many volumes and compare (0 disables)")
    p1.add_argument("--determinism_tol", type=float, default=1e-5)
    p1.add_argument("--zf_min_margin", type=float, default=0.05,
                    help="model SSIM must exceed zero-filled SSIM by at least this")
    p1.add_argument("--image_volumes", type=int, default=2, help="log example images for this many volumes")
    p1.add_argument("--log_every", type=int, default=10)
    p1.add_argument("--num_workers", type=int, default=4)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    # architecture: read from the checkpoint when one is loaded; these only
    # take effect with --random_init (harness smoke tests)
    for k, v in RELEASED_ARCH.items():
        p1.add_argument(f"--{k}", type=int, default=v)
    p1.add_argument("--random_init", action="store_true",
                    help="skip checkpoint loading, use a random model (harness smoke test only)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return tier0(args) if args.cmd == "tier0" else tier1(args)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
