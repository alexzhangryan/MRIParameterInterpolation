#!/usr/bin/env python
"""
divergence.py: how far did the two parameter sets of a trained DPI VarNet
actually move apart, and where?

DPI runs the network on lam(R) * weight + (1 - lam(R)) * weight_copy. At
initialisation the two sets are identical, so the network is rate-blind no
matter what lam is; conditioning only exists to the extent that training
pulled `weight` and `weight_copy` apart. This script measures that, per
tensor and grouped by the same layer families `--dpi_scope` selects, from a
checkpoint alone -- no data, no GPU, seconds on a laptop.

Two questions it answers before any further GPU time is spent:

  1. Did DPI collapse to the baseline?  If the relative divergence is ~0
     everywhere, the "conditioned" network is the blind one with twice the
     parameters, and lam is decorative. (Symptom in W&B: lambda/R4, lambda/R6
     flat after epoch 1.)
  2. Which layers specialised?  If the divergence concentrates in the first
     ConvBlock, the final 1x1 and dc_weight, that is the `io` scope measured
     rather than hypothesised, and the variant to run next. If it is spread
     evenly, no narrow scope will reproduce `full`.

    python divergence.py runs/brain/dpi/<Cluster>/checkpoints/last.ckpt
    make divergence CKPT=...          (inside the training image)

The checkpoint is a Lightning one from train_dpi.py (keys under `varnet.`),
about 700 MB for the real model: scp it from the access point first.
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

# --- layer families, matching unet_dpi_flags / the README scope table --------
# The up lists are deepest-first, so U-Net level 0 is the LAST index; the
# family names below are by level so they read like the scope table.
_RE = {
    "dc_weight": re.compile(r"^cascades\.\d+\.dc_weight$"),
    "down": re.compile(r"^(?P<net>cascades\.\d+\.model|sens_net\.norm_unet)\.unet\.down_sample_layers\.(?P<k>\d+)\."),
    "bottleneck": re.compile(r"^(?P<net>cascades\.\d+\.model|sens_net\.norm_unet)\.unet\.conv\."),
    "up": re.compile(r"^(?P<net>cascades\.\d+\.model|sens_net\.norm_unet)\.unet\.up_(?:transpose_)?conv\.(?P<k>\d+)\.(?P<sub>\d+\.)?"),
}


def family(key: str, n_up: int) -> str:
    """Name the layer family a base-tensor key belongs to."""
    sens = key.startswith("sens_net.")
    prefix = "sens " if sens else ""
    if _RE["dc_weight"].match(key):
        return "dc_weight"
    m = _RE["down"].match(key)
    if m:
        return f"{prefix}down L{int(m['k'])}"
    if _RE["bottleneck"].match(key):
        return f"{prefix}bottleneck"
    m = _RE["up"].match(key)
    if m:
        k = int(m["k"])
        level = n_up - 1 - k
        if level == 0 and "up_conv." in key and m["sub"] == "1.":
            return f"{prefix}final 1x1"
        return f"{prefix}up L{level}"
    return f"{prefix}other"


SCOPE_OF = {  # which families each scope duplicates (cascade side; sens mirrors)
    "io": {"down L0", "final 1x1", "dc_weight"},
    "shallow": {"down L0", "down L1", "up L0", "up L1", "final 1x1", "dc_weight"},
    "dc": {"dc_weight"},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", type=Path)
    p.add_argument("--top", type=int, default=12, help="tensors to list individually")
    args = p.parse_args()

    raw = torch.load(str(args.ckpt), map_location="cpu")
    hp = dict(raw.get("hyper_parameters") or {})
    sd = raw["state_dict"] if "state_dict" in raw else raw
    sd = {k[len("varnet."):]: v for k, v in sd.items() if k.startswith("varnet.")} or sd

    if "lambda_table.phi" not in sd:
        raise SystemExit(f"{args.ckpt}: no lambda_table.phi -- not a DPI checkpoint")

    n_up = 1 + max(int(m.group(1)) for k in sd for m in [re.search(r"cascades\.0\.model\.unet\.up_conv\.(\d+)\.", k)] if m)

    # --- lambda at the trained rates ----------------------------------------
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dpi_varnet import LambdaTable

    table = LambdaTable(length=int(hp.get("lambda_length", sd["lambda_table.phi"].numel())),
                        accel_min=float(hp.get("accel_min", 2.0)), accel_max=float(hp.get("accel_max", 8.0)),
                        spacing=str(hp.get("lambda_spacing", "log")))
    with torch.no_grad():
        table.phi.copy_(sd["lambda_table.phi"])
        rates = [float(r) for r in hp.get("log_accelerations", (2, 4, 6, 8))]
        lam = table(torch.tensor(rates)).tolist()
    print(f"checkpoint: {args.ckpt}  epoch={raw.get('epoch')}  dpi_scope={hp.get('dpi_scope', 'full (pre-flag)')}")
    print("lambda(R): " + "  ".join(f"R{int(r)}={l:.3f}" for r, l in zip(rates, lam)))
    print()

    # --- per-tensor divergence -----------------------------------------------
    rows = []
    for k, w in sd.items():
        if k.endswith("_copy") or k == "lambda_table.phi":
            continue
        wc = sd.get(k + "_copy")
        if wc is None:
            continue
        w, wc = w.float().flatten(), wc.float().flatten()
        d = w - wc
        rel = float(d.norm() / (w.norm() + 1e-12))
        cos = float(torch.dot(w, wc) / (w.norm() * wc.norm() + 1e-12))
        rows.append(dict(key=k, family=family(k, n_up), numel=w.numel(),
                         rel=rel, cos=cos, sq=float(d.pow(2).sum()), wsq=float(w.pow(2).sum())))
    if not rows:
        raise SystemExit("no *_copy tensors: this checkpoint has dpi_scope none")

    total_sq = sum(r["sq"] for r in rows)
    total_wsq = sum(r["wsq"] for r in rows)
    print(f"{len(rows)} duplicated tensors, {sum(r['numel'] for r in rows):,} parameters per set")
    print(f"overall relative divergence ||W - Wc|| / ||W|| = {math.sqrt(total_sq / total_wsq):.4f}")
    if total_sq == 0.0:
        print("  the two sets are IDENTICAL: an untrained checkpoint, or training never separated them")
    print()
    # shares are of the total movement; with none, report 0 rather than divide
    share_of = (lambda sq: sq / total_sq) if total_sq else (lambda sq: 0.0)

    # --- by family -----------------------------------------------------------
    fam = defaultdict(lambda: dict(n=0, numel=0, sq=0.0, wsq=0.0, rels=[]))
    for r in rows:
        f = fam[r["family"]]
        f["n"] += 1
        f["numel"] += r["numel"]
        f["sq"] += r["sq"]
        f["wsq"] += r["wsq"]
        f["rels"].append(r["rel"])

    def order(name):
        base = name.replace("sens ", "")
        rank = {"down": 0, "bottleneck": 1, "up": 2, "final 1x1": 3, "dc_weight": 4}
        key = next((v for k, v in rank.items() if base.startswith(k)), 9)
        return (name.startswith("sens "), key, base)

    print(f"{'family':16} {'tensors':>7} {'params':>12} {'rel div':>8} {'share of':>9}  {'in scope'}")
    print(f"{'':16} {'':>7} {'':>12} {'':>8} {'movement':>9}")
    for name in sorted(fam, key=order):
        f = fam[name]
        rel = math.sqrt(f["sq"] / f["wsq"]) if f["wsq"] else 0.0
        share = share_of(f["sq"])
        base = name.replace("sens ", "")
        scopes = [s for s, fams in SCOPE_OF.items() if base in fams]
        print(f"{name:16} {f['n']:>7} {f['numel']:>12,} {rel:>8.4f} {share:>8.1%}  {' '.join(scopes) or 'full only'}")

    # what fraction of the total movement each narrow scope would have kept
    print()
    print("movement captured by each scope (share of sum ||W - Wc||^2 over the tensors it duplicates):")
    for s, fams in SCOPE_OF.items():
        kept = sum(fam[n]["sq"] for n in fam if n.replace("sens ", "") in fams)
        params = sum(fam[n]["numel"] for n in fam if n.replace("sens ", "") in fams)
        print(f"  {s:8} {share_of(kept):>7.1%} of the movement, {params:>12,} of {sum(r['numel'] for r in rows):,} duplicated parameters")

    # --- top tensors -----------------------------------------------------------
    print()
    print(f"top {args.top} tensors by relative divergence:")
    for r in sorted(rows, key=lambda r: -r["rel"])[: args.top]:
        print(f"  {r['rel']:.4f}  cos={r['cos']:.5f}  {r['numel']:>9,}  {r['key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
