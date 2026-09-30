#!/usr/bin/env python3
"""Paired per-rate comparison of two scored runs, straight from W&B.

Implements VERIFICATION.md section 6.3: volume difficulty dominates the
variance in these metrics, so two runs are compared with a paired test over
volumes rather than by differencing aggregates. Reports the median paired
difference and its bootstrap interval alongside the mean-of-means, per that
section, plus the zero-filled column that Phase C block A asks for.

Reads the `R<N>/per_volume` tables that `verify_varnet.py` logs, so it needs
no run outputs on disk and no `wandb` package - GraphQL plus numpy only.

Usage
-----
    python verification/paired_from_wandb.py \
        --baseline verify-model2-brain --treatment verify-dpi-brain

    # a block B variant against the same baseline
    python verification/paired_from_wandb.py \
        --baseline verify-model2-brain --treatment verify-dpi-io-brain \
        --markdown

The W&B key is read from --env, defaulting to the repo-root `.env` and
falling back to `verification/.env`; WANDB_API_KEY in the environment wins.

Caveat this script cannot address: a paired test over volumes controls volume
difficulty and mask seed, NOT training-seed variance. Until block D measures
sigma across seeds, anything here under roughly 2 sigma stays provisional
(VERIFICATION.md section 6.2).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import urllib.request

import numpy as np

GRAPHQL = "https://api.wandb.ai/graphql"
DEFAULT_ENTITY = "alexzhangryan-uw-madison"
DEFAULT_PROJECT = "fastmri-varnet-verify"
RATES = ("R2", "R4", "R6", "R8", "mixed")
METRICS = (("ssim", 6, True), ("psnr", 3, True), ("nmse", 6, False))

_FILES_Q = """query($e:String!,$p:String!,$n:String!){
  project(name:$p, entityName:$e){ run(name:$n){
    files(first:500){ edges{ node{ name directUrl } } } } } }"""


def read_key(env_path: str | None) -> str:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [env_path] if env_path else [
        os.path.join(here, os.pardir, ".env"),
        os.path.join(here, ".env"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            for line in open(c, encoding="utf-8", errors="replace"):
                line = line.strip()
                if line.startswith("WANDB_API_KEY="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        return v
    sys.exit("no WANDB_API_KEY in environment or %s" % (candidates,))


class WB:
    def __init__(self, key: str, entity: str, project: str):
        self.hdr = {
            "Authorization": "Basic " + base64.b64encode(("api:" + key).encode()).decode(),
            "Content-Type": "application/json",
            "User-Agent": "paired_from_wandb/1",
        }
        self.entity, self.project = entity, project

    def _gql(self, query: str, variables: dict) -> dict:
        req = urllib.request.Request(
            GRAPHQL, data=json.dumps({"query": query, "variables": variables}).encode(),
            headers=self.hdr)
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode())

    def per_volume_tables(self, run: str) -> dict:
        d = self._gql(_FILES_Q, {"e": self.entity, "p": self.project, "n": run})
        node = ((d.get("data") or {}).get("project") or {}).get("run")
        if not node:
            sys.exit("run %r not found in %s/%s" % (run, self.entity, self.project))
        out = {}
        for e in (node.get("files") or {}).get("edges", []):
            name = e["node"]["name"]
            if name.startswith("media/table/") and "per_volume" in name:
                out[name.split("/")[2]] = e["node"]["directUrl"]
        if not out:
            sys.exit("run %r logged no R<N>/per_volume tables" % run)
        return out

    @staticmethod
    def table(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "paired_from_wandb/1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())


def wilcoxon(diff) -> tuple[float, float, int] | None:
    """Two-sided Wilcoxon signed-rank, normal approximation with tie correction.

    Zero differences are dropped (Wilcoxon's standard handling). Returns
    (z, p, n_nonzero), or None when too few non-zero pairs remain.
    """
    d = np.asarray([x for x in np.asarray(diff, float) if x != 0.0], float)
    n = d.size
    if n < 10:
        return None
    absd = np.abs(d)
    order = np.argsort(absd, kind="mergesort")
    absd, sign = absd[order], np.sign(d[order])
    ranks = np.empty(n, float)
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0  # mean of 1-based ranks
        run = j - i + 1
        if run > 1:
            ties += run ** 3 - run
        i = j + 1
    w = float(ranks[sign > 0].sum())
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - ties / 48.0
    if var <= 0:
        return None
    z = (w - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return z, p, n


def boot_ci(diff, stat="median", iters=8000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    draws = rng.choice(d, size=(iters, d.size), replace=True)
    vals = np.median(draws, axis=1) if stat == "median" else draws.mean(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compare(wb: WB, baseline: str, treatment: str) -> dict:
    tb, tt = wb.per_volume_tables(baseline), wb.per_volume_tables(treatment)
    results = {}
    for rate in RATES:
        if rate not in tb or rate not in tt:
            continue
        B, T = wb.table(tb[rate]), wb.table(tt[rate])
        kb, kt = B["columns"].index("fname"), T["columns"].index("fname")
        rec = {}
        for met, _nd, higher_better in METRICS:
            if met not in B["columns"] or met not in T["columns"]:
                continue
            ib, it = B["columns"].index(met), T["columns"].index(met)
            mb = {r[kb]: r[ib] for r in B["data"]}
            mt = {r[kt]: r[it] for r in T["data"]}
            zf = {}
            if "zf_" + met in B["columns"]:
                iz = B["columns"].index("zf_" + met)
                zf = {r[kb]: r[iz] for r in B["data"]}
            common = sorted(k for k in mb if k in mt)
            if not common:
                continue
            d = np.array([mt[k] - mb[k] for k in common], float)
            w = wilcoxon(d)
            rec[met] = dict(
                n=len(common),
                base=float(np.mean([mb[k] for k in common])),
                treat=float(np.mean([mt[k] for k in common])),
                zf=(float(np.mean([zf[k] for k in common])) if zf else None),
                mean=float(d.mean()), mean_ci=boot_ci(d, "mean"),
                median=float(np.median(d)), median_ci=boot_ci(d, "median"),
                p=(w[1] if w else None),
                # direction-aware: "wins" always means the TREATMENT is better,
                # so for a lower-is-better metric (nmse) a negative diff wins.
                wins=int((d > 0).sum() if higher_better else (d < 0).sum()),
                losses=int((d < 0).sum() if higher_better else (d > 0).sum()))
        if rec:
            results[rate] = rec
    return results


def check_monotone(results: dict) -> list[str]:
    """Block A gate: SSIM must fall monotonically from 2x to 8x for both models."""
    lines = []
    rates = [r for r in ("R2", "R4", "R6", "R8") if r in results and "ssim" in results[r]]
    for label, key in (("baseline", "base"), ("treatment", "treat")):
        vals = [results[r]["ssim"][key] for r in rates]
        ok = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        lines.append("%-10s %s  monotone=%s" % (
            label, " > ".join("%.6f" % v for v in vals), ok))
    return lines


def render(results: dict, baseline: str, treatment: str, markdown: bool) -> None:
    print("baseline  = %s" % baseline)
    print("treatment = %s" % treatment)
    print()
    print("SSIM monotonicity (block A gate):")
    for line in check_monotone(results):
        print("  " + line)
    print()
    for met, nd, higher_better in METRICS:
        print("### %s%s" % (met.upper(), "" if higher_better else " (lower is better)"))
        if markdown:
            print("| rate | zero-filled | baseline | treatment | mean diff [95% CI] "
                  "| median diff [95% CI] | treatment wins | p |")
            print("|---|---|---|---|---|---|---|---|")
        for rate in RATES:
            if rate not in results or met not in results[rate]:
                continue
            x = results[rate][met]
            zf = ("%.*f" % (nd, x["zf"])) if x["zf"] is not None else "-"
            if x["p"] is None:
                p = "-"
            elif x["p"] < 1e-12:
                p = "~0"
            else:
                p = "%.1e" % x["p"]
            fmt = "%+." + str(nd) + "f"
            row = (rate, zf, nd, x["base"], nd, x["treat"],
                   fmt % x["mean"], fmt % x["mean_ci"][0], fmt % x["mean_ci"][1],
                   fmt % x["median"], fmt % x["median_ci"][0], fmt % x["median_ci"][1],
                   x["wins"], x["wins"] + x["losses"], p)
            if markdown:
                print("| %s | %s | %.*f | %.*f | %s [%s, %s] | %s [%s, %s] | %d/%d | %s |" % row)
            else:
                print("  %-6s zf=%-10s base=%.*f treat=%.*f  mean=%s [%s,%s]  "
                      "median=%s [%s,%s]  wins=%d/%d  p=%s" % row)
        print()
    print("Reminder (VERIFICATION.md 6.2): this pairing controls volume difficulty and")
    print("mask seed, not training-seed variance. Until block D measures sigma across")
    print("seeds, effects under roughly 2 sigma are provisional.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default="verify-model2-brain")
    ap.add_argument("--treatment", default="verify-dpi-brain")
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--env", default=None, help="path to a .env holding WANDB_API_KEY")
    ap.add_argument("--markdown", action="store_true", help="emit markdown tables")
    ap.add_argument("--json", dest="json_out", default=None, help="also write raw stats here")
    args = ap.parse_args()

    wb = WB(read_key(args.env), args.entity, args.project)
    results = compare(wb, args.baseline, args.treatment)
    if not results:
        sys.exit("no rates in common between the two runs")
    render(results, args.baseline, args.treatment, args.markdown)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print("\nraw stats -> %s" % args.json_out)


if __name__ == "__main__":
    main()
