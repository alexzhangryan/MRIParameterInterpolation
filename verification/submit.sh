#!/bin/bash
# submit.sh: submit a verification job on the CHTC access point.
#
#   ./submit.sh                       Tier 1, full val, 4x and 8x
#   ./submit.sh tier0                 Tier 0 only
#   ./submit.sh volume_limit=20       Tier 1 on the first 20 volumes
#   ./submit.sh data=osdf:///chtc/staging/a/apryan3/fastmri/knee_multicoil_val_synthetic.tar request_disk=20GB
#   OFFLINE=1 ./submit.sh             skip the W&B key check, log offline
#
# Any name=value pairs are passed straight to condor_submit as macro overrides.
set -euo pipefail
cd "$(dirname "$0")"

SUB=verify.sub
if [ "${1:-}" = "tier0" ]; then SUB=verify_tier0.sub; shift; fi

mkdir -p logs runs

if grep -q CHANGE_ME "$SUB"; then
  echo "edit the image macro in $SUB before submitting" >&2
  exit 1
fi
if [ ! -x run_verify.sh ]; then chmod +x run_verify.sh; fi

if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${OFFLINE:-}" ]; then
  cat >&2 <<'EOF'
WANDB_API_KEY is not exported in this shell. Either
    export WANDB_API_KEY=...      (from wandb.ai/authorize, freshly rotated)
or  OFFLINE=1 ./submit.sh          to log offline and `wandb sync` later.
The key is passed to the job via HTCondor getenv and is never written to disk here.
EOF
  exit 1
fi

echo "submitting $SUB $*"
condor_submit "$SUB" "$@"
echo
echo "watch:   condor_q; tail -f logs/*.out"
echo "results: runs/<Cluster>/tier*_report.json, runs/<Cluster>/results.csv"
