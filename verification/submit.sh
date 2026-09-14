#!/bin/bash
# submit.sh: submit a VarNet evaluation job on the CHTC access point.
# Same interface as ../training/submit.sh: the first argument picks the model
# preset, any further name=value pairs are passed straight to condor_submit as
# macro overrides and win over the preset.
#
#   ./submit.sh model1                     acceleration 4 only
#   ./submit.sh model2                     accelerations 2 4 6 8, each scored over the whole val set
#   ./submit.sh model1 ckpt=runs/model1/<Cluster>/checkpoints/last.ckpt    score a checkpoint you trained
#   ./submit.sh model1 volume_limit=20 extra_args="--num_workers 0"
#   ./submit.sh tier0                      environment checks only, no data (verify_tier0.sub)
#   OFFLINE=1 ./submit.sh model1           skip the W&B key check, log offline
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-}"
SUB=verify.sub
case "$MODEL" in
  model1) shift; PRESET=(model=model1 accelerations="4"       center_fractions="0.08") ;;
  model2) shift; PRESET=(model=model2 accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04") ;;
  tier0)  shift; PRESET=(model=tier0); SUB=verify_tier0.sub ;;
  *) echo "usage: $0 {model1|model2|tier0} [name=value ...]" >&2; exit 2 ;;
esac

# the remap target's parent must exist before HTCondor writes output/ there
mkdir -p logs "runs/$MODEL"

if ! command -v condor_submit >/dev/null 2>&1; then
  echo "condor_submit not found: run this on ap2001.chtc.wisc.edu, not on the laptop" >&2
  exit 1
fi
if grep -q CHANGE_ME "$SUB"; then
  echo "edit the image macro in $SUB before submitting" >&2
  exit 1
fi
if [ ! -x run_verify.sh ]; then chmod +x run_verify.sh; fi

for kv in "$@"; do
  case "$kv" in
    ckpt=file://*|ckpt=osdf://*) ;;   # a /staging URL, checked by HTCondor at transfer time
    ckpt=*)
      ckpt="${kv#ckpt=}"
      if [ ! -f "$ckpt" ]; then echo "checkpoint not found: $ckpt" >&2; exit 1; fi
      ;;
  esac
done

if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${OFFLINE:-}" ]; then
  cat >&2 <<'EOF'
WANDB_API_KEY is not exported in this shell. Either
    export WANDB_API_KEY=...      (from wandb.ai/authorize, freshly rotated)
or  OFFLINE=1 ./submit.sh ...      to log offline and `wandb sync` later.
The key is passed to the job via HTCondor getenv and is never written to disk here.
EOF
  exit 1
fi

echo "submitting $SUB ${PRESET[*]} $*"
condor_submit "$SUB" "${PRESET[@]}" "$@"
echo
echo "watch:   condor_q -nobatch; tail -f logs/${MODEL}_*.out"
echo "results: runs/$MODEL/<Cluster>/tier1_report.json, runs/$MODEL/<Cluster>/results.csv"
