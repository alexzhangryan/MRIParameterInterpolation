#!/bin/bash
# submit.sh: submit a VarNet training job on the CHTC access point.
#
#   ./submit.sh model1                     acceleration 4 only
#   ./submit.sh model2                     accelerations 2 4 6 8, one drawn per sample
#   ./submit.sh model1 resume=runs/<model>/<Cluster>/checkpoints/last.ckpt
#   ./submit.sh model2 max_epochs=2 extra_args="--limit_train_batches 20 --limit_val_batches 5"
#   OFFLINE=1 ./submit.sh model1           skip the W&B key check, log offline
#
# The first argument picks the model preset; any further name=value pairs are
# passed straight to condor_submit as macro overrides and win over the preset.
set -euo pipefail
cd "$(dirname "$0")"

# .env (gitignored, see .env.example) holds the W&B key and the NYU URL.
# The file wins over the shell: leave a line out of .env if you would rather
# export that variable by hand.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

MODEL="${1:-}"
case "$MODEL" in
  model1) shift; PRESET=(model=model1 accelerations="4"       center_fractions="0.08") ;;
  model2) shift; PRESET=(model=model2 accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04") ;;
  *) echo "usage: $0 {model1|model2} [name=value ...]" >&2; exit 2 ;;
esac

SUB=train.sub
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
if [ ! -x run_train.sh ]; then chmod +x run_train.sh; fi

for kv in "$@"; do
  case "$kv" in
    resume=*)
      ckpt="${kv#resume=}"
      if [ ! -f "$ckpt" ]; then echo "resume checkpoint not found: $ckpt" >&2; exit 1; fi
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
echo "watch:       condor_q -nobatch; tail -f logs/${MODEL}_*.out"
echo "checkpoints: runs/<model>/<Cluster>/checkpoints/last.ckpt  (resume=... to continue)"
