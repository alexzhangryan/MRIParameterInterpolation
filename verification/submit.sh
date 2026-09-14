#!/bin/bash
# submit.sh: submit a verification job on the CHTC access point.
#
#   ./submit.sh                       Tier 1, full val, 4x and 8x
#   ./submit.sh model1                released weights, 4x,          on the val subset
#   ./submit.sh model2                released weights, 2x/4x/6x/8x, on the val subset
#   ./submit.sh tier0                 Tier 0 only
#   ./submit.sh volume_limit=20       Tier 1 on the first 20 volumes
#   ./submit.sh model2 volume_limit=5 preset, then override one macro
#   ./submit.sh data=osdf:///chtc/staging/a/apryan3/fastmri/knee_multicoil_val_synthetic.tar request_disk=20GB
#   OFFLINE=1 ./submit.sh             skip the W&B key check, log offline
#
# model1 / model2 mirror training/submit.sh model1 / model2 exactly -- same
# accelerations, same center fractions, same mask family, same val subset,
# same seed -- but evaluate the released NYU leaderboard checkpoint instead of
# training anything. Together with the two training runs they are the 2x2 in
# README "Pretrained vs. from-scratch".
#
# The first argument picks the preset; any further name=value pairs are passed
# straight to condor_submit as macro overrides and win over the preset.
set -euo pipefail
cd "$(dirname "$0")"

# ../.env (gitignored, see ../.env.example) is the single secrets file shared
# by training/ and verification/: the W&B key and the NYU presigned URLs. The
# file wins over the shell, so leave a line out of .env if you would rather
# export that variable by hand. ENV_FILE= points at a different one.
ENV_FILE="${ENV_FILE:-../.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ENV_FILE"
  set +a
fi

SUB=verify.sub
PRESET=()
MODEL=tier1
WANT_ACCEL=""
WANT_MASK=""
WANT_VAL=""

case "${1:-}" in
  tier0)
    SUB=verify_tier0.sub; shift
    ;;
  model1)
    shift
    MODEL=model1
    WANT_ACCEL="4"; WANT_MASK=equispaced_fraction
    WANT_VAL=knee_multicoil_val_subset.tar
    PRESET=(model=model1 accelerations="$WANT_ACCEL" center_fractions="0.08"
            mask_type=$WANT_MASK val_file=$WANT_VAL request_disk=60GB)
    ;;
  model2)
    shift
    MODEL=model2
    WANT_ACCEL="2 4 6 8"; WANT_MASK=equispaced_fraction
    WANT_VAL=knee_multicoil_val_subset.tar
    PRESET=(model=model2 accelerations="$WANT_ACCEL" center_fractions="0.16 0.08 0.0533 0.04"
            mask_type=$WANT_MASK val_file=$WANT_VAL request_disk=60GB)
    ;;
esac

# the remap target's parent must exist before HTCondor writes results/ there
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

if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${OFFLINE:-}" ]; then
  cat >&2 <<'EOF'
WANDB_API_KEY is not exported in this shell. Either
    export WANDB_API_KEY=...      (from wandb.ai/authorize, freshly rotated)
or  OFFLINE=1 ./submit.sh          to log offline and `wandb sync` later.
The key is passed to the job via HTCondor getenv and is never written to disk here.
EOF
  exit 1
fi

# --- preflight: prove the preset survived into the job ad --------------------
# condor_submit parses a command-line `name=value` "as if it was placed at the
# beginning of the submit description file", so any UNGUARDED assignment in
# verify.sub wins over the preset above -- silently. That is the bug that cost
# three hours on the training side on 2026-09-13 (see training/submit.sh);
# verify.sub now wraps every default in `if ! defined`, and this is what keeps
# it that way. -dry-run builds the job ad locally and never contacts the schedd.
if [ "$SUB" = verify.sub ]; then
  DRY="$(mktemp)"
  trap 'rm -f "$DRY"' EXIT
  condor_submit -dry-run "$DRY" "$SUB" ${PRESET[@]+"${PRESET[@]}"} "$@" >/dev/null
  ARGS_LINE="$(grep -m1 -E '^(Args|Arguments) *=' "$DRY" || true)"
  XFER_LINE="$(grep -m1 -E '^TransferInput *=' "$DRY" || true)"
  echo "job args: ${ARGS_LINE#*=}"
  grep -E '^Request(Cpus|Memory|Disk|GPUs) *=' "$DRY" | tr '\n' ' '; echo

  # Only assert on knobs the caller did not override by hand.
  fail=""
  case " $* " in *" run_name="*) ;; *)
    case "$ARGS_LINE" in *"--run_name $MODEL-"*) ;; *) fail="$fail --run_name $MODEL-<Cluster>" ;; esac ;;
  esac
  if [ -n "$WANT_ACCEL" ]; then
    case " $* " in *" accelerations="*) ;; *)
      case "$ARGS_LINE" in *"--accelerations $WANT_ACCEL "*) ;; *) fail="$fail --accelerations $WANT_ACCEL" ;; esac ;;
    esac
    case " $* " in *" mask_type="*) ;; *)
      case "$ARGS_LINE" in *"--mask_type $WANT_MASK "*) ;; *) fail="$fail --mask_type $WANT_MASK" ;; esac ;;
    esac
  fi
  if [ -n "$fail" ]; then
    cat >&2 <<EOF
refusing to submit: the $MODEL preset did not reach the job ad.
  expected:$fail
  got:      ${ARGS_LINE#*=}
A plain assignment in $SUB is overwriting the command-line macro. Wrap that
line in 'if ! defined <name> / ... / endif' (every other default there is).
EOF
    exit 1
  fi
  # The val subset is the whole point of the model presets: silently falling
  # back to the 199-volume tarball is both wrong and hours of extra GPU time.
  if [ -n "$WANT_VAL" ]; then
    case " $* " in *" val_file="*|*" data="*) ;; *)
      case "$XFER_LINE" in
        *"$WANT_VAL"*) ;;
        *) echo "refusing to submit: $WANT_VAL is not in TransferInput" >&2
           echo "  got: ${XFER_LINE#*=}" >&2; exit 1 ;;
      esac ;;
    esac
  fi
fi

echo "submitting $SUB ${PRESET[*]-} $*"
condor_submit "$SUB" ${PRESET[@]+"${PRESET[@]}"} "$@"
echo
echo "watch:   condor_q; tail -f logs/${MODEL}_*.out"
echo "results: runs/$MODEL/<Cluster>/tier1_report.json, runs/$MODEL/<Cluster>/results.csv"
