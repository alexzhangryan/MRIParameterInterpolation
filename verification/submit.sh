#!/bin/bash
# submit.sh: submit a VarNet evaluation job on the CHTC access point.
# Same interface as ../training/submit.sh: the first argument picks the
# preset, any further name=value pairs are passed straight to condor_submit as
# macro overrides and win over the preset.
#
#   ./submit.sh model2 ckpt=../training/runs/brain/model2/<Cluster>/checkpoints/last.ckpt run_name=verify-model2-brain
#                                          trained baseline: 2x/4x/6x/8x, one pass per rate, plus the mixed pass, brain val batch 0
#   ./submit.sh model2 ckpt=../dpi/runs/brain/dpi/<Cluster>/checkpoints/last.ckpt run_name=verify-dpi-brain
#                                          a DPI checkpoint, same passes; the harness detects it
#   ./submit.sh model1                     released knee weights, 4x, on brain val batch 0 (a sanity row)
#   ./submit.sh tier1                      the Claim A sweep: random masks 4x/8x, knee full 199-volume val, reference verdicts
#   ./submit.sh tier0                      environment checks only, no data (verify_tier0.sub)
#   ./submit.sh model2 volume_limit=5 extra_args="--num_workers 0" ckpt=...
#   ./submit.sh model2 dataset=knee staging=/staging/a/apryan3/fastmri val_data=file:///staging/a/apryan3/fastmri/knee_multicoil_val_subset.tar request_disk=60GB
#   OFFLINE=1 ./submit.sh model2 ...       skip the W&B key check, log offline
#
# model1 / model2 mirror training/submit.sh model1 / model2 exactly -- same
# accelerations, same center fractions, same mask family, same val data (brain
# val batch 0 since 2026-09-18), same seed -- but score a checkpoint (the
# released NYU knee leaderboard weights by default) instead of training one.
# Together with the two training runs they are the 2x2 in README "Pretrained
# vs. from-scratch". `dataset=` picks the runs/ and logs/ subdirectory as in
# training (default brain).
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

MODEL="${1:-}"
SUB=verify.sub
WANT_ACCEL=""
WANT_MASK=""
case "$MODEL" in
  model1) shift; WANT_ACCEL="4";       WANT_MASK=equispaced_fraction
          PRESET=(model=model1 accelerations="$WANT_ACCEL" center_fractions="0.08"                  mask_type=$WANT_MASK) ;;
  model2) shift; WANT_ACCEL="2 4 6 8"; WANT_MASK=equispaced_fraction
          PRESET=(model=model2 accelerations="$WANT_ACCEL" center_fractions="0.16 0.08 0.0533 0.04" mask_type=$WANT_MASK) ;;
  # The verification sweep proper: the fastMRI knee convention the third-party
  # reference values are keyed on, over the full split. Only possible while
  # knee_multicoil_val.tar.xz is still in /staging (README "Run Tier 1 before
  # the training repack").
  tier1)  shift; WANT_ACCEL="4 8";     WANT_MASK=random
          PRESET=(model=tier1 dataset=knee accelerations="$WANT_ACCEL" center_fractions="0.08 0.04" mask_type=$WANT_MASK
                  val_data="file:///staging/a/apryan3/fastmri/knee_multicoil_val.tar.xz" request_disk=320GB) ;;
  tier0)  shift; PRESET=(model=tier0); SUB=verify_tier0.sub ;;
  *) echo "usage: $0 {model1|model2|tier1|tier0} [name=value ...]" >&2; exit 2 ;;
esac

# dataset= names the runs/<dataset>/<model>/ and logs/ subtree, as in
# training/submit.sh. The tier1 preset is knee by definition; a command-line
# dataset= wins over everything.
DATASET=brain
case "$MODEL" in tier1) DATASET=knee ;; esac
HAVE_CKPT=""
for kv in "$@"; do
  case "$kv" in
    dataset=*) DATASET="${kv#dataset=}" ;;
    ckpt=*)    HAVE_CKPT=1 ;;
  esac
done

# the remap target's parent must exist before HTCondor writes output/ there
mkdir -p logs "runs/$DATASET/$MODEL"

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

if [ -z "$HAVE_CKPT" ] && [ "$DATASET" != knee ] && [ "$MODEL" != tier0 ]; then
  echo "note: no ckpt= given, so this scores the released fastMRI KNEE checkpoint on $DATASET data." >&2
  echo "      To score a trained model pass ckpt=../training/runs/$DATASET/$MODEL/<Cluster>/checkpoints/last.ckpt" >&2
  echo "      (or ../dpi/runs/$DATASET/dpi/<Cluster>/checkpoints/last.ckpt) and a run_name=." >&2
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${OFFLINE:-}" ]; then
  cat >&2 <<'EOF'
WANDB_API_KEY is not exported in this shell. Either
    export WANDB_API_KEY=...      (from wandb.ai/authorize, freshly rotated)
or  OFFLINE=1 ./submit.sh ...      to log offline and `wandb sync` later.
The key is passed to the job via HTCondor getenv and is never written to disk here.
EOF
  exit 1
fi

# --- preflight: prove the preset survived into the job ad --------------------
# condor_submit parses `name=value` from the command line as if it were at the
# TOP of the submit file, so any unguarded assignment in verify.sub wins over
# the preset above. That failure is silent and expensive: on 2026-09-13 a
# training `model2` submission trained acceleration 4 and logged into the
# varnet-model1 W&B run for three hours before anyone noticed. verify.sub
# guards its defaults with `if ! defined`; this check is what keeps it that
# way. -dry-run builds the job ad locally and never contacts the schedd.
DRY="$(mktemp)"
trap 'rm -f "$DRY"' EXIT
condor_submit -dry-run "$DRY" "$SUB" "${PRESET[@]}" "$@" >/dev/null
ARGS_LINE="$(grep -m1 -E '^(Args|Arguments) *=' "$DRY" || true)"
echo "job args: ${ARGS_LINE#*=}"
# Print the resource requests as HTCondor actually resolved them. request_cpus
# and request_memory are pre-seeded in the macro table, so a mistake there does
# not raise an error, it just quietly asks for 1 CPU (see verify.sub).
grep -E '^Request(Cpus|Memory|Disk|GPUs) *=' "$DRY" | tr '\n' ' '; echo

# Only assert on knobs the caller did not override by hand. ARGS_CHK is the
# line with its closing quote replaced by a space, so every flag, including
# the last one on the line (tier0 has only --run_name), matches "flag value ".
want_name="verify-$MODEL"
ARGS_CHK="${ARGS_LINE%\"} "
fail=""
case " $* " in *" run_name="*) ;; *)
  case "$ARGS_CHK" in *"--run_name $want_name "*) ;; *) fail="$fail --run_name $want_name" ;; esac ;;
esac
if [ -n "$WANT_ACCEL" ]; then
  case " $* " in *" accelerations="*) ;; *)
    case "$ARGS_CHK" in *"--accelerations $WANT_ACCEL "*) ;; *) fail="$fail --accelerations $WANT_ACCEL" ;; esac ;;
  esac
  case " $* " in *" mask_type="*) ;; *)
    case "$ARGS_CHK" in *"--mask_type $WANT_MASK "*) ;; *) fail="$fail --mask_type $WANT_MASK" ;; esac ;;
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

echo "submitting $SUB ${PRESET[*]} $*"
condor_submit "$SUB" "${PRESET[@]}" "$@"
echo
echo "watch:   condor_q -nobatch; tail -f logs/${DATASET}_${MODEL}_*.out"
echo "results: runs/$DATASET/$MODEL/<Cluster>/tier1_report.json, per_volume_R<N>.csv, per_volume_mixed.csv, results.csv"
