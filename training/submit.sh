#!/bin/bash
# submit.sh: submit a VarNet training job on the CHTC access point.
#
#   ./submit.sh model2                     accelerations 2 4 6 8, one drawn per sample (brain);
#                                          W&B run "mixed acceleration brain" by convention
#   ./submit.sh model2 name="mixed acceleration brain v2"     pick the W&B run name (id = its slug)
#   ./submit.sh model1                     acceleration 4 only (brain); W&B run varnet-brain-model1
#   ./submit.sh model2 resume=runs/brain/<model>/<Cluster>/checkpoints/last.ckpt   (same name -> same W&B run)
#   ./submit.sh model2 max_epochs=2 extra_args="--limit_train_batches 20 --limit_val_batches 5"
#   ./submit.sh model2 dataset=knee staging=... train_data=... val_data=...   the 2026-09 knee subsets
#   OFFLINE=1 ./submit.sh model2           skip the W&B key check, log offline
#
# The first argument picks the model preset; any further name=value pairs are
# passed straight to condor_submit as macro overrides and win over the preset.
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
case "$MODEL" in
  model1|model2) shift ;;
  *) echo "usage: $0 {model2|model1} [name=value ...]" >&2; exit 2 ;;
esac

SUB=train.sub
# Parameters read here (everything else is passed to condor_submit verbatim):
#   dataset=<d>   train.sub macro (default brain); the runs/ directory, the
#                 default run name and the preflight follow it.
#   name="..."    the W&B run name, spaces allowed (2026-09-18: "mixed
#                 acceleration brain"). Becomes wandb_name (displayed) and
#                 run_name (the W&B id: the name lower-cased, anything that is
#                 not a letter, digit, . _ or - turned into -). Not a train.sub
#                 macro, so it is consumed here.
#   run_name=<id> sets the id directly, no display-name change; wins over both.
DATASET=brain
NAME=""
USER_RUN_NAME=""
PASS=()
for kv in "$@"; do
  case "$kv" in
    dataset=*)  DATASET="${kv#dataset=}"; PASS+=("$kv") ;;
    name=*)     NAME="${kv#name=}" ;;
    run_name=*) USER_RUN_NAME="${kv#run_name=}"; PASS+=("$kv") ;;
    *)          PASS+=("$kv") ;;
  esac
done
# ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": under set -u, bash < 4.4
# treats an EMPTY array expansion as an unbound variable.
set -- ${PASS[@]+"${PASS[@]}"}

slug() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'; }

# The naming convention when no name= is given. model2 is the multi-rate run:
# "mixed acceleration <dataset>" / mixed-acceleration-<dataset>. model1 keeps
# train.sub's derived default. Both are new ids, so neither can continue the
# 2026-09 knee runs (varnet-model1 / varnet-model2).
case "$MODEL" in
  model1) DEFAULT_NAME="varnet-$DATASET-model1" ;;
  model2) DEFAULT_NAME="mixed acceleration $DATASET" ;;
esac
if [ -n "$USER_RUN_NAME" ]; then
  WANT_NAME="$USER_RUN_NAME"
  NAME_MACROS=()
  [ -n "$NAME" ] && NAME_MACROS=(wandb_name="$NAME")
else
  [ -n "$NAME" ] || NAME="$DEFAULT_NAME"
  WANT_NAME="$(slug "$NAME")"
  [ -n "$WANT_NAME" ] || { echo "name='$NAME' has no usable characters for a W&B run id" >&2; exit 2; }
  NAME_MACROS=(run_name="$WANT_NAME" wandb_name="$NAME")
fi

case "$MODEL" in
  model1) PRESET=(model=model1 accelerations="4"       center_fractions="0.08"                  ${NAME_MACROS[@]+"${NAME_MACROS[@]}"}) ;;
  model2) PRESET=(model=model2 accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04" ${NAME_MACROS[@]+"${NAME_MACROS[@]}"}) ;;
esac
# OFFLINE=1 is the only sanctioned way to run without a working W&B
# connection; it becomes WANDB_ALLOW_OFFLINE=1 inside the job. Without it a
# missing or rejected key HOLDS the job (train.sub on_exit_hold).
[ -n "${OFFLINE:-}" ] && PRESET+=(allow_offline=1)
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
  cat >&2 <<EOF
WANDB_API_KEY is not set. It is read from the repo-root secrets file
    $ENV_FILE   (cp ../.env.example ../.env; chmod 600 ../.env; paste the key)
which is the single .env for training/ and verification/; or export it in this
shell; or run  OFFLINE=1 ./submit.sh ...  to log offline and \`wandb sync\` later.
The key is passed to the job via HTCondor getenv and is never written to disk here.
EOF
  exit 1
fi
echo "W&B key: from $( [ -f "$ENV_FILE" ] && grep -q '^WANDB_API_KEY=.\{2,\}' "$ENV_FILE" && echo "$ENV_FILE" || echo "the shell environment" )"

# --- preflight: prove the preset survived into the job ad --------------------
# condor_submit parses `name=value` from the command line as if it were at the
# TOP of the submit file, so any unguarded assignment in train.sub wins over
# the preset above. That failure is silent and expensive: on 2026-09-13 a
# `model2` submission trained acceleration 4 and logged into the varnet-model1
# W&B run for three hours before anyone noticed. train.sub now guards its
# defaults with `if ! defined`; this check is what keeps it that way.
# -dry-run builds the job ad locally and never contacts the schedd.
DRY="$(mktemp)"
trap 'rm -f "$DRY"' EXIT
condor_submit -dry-run "$DRY" "$SUB" "${PRESET[@]}" "$@" >/dev/null
ARGS_LINE="$(grep -m1 -E '^(Args|Arguments) *=' "$DRY" || true)"
echo "job args: ${ARGS_LINE#*=}"
# Print the resource requests as HTCondor actually resolved them. request_cpus
# and request_memory are pre-seeded in the macro table, so a mistake there does
# not raise an error, it just quietly asks for 1 CPU (see train.sub).
grep -E '^Request(Cpus|Memory|Disk|GPUs) *=' "$DRY" | tr '\n' ' '; echo

# Only assert on knobs the caller did not override by hand (the run name is
# always asserted: it is either the caller's, the name= slug, or the convention).
want_name="$WANT_NAME"
want_accel="$(printf '%s' "${PRESET[1]}" | sed 's/^accelerations=//')"
fail=""
case "$ARGS_LINE" in *"--run_name $want_name "*) ;; *) fail="$fail --run_name $want_name" ;; esac
case " $* " in *" accelerations="*) ;; *)
  case "$ARGS_LINE" in *"--accelerations $want_accel "*) ;; *) fail="$fail --accelerations $want_accel" ;; esac ;;
esac
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
echo "W&B run:     '${NAME:-$want_name}'  id=$want_name  project=${WANDB_PROJECT:-fastmri-varnet-train}${OFFLINE:+  (OFFLINE=1: logging offline, no hold on a bad key)}"
echo "watch:       condor_q -nobatch; tail -f logs/${DATASET}_${MODEL}_*.out"
echo "checkpoints: runs/$DATASET/$MODEL/<Cluster>/checkpoints/last.ckpt  (resume=... to continue)"
