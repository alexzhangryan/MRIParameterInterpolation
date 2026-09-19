#!/bin/bash
# submit.sh: submit a DPI VarNet training job on the CHTC access point.
#
#   ./submit.sh                            2x/4x/6x/8x brain, from scratch
#                                          W&B run "dpi mixed acceleration brain"
#   ./submit.sh name="dpi mixed acceleration brain v2"   pick the W&B run name
#   ./submit.sh init=../training/runs/brain/model2/<Cluster>/checkpoints/last.ckpt
#                                          warm start both parameter sets from
#                                          the trained baseline
#   ./submit.sh resume=runs/brain/dpi/<Cluster>/checkpoints/last.ckpt
#   ./submit.sh max_epochs=1 extra_args="--limit_train_batches 50 --limit_val_batches 10"
#   OFFLINE=1 ./submit.sh                  log offline instead of holding on a bad key
#
# Any further name=value pair is passed straight to condor_submit as a macro
# override and wins over the preset. Deliberately the same shape as
# ../training/submit.sh.
set -euo pipefail
cd "$(dirname "$0")"

# ../.env is the repo-root secrets file, the single one shared by
# training/, verification/ and this directory: the W&B key and the NYU URLs.
# The file wins over the shell. ENV_FILE= points at a different one.
ENV_FILE="${ENV_FILE:-../.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ENV_FILE"
  set +a
fi

SUB=train.sub
DATASET=brain
NAME=""
USER_RUN_NAME=""
INIT=""
PASS=()
for kv in "$@"; do
  case "$kv" in
    dataset=*)  DATASET="${kv#dataset=}"; PASS+=("$kv") ;;
    name=*)     NAME="${kv#name=}" ;;
    run_name=*) USER_RUN_NAME="${kv#run_name=}"; PASS+=("$kv") ;;
    init=*)     INIT="${kv#init=}" ;;
    *)          PASS+=("$kv") ;;
  esac
done
# ${arr[@]+"${arr[@]}"}: under set -u, bash < 4.4 treats an empty array
# expansion as an unbound variable.
set -- ${PASS[@]+"${PASS[@]}"}

slug() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'; }

# The naming convention when no name= is given, alongside the baseline's
# "mixed acceleration brain".
DEFAULT_NAME="dpi mixed acceleration $DATASET"
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

PRESET=(model=dpi accelerations="2 4 6 8" center_fractions="0.16 0.08 0.0533 0.04"
        ${NAME_MACROS[@]+"${NAME_MACROS[@]}"})
[ -n "${OFFLINE:-}" ] && PRESET+=(allow_offline=1)

mkdir -p logs "runs/$DATASET/dpi"

if ! command -v condor_submit >/dev/null 2>&1; then
  echo "condor_submit not found: run this on ap2001.chtc.wisc.edu, not on the laptop" >&2
  exit 1
fi
if grep -q CHANGE_ME "$SUB"; then
  echo "edit the image macro in $SUB before submitting" >&2
  exit 1
fi
chmod +x ../training/run_train.sh 2>/dev/null || true

for f in ../training/train_wandb.py ../training/run_train.sh train_dpi.py dpi_varnet.py dpi_module.py dpi_transforms.py; do
  [ -f "$f" ] || { echo "missing $f: this job transfers it into the sandbox" >&2; exit 1; }
done

# --- the warm-start checkpoint ----------------------------------------------
# ../training/run_train.sh moves any *.ckpt it finds into output/checkpoints
# and resumes from it -- except one named baseline_*.ckpt, which it leaves
# alone. An initialisation is not a resume point, so the file is staged under
# that name here.
if [ -n "$INIT" ]; then
  if [ ! -f "$INIT" ]; then echo "init checkpoint not found: $INIT" >&2; exit 1; fi
  if [ "$(basename "$INIT")" != "baseline_init.ckpt" ]; then
    echo "staging $INIT as baseline_init.ckpt (an init, not a resume point)"
    cp -f "$INIT" baseline_init.ckpt
  fi
  PRESET+=(init=baseline_init.ckpt)
fi

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
which is the single .env for training/, verification/ and dpi/; or export it
in this shell; or run  OFFLINE=1 ./submit.sh ...  to log offline instead.
Without either, a job whose W&B key does not work HOLDS itself rather than
training for days with no observability.
EOF
  exit 1
fi
echo "W&B key: from $( [ -f "$ENV_FILE" ] && grep -q '^WANDB_API_KEY=.\{2,\}' "$ENV_FILE" && echo "$ENV_FILE" || echo "the shell environment" )"

# --- preflight: prove the preset survived into the job ad --------------------
# condor_submit parses a command-line name=value as if it were the first line
# of the submit file, so an unguarded assignment there would silently win.
# -dry-run builds the job ad locally and never contacts the schedd.
DRY="$(mktemp)"
trap 'rm -f "$DRY"' EXIT
condor_submit -dry-run "$DRY" "$SUB" "${PRESET[@]}" "$@" >/dev/null
ARGS_LINE="$(grep -m1 -E '^(Args|Arguments) *=' "$DRY" || true)"
echo "job args: ${ARGS_LINE#*=}"
grep -E '^Request(Cpus|Memory|Disk|GPUs) *=' "$DRY" | tr '\n' ' '; echo
grep -E '^(Environment|TransferInput) *=' "$DRY" | cut -c1-200

fail=""
case "$ARGS_LINE" in *"--run_name $WANT_NAME "*) ;; *) fail="$fail --run_name $WANT_NAME" ;; esac
case " $* " in *" accelerations="*) ;; *)
  case "$ARGS_LINE" in *"--accelerations 2 4 6 8 "*) ;; *) fail="$fail --accelerations 2 4 6 8" ;; esac ;;
esac
# The whole point of this submit file: the DPI driver must be the one that runs.
case "$(grep -m1 -E '^Environment *=' "$DRY" || true)" in
  *TRAIN_DRIVER=train_dpi.py*) ;;
  *) fail="$fail TRAIN_DRIVER=train_dpi.py" ;;
esac
if [ -n "$fail" ]; then
  cat >&2 <<EOF
refusing to submit: the preset did not reach the job ad.
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
echo "W&B run:     '${NAME:-$WANT_NAME}'  id=$WANT_NAME  project=${WANDB_PROJECT:-fastmri-varnet-train}${OFFLINE:+  (OFFLINE=1: logging offline)}"
echo "watch:       condor_q -nobatch; tail -f logs/${DATASET}_dpi_*.out"
echo "checkpoints: runs/$DATASET/dpi/<Cluster>/checkpoints/last.ckpt  (resume=... to continue)"
