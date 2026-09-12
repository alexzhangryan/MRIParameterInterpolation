#!/bin/bash
# run_train.sh: HTCondor job executable. Runs inside the container on the
# execute node with the job scratch directory as cwd.
#
#   run_train.sh [train_wandb.py args]
#
# Expects, in cwd (delivered by transfer_input_files):
#   train_wandb.py
#   knee_multicoil_train*.tar.xz | *.tar     training split
#   knee_multicoil_val*.tar.xz   | *.tar     validation split
#   *.ckpt                                   optional: a checkpoint to resume from
#   output/                                  optional: brought back by HTCondor after an
#                                            eviction (when_to_transfer_output = ON_EXIT_OR_EVICT)
# Produces output/ (checkpoints/, wandb/, lightning logs) which train.sub
# transfers back to runs/<Cluster>/.
#
# Environment (set by train.sub from the submit shell, never hardcoded here):
#   WANDB_API_KEY   optional. Absent -> W&B offline, run saved under output/wandb
#   WANDB_PROJECT   default fastmri-varnet-train
#   WANDB_ENTITY    optional
#   KEEP_TARBALL    optional. Set to 1 to keep tarballs after extraction (local runs)

set -uo pipefail

mkdir -p output/checkpoints data/knee

# Keep every W&B / cache write inside scratch. CHTC containers do not have a
# writable home, and nothing here should depend on one.
export WANDB_DIR="$PWD/output"
export WANDB_CACHE_DIR="$PWD/.cache/wandb"
export WANDB_CONFIG_DIR="$PWD/.config/wandb"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/mpl"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[run_train] WANDB_API_KEY not set: W&B will run offline (sync later from output/wandb)"
  export WANDB_MODE=offline
fi

echo "[run_train] host=$(hostname) image=${TRAIN_IMAGE:-?} cwd=$PWD"
echo "[run_train] job=${_CONDOR_JOB_AD:-n/a}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "[run_train] no nvidia-smi"
df -h . | tail -1

# ---- resume checkpoint ------------------------------------------------------
# Two ways a checkpoint can already be here: HTCondor brought output/ back
# after an eviction, or the submitter passed resume=<file> (lands in cwd).
# The evicted output/ is newer by construction, so a staged file is only
# used when output/checkpoints is still empty. train_wandb.py then picks the
# newest checkpoint in output/checkpoints itself (last.ckpt preferred).
if [ -z "$(ls -A output/checkpoints)" ]; then
  for f in ./*.ckpt; do
    [ -f "$f" ] || continue
    echo "[run_train] seeding output/checkpoints with $f"
    mv "$f" output/checkpoints/
  done
else
  echo "[run_train] output/checkpoints already populated (evicted job resumed by HTCondor):"
  ls -la output/checkpoints
fi

# ---- data -------------------------------------------------------------------
# Extract every tarball in cwd into data/, then expose the two split
# directories under data/knee/ whatever depth the archives put them at.
extract() {  # extract <tarball>
  echo "[run_train] extracting $1 ($(du -h "$1" | cut -f1)) with $(nproc) threads"
  local t0; t0=$(date +%s)
  case "$1" in
    *.xz) xz -dc -T0 "$1" | tar -x -C data ;;
    *)    tar -xf "$1" -C data ;;
  esac
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[run_train] ERROR: extraction of $1 failed ($rc)" >&2; exit 3; fi
  echo "[run_train] extracted $1 in $(( $(date +%s) - t0 ))s"
  if [ -z "${KEEP_TARBALL:-}" ]; then rm -f "$1"; fi   # reclaim scratch before training
}

shopt -s nullglob
TARBALLS=(knee_multicoil_*.tar.xz knee_multicoil_*.tar)
shopt -u nullglob
if [ ${#TARBALLS[@]} -eq 0 ] && [ ! -d data/knee/multicoil_train ]; then
  echo "[run_train] ERROR: no knee_multicoil_*.tar(.xz) in cwd and no data/knee/multicoil_train" >&2
  exit 3
fi
for t in "${TARBALLS[@]}"; do extract "$t"; done

for split in multicoil_train multicoil_val; do
  if [ ! -d "data/knee/$split" ]; then
    found="$(find data -mindepth 1 -maxdepth 4 -type d -name "$split" -not -path "data/knee/*" | head -n1)"
    if [ -z "$found" ]; then
      echo "[run_train] ERROR: no $split directory found after extraction" >&2
      find data -maxdepth 3 -type d | head -50 >&2
      exit 3
    fi
    ln -s "$PWD/$found" "data/knee/$split"
  fi
  n=$(ls -1 "data/knee/$split"/*.h5 2>/dev/null | wc -l)
  echo "[run_train] $split: $n volumes ($(readlink -f "data/knee/$split"))"
  if [ "$n" -eq 0 ]; then echo "[run_train] ERROR: no .h5 files in $split" >&2; exit 3; fi
done

# ---- train ------------------------------------------------------------------
# Eviction handling. HTCondor vacates a job by sending SIGTERM to the executable
# and SIGKILLing the whole thing after MachineMaxVacateTime. Everything below
# exists so the cleanup at the bottom -- the symlink sweep in particular --
# always runs before that hard kill, because output transfer happens on eviction
# too (when_to_transfer_output = ON_EXIT_OR_EVICT) and a surviving symlink holds
# the job. `make job-evict` tests exactly this.
#
# Two facts drive the shape of it, both measured rather than assumed:
#
#  1. pytorch-lightning 1.9.5 installs NO SIGTERM handler here, so python takes
#     the default disposition and dies at once; the observed exit code is 143
#     (128 + SIGTERM), not a clean 0. The grace loop therefore almost always
#     finishes in about a second. It is kept as a bounded safety net for an
#     in-flight torch.save, and in case a future torch/Lightning does trap the
#     signal -- never as an unbounded wait, which would run past the SIGKILL and
#     skip the cleanup entirely.
#  2. `wait` returns immediately with status >128 when a trapped signal arrives,
#     WITHOUT reaping the child. A single bare `wait` would let this script exit
#     while python was still running and writing into output/. Hence the re-wait
#     loop below.
#
# Keeping python a CHILD of this script rather than the job executable is
# load-bearing for (1): the kernel discards an unhandled signal sent to PID 1,
# so a container running python directly would ignore the vacate notice and
# train on until the SIGKILL. Verified both ways.
#
# Nothing is lost by python dying here: the checkpoint resumed from is the one
# ModelCheckpoint wrote at the last completed validation epoch, already on disk.
VACATE_GRACE="${VACATE_GRACE:-20}"
PY=

on_term() {
  echo "[run_train] SIGTERM received (HTCondor eviction notice)"
  [ -n "$PY" ] || return 0
  kill -TERM "$PY" 2>/dev/null
  local i=0
  while [ "$i" -lt "$VACATE_GRACE" ]; do
    kill -0 "$PY" 2>/dev/null || { echo "[run_train] python exited after ${i}s"; return 0; }
    sleep 1
    i=$(( i + 1 ))
  done
  echo "[run_train] python still running ${VACATE_GRACE}s after SIGTERM (expected: PL 1.9 ignores it), sending SIGKILL"
  kill -KILL "$PY" 2>/dev/null
}
trap on_term TERM INT

python train_wandb.py --data_path data/knee --default_root_dir output "$@" &
PY=$!
wait "$PY"; RC=$?
while [ "$RC" -gt 128 ] && kill -0 "$PY" 2>/dev/null; do wait "$PY"; RC=$?; done
trap - TERM INT
echo "[run_train] exit code $RC"
# HTCondor cannot transfer a symlink to a directory and wandb creates
# output/wandb/latest-run -> run-<id>, which holds the job on output
# transfer (ON_EXIT_OR_EVICT, so on eviction too). Drop every symlink
# under output/ before the sandbox goes back; the real run dir stays.
find output -type l -delete
ls -la output output/checkpoints
# output/ must exist and be non-empty for transfer_output_files even on failure
[ -n "$(ls -A output)" ] || echo "no output produced, rc=$RC" > output/EMPTY
exit $RC
