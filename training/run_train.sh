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
# Forward SIGTERM (HTCondor's eviction notice) to Python so Lightning can stop
# cleanly; the checkpoint on disk from the last validation epoch is what gets
# resumed either way.
python train_wandb.py --data_path data/knee --default_root_dir output "$@" &
PY=$!
trap 'echo "[run_train] SIGTERM received, forwarding to python"; kill -TERM $PY 2>/dev/null' TERM INT
wait $PY
RC=$?
trap - TERM INT

echo "[run_train] exit code $RC"
# W&B leaves output/wandb/latest-run as a symlink to the run directory.
# HTCondor refuses to transfer a symlink to a directory and puts the job on
# hold instead ("Transfer of symlinks to directories is not supported"), so
# every symlink under output/ is dropped before exit. Nothing is lost: the
# run directory itself is transferred.
find output -type l -exec rm -f {} +
ls -la output output/checkpoints
# output/ must exist and be non-empty for transfer_output_files even on failure
[ -n "$(ls -A output)" ] || echo "no output produced, rc=$RC" > output/EMPTY
exit $RC
