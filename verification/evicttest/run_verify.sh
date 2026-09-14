#!/bin/bash
# run_verify.sh: HTCondor job executable. Runs inside the container on the
# execute node with the job scratch directory as cwd. Same structure as
# ../training/run_train.sh: the same environment block, the same extract(),
# the same data/knee/<split> layout, the same eviction handling, the same
# output/ contract.
#
#   run_verify.sh tier1 [verify_varnet.py args]     score a checkpoint on multicoil_val
#   run_verify.sh tier0 [verify_varnet.py args]     environment checks only, no data
#
# Expects, in cwd (delivered by transfer_input_files):
#   verify_varnet.py
#   knee_multicoil_val*.tar.xz | *.tar       validation split                (tier1)
#   *.pt | *.ckpt                            the checkpoint to score. Absent -> the
#                                            released fastMRI knee checkpoint is downloaded
#   output/                                  optional: brought back by HTCondor after an
#                                            eviction (when_to_transfer_output = ON_EXIT_OR_EVICT)
# Produces output/ (report JSON, per-volume CSVs, results.csv, wandb/) which
# verify.sub transfers back to runs/<model>/<Cluster>/.
#
# Environment (set by verify.sub from the submit shell, never hardcoded here):
#   WANDB_API_KEY   optional. Absent -> W&B offline, run saved under output/wandb
#   WANDB_PROJECT   default fastmri-varnet-verify
#   WANDB_ENTITY    optional
#   KEEP_TARBALL    optional. Set to 1 to keep tarballs after extraction (local runs)

set -uo pipefail

TIER="${1:-tier1}"
shift || true

mkdir -p output data/knee

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
  echo "[run_verify] WANDB_API_KEY not set: W&B will run offline (sync later from output/wandb)"
  export WANDB_MODE=offline
fi

echo "[run_verify] host=$(hostname) tier=$TIER image=${VERIFY_IMAGE:-?} cwd=$PWD"
echo "[run_verify] job=${_CONDOR_JOB_AD:-n/a}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "[run_verify] no nvidia-smi"
df -h . | tail -1

if [ "$TIER" = "tier0" ]; then
  TIER_ARGS=(--run_pytest)
else
  # ---- data -----------------------------------------------------------------
  # Extract every tarball in cwd into data/, then expose the split directory
  # under data/knee/ whatever depth the archive put it at.
  extract() {  # extract <tarball>
    echo "[run_verify] extracting $1 ($(du -h "$1" | cut -f1)) with $(nproc) threads"
    local t0; t0=$(date +%s)
    case "$1" in
      *.xz) xz -dc -T0 "$1" | tar -x -C data ;;
      *)    tar -xf "$1" -C data ;;
    esac
    local rc=$?
    if [ $rc -ne 0 ]; then echo "[run_verify] ERROR: extraction of $1 failed ($rc)" >&2; exit 3; fi
    echo "[run_verify] extracted $1 in $(( $(date +%s) - t0 ))s"
    if [ -z "${KEEP_TARBALL:-}" ]; then rm -f "$1"; fi   # reclaim scratch before inference
  }

  shopt -s nullglob
  TARBALLS=(knee_multicoil_*.tar.xz knee_multicoil_*.tar)
  shopt -u nullglob
  if [ ${#TARBALLS[@]} -eq 0 ] && [ ! -d data/knee/multicoil_val ]; then
    echo "[run_verify] ERROR: no knee_multicoil_*.tar(.xz) in cwd and no data/knee/multicoil_val" >&2
    exit 3
  fi
  for t in "${TARBALLS[@]}"; do extract "$t"; done

  for split in multicoil_val; do
    if [ ! -d "data/knee/$split" ]; then
      found="$(find data -mindepth 1 -maxdepth 4 -type d -name "$split" -not -path "data/knee/*" | head -n1)"
      if [ -z "$found" ]; then
        echo "[run_verify] ERROR: no $split directory found after extraction" >&2
        find data -maxdepth 3 -type d | head -50 >&2
        exit 3
      fi
      ln -s "$PWD/$found" "data/knee/$split"
    fi
    n=$(ls -1 "data/knee/$split"/*.h5 2>/dev/null | wc -l)
    echo "[run_verify] $split: $n volumes ($(readlink -f "data/knee/$split"))"
    if [ "$n" -eq 0 ]; then echo "[run_verify] ERROR: no .h5 files in $split" >&2; exit 3; fi
  done

  # ---- checkpoint -----------------------------------------------------------
  # Whatever verify.sub's `ckpt` macro delivered: the released state dict by
  # default, or a Lightning checkpoint from ../training. verify_varnet.py
  # reads the architecture from the file, so both load the same way.
  shopt -s nullglob
  CKPTS=()
  for f in *.pt *.ckpt; do [ -f "$f" ] && CKPTS+=("$f"); done   # regular files only
  shopt -u nullglob
  if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[run_verify] no *.pt / *.ckpt staged: will download the released checkpoint from dl.fbaipublicfiles.com"
    TIER_ARGS=(--data_path data/knee/multicoil_val --state_dict knee_leaderboard_state_dict.pt --download_state_dict)
  else
    echo "[run_verify] checkpoint: ${CKPTS[0]} ($(du -h "${CKPTS[0]}" | cut -f1))"
    TIER_ARGS=(--data_path data/knee/multicoil_val --state_dict "${CKPTS[0]}")
  fi
fi

# ---- run --------------------------------------------------------------------
# Eviction handling, copied from run_train.sh. HTCondor vacates a job by
# sending SIGTERM to the executable and SIGKILLing the whole thing after
# MachineMaxVacateTime. Everything below exists so the cleanup at the bottom
# (the symlink sweep in particular) always runs before that hard kill, because
# output transfer happens on eviction too (ON_EXIT_OR_EVICT) and a surviving
# symlink holds the job. `make job-evict` tests exactly this.
#
#  1. verify_varnet.py installs no SIGTERM handler, so python takes the default
#     disposition and dies at once (exit 143). The grace loop is a bounded
#     safety net for an in-flight write, never an unbounded wait, which would
#     run past the SIGKILL and skip the cleanup entirely.
#  2. `wait` returns immediately with status >128 when a trapped signal
#     arrives, WITHOUT reaping the child. A single bare `wait` would let this
#     script exit while python was still writing into output/. Hence the
#     re-wait loop below.
#
# Keeping python a CHILD of this script rather than the job executable is
# load-bearing for (1): the kernel discards an unhandled signal sent to PID 1.
VACATE_GRACE="${VACATE_GRACE:-20}"
PY=

on_term() {
  echo "[run_verify] SIGTERM received (HTCondor eviction notice)"
  [ -n "$PY" ] || return 0
  kill -TERM "$PY" 2>/dev/null
  local i=0
  while [ "$i" -lt "$VACATE_GRACE" ]; do
    kill -0 "$PY" 2>/dev/null || { echo "[run_verify] python exited after ${i}s"; return 0; }
    sleep 1
    i=$(( i + 1 ))
  done
  echo "[run_verify] python still running ${VACATE_GRACE}s after SIGTERM, sending SIGKILL"
  kill -KILL "$PY" 2>/dev/null
}
trap on_term TERM INT

python verify_varnet.py "$TIER" --output_dir output --fastmri_repo "${FASTMRI_REPO:-/opt/fastMRI}" \
    "${TIER_ARGS[@]}" "$@" &
PY=$!
wait "$PY"; RC=$?
while [ "$RC" -gt 128 ] && kill -0 "$PY" 2>/dev/null; do wait "$PY"; RC=$?; done
trap - TERM INT
echo "[run_verify] exit code $RC"
# HTCondor cannot transfer a symlink to a directory and wandb creates
# output/wandb/latest-run -> run-<id>, which holds the job on output
# transfer (ON_EXIT_OR_EVICT, so on eviction too). Drop every symlink
# under output/ before the sandbox goes back; the real run dir stays.
find output -type l -delete
ls -la output
# output/ must exist and be non-empty for transfer_output_files even on failure
[ -n "$(ls -A output)" ] || echo "no output produced, rc=$RC" > output/EMPTY
exit $RC
