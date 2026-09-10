#!/bin/bash
# run_verify.sh: HTCondor job executable. Runs inside the container on the
# execute node with the job scratch directory as cwd.
#
#   run_verify.sh tier0 [extra verify_varnet.py args]
#   run_verify.sh tier1 [extra verify_varnet.py args]
#
# Expects, in cwd (delivered by transfer_input_files):
#   verify_varnet.py
#   knee_leaderboard_state_dict.pt            (tier1)
#   knee_multicoil_val*.tar.xz  or  *.tar     (tier1)
# Produces results/ which verify.sub transfers back.
#
# Environment (set by verify.sub from the submit shell, never hardcoded here):
#   WANDB_API_KEY   optional. Absent -> W&B offline, run saved under results/wandb
#   WANDB_PROJECT   default fastmri-varnet-verify
#   WANDB_ENTITY    optional
#   KEEP_TARBALL    optional. Set to 1 to keep the tarball after extraction (local runs)

set -uo pipefail

TIER="${1:-tier1}"
shift || true

mkdir -p results data

# Keep every W&B / cache write inside scratch. CHTC containers do not have a
# writable home, and nothing here should depend on one.
export WANDB_DIR="$PWD/results"
export WANDB_CACHE_DIR="$PWD/.cache/wandb"
export WANDB_CONFIG_DIR="$PWD/.config/wandb"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/mpl"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[run_verify] WANDB_API_KEY not set: W&B will run offline (sync later from results/wandb)"
  export WANDB_MODE=offline
fi

echo "[run_verify] host=$(hostname) tier=$TIER image=${VERIFY_IMAGE:-?} cwd=$PWD"
echo "[run_verify] job=${_CONDOR_JOB_AD:-n/a}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "[run_verify] no nvidia-smi"
df -h . | tail -1

COMMON=(--output_dir results --fastmri_repo "${FASTMRI_REPO:-/opt/fastMRI}")

if [ "$TIER" = "tier0" ]; then
  python verify_varnet.py tier0 "${COMMON[@]}" --run_pytest "$@"
  RC=$?
else
  # ---- data ---------------------------------------------------------------
  DATA_DIR=""
  if [ -d data/multicoil_val ]; then
    DATA_DIR="data/multicoil_val"
  else
    TARBALL="$(ls -1 knee_multicoil_val*.tar.xz knee_multicoil_val*.tar 2>/dev/null | head -n1 || true)"
    if [ -z "$TARBALL" ]; then
      echo "[run_verify] ERROR: no knee_multicoil_val tarball and no data/multicoil_val directory" >&2
      exit 3
    fi
    echo "[run_verify] extracting $TARBALL ($(du -h "$TARBALL" | cut -f1)) with $(nproc) threads"
    T0=$(date +%s)
    case "$TARBALL" in
      *.xz) xz -dc -T0 "$TARBALL" | tar -x -C data ;;
      *)    tar -xf "$TARBALL" -C data ;;
    esac
    RC=$?
    if [ $RC -ne 0 ]; then echo "[run_verify] ERROR: extraction failed ($RC)" >&2; exit 3; fi
    if [ -z "${KEEP_TARBALL:-}" ]; then
      rm -f "$TARBALL"   # reclaim scratch before inference (set KEEP_TARBALL=1 when running locally)
    fi
    echo "[run_verify] extracted in $(( $(date +%s) - T0 ))s"
    DATA_DIR="$(find data -maxdepth 3 -type d -name multicoil_val | head -n1)"
  fi
  N_H5=$(ls -1 "$DATA_DIR"/*.h5 2>/dev/null | wc -l)
  echo "[run_verify] data: $DATA_DIR ($N_H5 volumes)"
  if [ "$N_H5" -eq 0 ]; then echo "[run_verify] ERROR: no .h5 files in $DATA_DIR" >&2; exit 3; fi

  # ---- checkpoint -----------------------------------------------------------
  SD="knee_leaderboard_state_dict.pt"
  DL=()
  if [ ! -f "$SD" ]; then
    echo "[run_verify] $SD not staged, will download from dl.fbaipublicfiles.com"
    DL=(--download_state_dict)
  fi

  python verify_varnet.py tier1 "${COMMON[@]}" \
      --data_path "$DATA_DIR" --state_dict "$SD" "${DL[@]}" "$@"
  RC=$?
fi

echo "[run_verify] exit code $RC"
ls -la results
# results/ must exist and be non-empty for transfer_output_files even on failure
[ -n "$(ls -A results)" ] || echo "no output produced, rc=$RC" > results/EMPTY
exit $RC
