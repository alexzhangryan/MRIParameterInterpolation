#!/bin/bash
# make_subset.sh: HTCondor job executable that builds a subset tarball of one
# fastMRI knee split, so both splits fit under the 100 GB /staging quota.
# Runs inside the training container (xz, curl, tar, python+h5py) on a CPU
# slot. Submitted by subset_val.sub / subset_train.sub; see README section 3b.
#
#   make_subset.sh val
#       Input:  knee_multicoil_val.tar.xz in cwd (transferred from /staging)
#       Output: knee_multicoil_val_subset.tar        plain tar, multicoil_val/<N_VAL files>
#               val_subset_files.txt                 which volumes, and the tar's sha256
#       N_VAL volumes (default 20), evenly spaced through the sorted file list
#       so both PD and PDFS scans are represented.
#
#   make_subset.sh train
#       Input:  FASTMRI_TRAIN_URL (env, via getenv): the NYU presigned URL for
#               knee_multicoil_train_batch_0.tar.xz (the split ships as five
#               ~91 GB batches; batch 0 is the one used)
#       Output: knee_multicoil_train_subset.tar.xz   multicoil_train/<every complete volume>
#               train_subset_files.txt
#       The batch is never stored in full anywhere. The first
#       TRAIN_PREFIX_GB (default 65) gigabytes of it are fetched with an HTTP
#       range request and decoded on the fly; xz and tar are sequential, so a
#       prefix of the archive yields a prefix of the volumes, complete except
#       for the last one, which is dropped. The same TRAIN_PREFIX_GB gives the
#       same volumes every time. The complete volumes are re-packed with xz so
#       the result is a well-formed archive of about TRAIN_PREFIX_GB again.
#
# Every kept .h5 is opened with h5py and fully read once. Anything that fails
# (a truncated tail file, a bad download) is deleted before packing.

set -uo pipefail
MODE="${1:?usage: make_subset.sh val|train}"
N_VAL="${N_VAL:-20}"
TRAIN_PREFIX_GB="${TRAIN_PREFIX_GB:-65}"

echo "[make_subset] mode=$MODE host=$(hostname) cwd=$PWD cpus=$(nproc)"
df -h . | tail -1
mkdir -p data

# validate <file>...: fully read each h5 file, delete the ones that fail
validate() {
  python - "$@" <<'EOF'
import os, sys, h5py
ok, bad = 0, []
for p in sys.argv[1:]:
    try:
        with h5py.File(p, "r") as f:
            k = f["kspace"]
            for i in range(k.shape[0]):
                k[i]
            if "reconstruction_rss" in f:
                f["reconstruction_rss"][()]
            _ = f.attrs["max"]
        ok += 1
    except Exception as e:
        bad.append(p)
        print(f"[make_subset] BAD {p}: {e}")
for p in bad:
    os.remove(p)
print(f"[make_subset] validated {ok} volumes, removed {len(bad)}")
EOF
}

case "$MODE" in
  val)
    src_tar=knee_multicoil_val.tar.xz
    [ -f "$src_tar" ] || { echo "[make_subset] ERROR: $src_tar not in cwd" >&2; exit 3; }
    echo "[make_subset] extracting $src_tar ($(du -h "$src_tar" | cut -f1))"
    t0=$(date +%s)
    xz -dc -T0 "$src_tar" | tar -x -C data
    rc=$?; [ $rc -eq 0 ] || { echo "[make_subset] ERROR: extraction failed ($rc)" >&2; exit 3; }
    rm -f "$src_tar"
    echo "[make_subset] extracted in $(( $(date +%s) - t0 ))s"

    split=multicoil_val
    src="$(find data -type d -name "$split" | head -n1)"
    [ -n "$src" ] || { echo "[make_subset] ERROR: no $split directory after extraction" >&2; exit 3; }
    mapfile -t all < <(ls "$src"/*.h5 | sort)
    total=${#all[@]}
    [ "$total" -ge "$N_VAL" ] || { echo "[make_subset] ERROR: only $total volumes, wanted $N_VAL" >&2; exit 3; }
    step=$(( total / N_VAL ))
    keep=()
    for ((i = 0; i < N_VAL; i++)); do keep+=("${all[i * step]}"); done
    echo "[make_subset] keeping $N_VAL of $total volumes (every ${step}th)"
    validate "${keep[@]}"
    # rebuild the list from what survived validation
    keep=()
    for ((i = 0; i < N_VAL; i++)); do [ -f "${all[i * step]}" ] && keep+=("${all[i * step]}"); done

    out=knee_multicoil_val_subset.tar
    list=val_subset_files.txt
    : > "$list"
    for f in "${keep[@]}"; do echo "$split/$(basename "$f")" >> "$list"; done
    tar -cf "$out" -C "$(dirname "$src")" -T "$list"
    rc=$?; [ $rc -eq 0 ] || { echo "[make_subset] ERROR: tar failed ($rc)" >&2; exit 3; }
    ;;

  train)
    : "${FASTMRI_TRAIN_URL:?FASTMRI_TRAIN_URL not set: export it (via .env + getenv) before submitting}"
    # TRAIN_PREFIX_BYTES exists for testing against a small local archive
    bytes="${TRAIN_PREFIX_BYTES:-$(( TRAIN_PREFIX_GB * 1000000000 ))}"
    echo "[make_subset] streaming the first ${TRAIN_PREFIX_GB} GB of the train archive from NYU: ${FASTMRI_TRAIN_URL%%\?*}"
    t0=$(date +%s)
    # xz reports "Unexpected end of input" and tar "Unexpected EOF" at the
    # cut: expected, the volumes written before it are complete. tar -v
    # prints each member as it starts extracting it, so the last name in
    # tar_members.txt is the one that was cut. (File mtimes are useless for
    # this: tar restores the archived timestamps.)
    curl -fsS -r "0-$(( bytes - 1 ))" "$FASTMRI_TRAIN_URL" | xz -dc | tar -xv -C data > tar_members.txt
    echo "[make_subset] stream ended after $(( $(date +%s) - t0 ))s (an EOF error above is expected)"

    split=multicoil_train
    src="$(find data -type d -name "$split" | head -n1)"
    [ -n "$src" ] || { echo "[make_subset] ERROR: no $split directory: did the download start? (check the URL)" >&2; exit 3; }
    last="$(grep '\.h5$' tar_members.txt | tail -n1)"
    [ -n "$last" ] || { echo "[make_subset] ERROR: no volumes decoded" >&2; exit 3; }
    echo "[make_subset] dropping the volume the cut landed in: $last"
    rm -f "data/$last" "$src/$(basename "$last")"
    mapfile -t keep < <(ls "$src"/*.h5 | sort)
    [ ${#keep[@]} -ge 1 ] || { echo "[make_subset] ERROR: no complete volumes: raise TRAIN_PREFIX_GB" >&2; exit 3; }
    validate "${keep[@]}"
    mapfile -t keep < <(ls "$src"/*.h5 | sort)
    echo "[make_subset] $(( ${#keep[@]} )) complete volumes, $(du -sh "$src" | cut -f1) on disk"

    out=knee_multicoil_train_subset.tar.xz
    list=train_subset_files.txt
    : > "$list"
    for f in "${keep[@]}"; do echo "$split/$(basename "$f")" >> "$list"; done
    echo "[make_subset] packing with xz -T$(nproc) -3"
    t0=$(date +%s)
    tar -cf - -C "$(dirname "$src")" -T "$list" | xz -T0 -3 > "$out"
    rc=$?; [ $rc -eq 0 ] || { echo "[make_subset] ERROR: pack failed ($rc)" >&2; exit 3; }
    echo "[make_subset] packed in $(( $(date +%s) - t0 ))s"
    ;;

  *) echo "usage: make_subset.sh val|train" >&2; exit 2 ;;
esac

rm -rf data   # scratch is wiped anyway; this just keeps the output listing honest
{
  echo "# prefix_gb=${TRAIN_PREFIX_GB} n_val=${N_VAL} built=$(date -u +%FT%TZ) host=$(hostname)"
  echo "# sha256 $(sha256sum "$out" | cut -d' ' -f1)  $out  $(stat -c %s "$out") bytes"
} >> "$list"
echo "[make_subset] done: $out ($(du -h "$out" | cut -f1)), $(grep -c '\.h5$' "$list") volumes"
cat "$list"
