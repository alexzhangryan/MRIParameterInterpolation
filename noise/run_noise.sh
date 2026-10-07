#!/bin/bash
# run_noise.sh: HTCondor job executable for noise.sub. CPU only.
#
#   run_noise.sh <split> [measure_noise.py args]
#
# Streams the one *.tar.xz HTCondor delivered through `xz | python` without
# extracting it (each .h5 is spooled to scratch, measured, deleted), so peak
# scratch is the tarball plus one volume. Writes output/<split>_{slices,volumes}.csv
# and output/<split>_summary.json, which noise.sub brings back to
# runs/<split>/<Cluster>/.
set -uo pipefail

SPLIT="${1:?usage: run_noise.sh <split> [args]}"
shift || true
mkdir -p output spool
export TMPDIR="$PWD/spool"

echo "[run_noise] host=$(hostname) split=$SPLIT cwd=$PWD nproc=$(nproc)"
df -h . | tail -1

shopt -s nullglob
TARBALLS=(*.tar.xz *.tar)
shopt -u nullglob
if [ ${#TARBALLS[@]} -ne 1 ]; then
  echo "[run_noise] ERROR: expected exactly one tarball in cwd, found ${#TARBALLS[@]}: ${TARBALLS[*]-}" >&2
  echo "nothing measured" > output/EMPTY
  exit 3
fi
T="${TARBALLS[0]}"
echo "[run_noise] streaming $T ($(du -h "$T" | cut -f1))"

t0=$(date +%s)
if [[ "$T" == *.xz ]]; then
  xz -dc -T0 "$T" | python measure_noise.py --split "$SPLIT" --tar_stdin --tmp_dir "$TMPDIR" --out output "$@"
  ST=("${PIPESTATUS[@]}")
  RC=${ST[1]}
  # xz dies of SIGPIPE (141) when --volume_limit makes python stop reading early; that is fine
  if [ "${ST[0]}" -ne 0 ] && [ "${ST[0]}" -ne 141 ]; then
    echo "[run_noise] ERROR: xz exited ${ST[0]}" >&2
    [ "$RC" -ne 0 ] || RC=${ST[0]}
  fi
else
  python measure_noise.py --split "$SPLIT" --tar_stdin --tmp_dir "$TMPDIR" --out output "$@" < "$T"
  RC=$?
fi
echo "[run_noise] exit code $RC after $(( $(date +%s) - t0 ))s"
ls -la output
[ -n "$(ls -A output)" ] || echo "no output produced, rc=$RC" > output/EMPTY
exit "$RC"
