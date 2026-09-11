#!/bin/bash
# prepare_staging.sh: one-time data preparation on CHTC.
#
# Puts the two Tier 1 inputs into /staging/a/apryan3/fastmri/ :
#   knee_multicoil_val.tar.xz          (93.8 GB, from the NYU presigned URL)
#   knee_leaderboard_state_dict.pt     (released checkpoint, ~120 MB)
# and verifies the tarball against NYU's SHA256 manifest.
#
# Usage (on transfer.chtc.wisc.edu):
#   export FASTMRI_VAL_URL='https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_val.tar.xz?AWSAccessKeyId=...'
#   export FASTMRI_SHA_URL='https://fastmri-dataset.s3.amazonaws.com/v3.0/SHA256?AWSAccessKeyId=...'
#   ./prepare_staging.sh
#
# The presigned URLs come from the NYU email and expire ~90 days after issue
# (the current batch around 2026-12-08). Paste them from the email; do not
# commit them. curl -C - resumes a partial download, so re-running after an
# interruption is safe, and files that are already complete are skipped.
#
# CHTC notes:
#   - Run this on transfer.chtc.wisc.edu, the host CHTC provides for bulk data
#     movement, not on the access point. Both mount /staging; the access point
#     is not meant to carry 94 GB of traffic.
#   - Run it under nohup or tmux: 94 GB takes hours.
#   - Set PARALLEL low (1 to 4) on CHTC. The 16-way ranged fetch below exists
#     to beat per-flow shaping on a dorm ISP; from campus to S3 a single
#     stream is already fast and 16 flows only load a shared host. See the
#     parallel_fetch comment for the measurements behind that.
#   - Do not scp a local copy up instead. The same shaping that motivated
#     parallel_fetch applies to outbound traffic, so a 94 GB upload takes days
#     where an S3-to-campus fetch takes hours.
#   - Personal /staging quota defaults to 100 GB / 1000 items. The 93.8 GB
#     tarball plus the ~120 MB checkpoint fits with about 5 GB to spare.
#     multicoil_train (~931 GB unpacked) does not: that needs a quota increase
#     or access to the kamilov_group staging directory.
set -euo pipefail

NETID="${NETID:-${USER:-unknown}}"
# CHTC personal staging is sharded by the first letter of the netid:
# /staging/a/apryan3, not /staging/apryan3. STAGING_DIR overrides the whole
# path -- a local directory off-cluster, or the group directory if access to
# /staging/groups/kamilov_group/Kamilov-SciAI-datasets is granted later.
DEST="${STAGING_DIR:-/staging/${NETID:0:1}/$NETID/fastmri}"
CKPT_URL="https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt"

: "${FASTMRI_VAL_URL:?export FASTMRI_VAL_URL from the NYU email first}"
: "${FASTMRI_SHA_URL:?export FASTMRI_SHA_URL from the NYU email first}"

mkdir -p "$DEST"
cd "$DEST"
echo "staging into $DEST"
df -h "$DEST" | tail -1

# Safe to re-run: a file whose size already matches the server is skipped, a
# partial one is resumed. Size is read with a 1-byte ranged GET because S3
# presigned URLs are signed for GET only and reject HEAD.
local_size()  { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }
remote_size() {
  curl -sS -L -r 0-0 -o /dev/null -D - "$1" 2>/dev/null | awk -F/ 'tolower($0) ~ /^content-range:/ {v=$2; gsub(/[^0-9]/, "", v); print v}' | tail -1
}
fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2" have want
  if [ -s "$dest" ]; then
    have="$(local_size "$dest")"; want="$(remote_size "$url")"
    if [ -n "$want" ] && [ "$have" = "$want" ]; then
      echo "   $dest already complete ($have bytes), skipping"
      return 0
    fi
    echo "   $dest is partial ($have of ${want:-?} bytes), resuming"
  fi
  curl -C - "$url" --output "$dest"
}

# Parallel ranged download, used for the tarball only.
#
# The ISP here (ResTech AS46262) shapes every TCP flow to ~2 Mbit/s for
# off-net traffic: a single stream to S3, or to an unrelated host in Newark,
# meters out at 0.24 MB/s, while CDN content cached on-net pulls 47 MB/s on
# one connection. Aggregate capacity is fine -- 28 MB/s measured at 192
# flows -- so the fix is many flows, not a faster one. On a network without
# per-flow shaping PARALLEL=1 is the right setting and this all collapses
# back to a plain ranged curl.
#
# Each worker pipes its range into the output file at that byte offset, so
# there is one output file and no reassembly pass. A chunk is marked done
# only after its curl exits 0, so an interrupted run refetches just the
# chunks that were in flight.
PARALLEL="${PARALLEL:-16}"
CHUNK_MB="${CHUNK_MB:-16}"

parallel_fetch() {  # parallel_fetch <url> <dest>
  local url="$1" dest="$2"
  local total chunk nchunks state meta sig i st en pass running ndone

  total="$(remote_size "$url")"
  case "$total" in
    "" | *[!0-9]* ) echo "could not read a valid remote size for $dest (got: '$total')" >&2; return 1 ;;
  esac
  [ "$total" -gt 0 ] || { echo "remote size for $dest is zero" >&2; return 1; }

  chunk=$(( CHUNK_MB * 1024 * 1024 ))
  nchunks=$(( (total + chunk - 1) / chunk ))
  state="$dest.parts"
  meta="$dest.meta"
  sig="$total/$chunk"

  # One run at a time. Two concurrent runs share $state, and whichever
  # finishes first deletes it out from under the other; worse, runs with
  # different CHUNK_MB write incompatible marker indices into it, so a
  # resuming run skips chunks it never fetched and leaves holes.
  FETCH_LOCK="$dest.lock"
  if ! mkdir "$FETCH_LOCK" 2>/dev/null; then
    echo "another run is already fetching $dest" >&2
    echo "if no container is running (docker ps), the lock is stale: rm -rf '$FETCH_LOCK'" >&2
    return 1
  fi
  trap 'if [ -n "${FETCH_LOCK:-}" ]; then rm -rf "$FETCH_LOCK"; fi' EXIT INT TERM

  if [ -s "$dest" ] && [ ! -d "$state" ] && [ "$(local_size "$dest")" = "$total" ]; then
    echo "   $dest already complete ($total bytes), skipping"
    rm -rf "$FETCH_LOCK"
    return 0
  fi

  # Marker index N means a different byte range under a different chunk size,
  # so markers are only meaningful for the layout that produced them.
  if [ -f "$meta" ] && [ "$(cat "$meta")" != "$sig" ]; then
    echo "   chunk layout changed ($(cat "$meta") -> $sig): discarding stale markers"
    rm -rf "$state"
  fi
  mkdir -p "$state"
  echo "$sig" > "$meta"

  # The bind mount does not do sparse files, so this allocates the full size
  # up front. It also repairs a file left larger than the remote by an
  # earlier bad run -- truncate sets the exact length in both directions.
  if [ ! -f "$dest" ] || [ "$(local_size "$dest")" != "$total" ]; then
    truncate -s "$total" "$dest"
  fi

  echo "   $total bytes, $nchunks x ${CHUNK_MB}M chunks, $PARALLEL at a time"
  ndone=$(ls -1 "$state" | wc -l)
  [ "$ndone" -gt 0 ] && echo "   resuming: $ndone chunks already done"

  # Progress is reported from the dispatch loop, time-gated. It deliberately
  # does NOT run as a background job: `wait` with no arguments waits for every
  # child, so a `while true` monitor would make the end-of-pass wait hang
  # forever.
  pstart=$(date +%s)
  pbase=$ndone
  plast=$pstart

  for pass in 1 2 3; do
    running=0
    for (( i = 0; i < nchunks; i++ )); do
      [ -f "$state/$i" ] && continue
      st=$(( i * chunk ))
      en=$(( st + chunk - 1 ))
      [ "$en" -ge "$total" ] && en=$(( total - 1 ))
      # count_bytes caps the write at the range length: if the server ever
      # ignored Range and replied 200, an uncapped dd would splatter the
      # whole 94 GB at this offset.
      ( if curl -sS --fail --retry 5 --retry-delay 3 --retry-connrefused -r "${st}-${en}" "$url" | dd of="$dest" bs=1M seek="$st" oflag=seek_bytes iflag=fullblock,count_bytes count=$(( en - st + 1 )) conv=notrunc status=none; then touch "$state/$i"; fi ) &
      running=$(( running + 1 ))
      if [ "$running" -ge "$PARALLEL" ]; then
        wait -n || true
        running=$(( running - 1 ))
        pnow=$(date +%s)
        if [ $(( pnow - plast )) -ge 10 ]; then
          plast=$pnow
          ndone=$(ls -1 "$state" | wc -l)
          awk -v d="$ndone" -v b="$pbase" -v n="$nchunks" -v cm="$CHUNK_MB" -v el="$(( pnow - pstart ))" 'BEGIN{ got=(d-b)*cm; rate=(el>0 ? got/el : 0); eta=(rate>0 ? (n-d)*cm/rate : 0); printf "   %d/%d chunks  %.1f%%  %.1f GB  %.2f MB/s  eta %dh%02dm\n", d, n, d*100/n, d*cm/1024, rate, int(eta/3600), int(eta%3600/60) }'
        fi
      fi
    done
    wait || true
    ndone=$(ls -1 "$state" | wc -l)
    [ "$ndone" -eq "$nchunks" ] && break
    echo "   $(( nchunks - ndone )) chunks failed on pass $pass, retrying them"
  done

  ndone=$(ls -1 "$state" | wc -l)
  [ "$ndone" -eq "$nchunks" ] || { echo "$(( nchunks - ndone )) chunks still missing after 3 passes; re-run to resume" >&2; rm -rf "$FETCH_LOCK"; return 1; }
  [ "$(local_size "$dest")" = "$total" ] || { echo "size is $(local_size "$dest"), expected $total" >&2; rm -rf "$FETCH_LOCK"; return 1; }
  rm -rf "$state" "$meta" "$FETCH_LOCK"
  echo "   $dest complete ($total bytes)"
}

echo "== SHA256 manifest"
curl -sS "$FASTMRI_SHA_URL" --output SHA256
grep -c . SHA256 >/dev/null || { echo "SHA256 manifest is empty" >&2; exit 1; }

echo "== released checkpoint"
fetch "$CKPT_URL" knee_leaderboard_state_dict.pt
ls -la knee_leaderboard_state_dict.pt

echo "== knee_multicoil_val.tar.xz (this is the long step)"
parallel_fetch "$FASTMRI_VAL_URL" knee_multicoil_val.tar.xz

echo "== verifying"
if grep -q "knee_multicoil_val.tar.xz" SHA256; then
  grep "knee_multicoil_val.tar.xz" SHA256 | sha256sum -c -
else
  echo "knee_multicoil_val.tar.xz not listed in the manifest; computing the hash for the record"
  sha256sum knee_multicoil_val.tar.xz | tee knee_multicoil_val.tar.xz.sha256
fi

echo
echo "done. In verify.sub set:"
echo "  staging = $DEST"
echo "and the data/ckpt paths will resolve to file://$DEST/..."
ls -la "$DEST"
