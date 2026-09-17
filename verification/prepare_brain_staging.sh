#!/bin/bash
# prepare_brain_staging.sh: move the fastMRI BRAIN dataset into the Kamilov
# group staging directory on CHTC.
#
#   /staging/groups/kamilov_group/Kamilov-SciAI-datasets/fastMRI_brain/
#
# 20 tarballs, ~1.37 TB total, left COMPRESSED. Nothing is ever extracted:
# the group staging quota is 2.5 TB and 10,000 items, and the brain splits
# unpack to well over 100,000 .h5 files, which would blow the item quota many
# times over. That cap also drives where this script puts its own bookkeeping
# -- see the CHTC notes below.
#
# This is the brain sibling of prepare_staging.sh (knee val, personal staging)
# and reuses its remote-size and resume conventions.
#
# ---------------------------------------------------------------- usage ----
#
#   # 1. Save the NYU email's curl block to a file on the transfer node. Paste
#   #    the whole thing, knee lines included -- brain lines are selected here.
#   vi ~/fastmri_urls.txt && chmod 600 ~/fastmri_urls.txt
#
#   # 2. Dry run. Checks the quota, resolves every remote size, confirms
#   #    nothing has expired. Downloads nothing.
#   ./prepare_brain_staging.sh check
#
#   # 3. The real thing, in tmux because it runs for hours:
#   tmux new -s brain
#   ./prepare_brain_staging.sh fetch
#
#   # 4. Re-check the hashes later without re-downloading:
#   ./prepare_brain_staging.sh verify
#
# Re-running `fetch` is safe and is the intended way to recover from an
# interruption: a file whose size already matches the server is skipped and a
# partial one resumes with curl -C -.
#
# The URLs in that file embed AWSAccessKeyId and Signature, so they ARE
# credentials. chmod 600 the file, keep it out of git, and delete it when the
# transfer is done.
#
# ------------------------------------------------------------ CHTC notes ----
#
#   - Run this on transfer.chtc.wisc.edu, the host CHTC provides for bulk data
#     movement, not on ap2001. Both mount /staging; the access point is not
#     meant to carry 1.37 TB.
#   - ITEM QUOTA: the group directory is capped at 10,000 items, where an item
#     is any file OR directory. 20 tarballs plus the manifest is 22 items with
#     the directory itself. So this script keeps its per-file markers and logs
#     in $HOME, NOT in /staging. That is also why it does not reuse
#     prepare_staging.sh's chunked parallel_fetch: that function's marker
#     directory would have written ~6,000 tiny files per tarball into the very
#     directory with the item cap.
#   - NEVER extract a tarball in /staging. Jobs pull the .tar.xz into scratch
#     with transfer_input_files and unpack it there.
#   - JOBS is whole-file concurrency, not the per-flow chunking the knee
#     download needed. Campus-to-S3 is not shaped the way that dorm ISP was,
#     so a few concurrent streams saturate the link and raising JOBS mostly
#     just loads a host shared with the rest of CHTC.
#
set -euo pipefail

GROUP_STAGING="${GROUP_STAGING:-/staging/groups/kamilov_group/Kamilov-SciAI-datasets}"
DATASET_DIR="${DATASET_DIR:-fastMRI_brain}"
DEST="${DEST:-$GROUP_STAGING/$DATASET_DIR}"

URLS_FILE="${URLS_FILE:-$HOME/fastmri_urls.txt}"
JOBS="${JOBS:-3}"

# Bookkeeping lives in $HOME on purpose -- see the item-quota note above.
WORK_DIR="${WORK_DIR:-$HOME/.fastmri_brain}"
DONE_DIR="$WORK_DIR/done"
LOG_DIR="$WORK_DIR/logs"

MODE="${1:-fetch}"
case "$MODE" in
  check|fetch|verify) ;;
  *) echo "usage: $0 [check|fetch|verify]" >&2; exit 2 ;;
esac

mkdir -p "$DONE_DIR" "$LOG_DIR"

say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
human() { awk -v b="${1:-0}" 'BEGIN{ printf "%.1f GB", b/1e9 }'; }
local_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null; }

# S3 presigned URLs are signed for GET only and reject HEAD, so the length has
# to come from a 1-byte ranged GET's Content-Range: bytes 0-0/<total>.
remote_size() {
  curl -sS -L -r 0-0 -o /dev/null -D - "$1" 2>/dev/null \
    | awk -F/ 'tolower($0) ~ /^content-range:/ {v=$2; gsub(/[^0-9]/, "", v); print v}' \
    | tail -1
}

# ------------------------------------------------------------ url parsing ---
#
# The NYU email ships a block of lines shaped like
#
#   curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_0.tar.xz?AWSAccessKeyId=...&Signature=...&Expires=..." --output brain_multicoil_train_batch_0.tar.xz
#
# but mail clients hard-wrap those lines at unpredictable columns, so this
# pulls every https URL out of the file as a whole rather than reading it line
# by line, and takes each name from the URL path rather than from --output,
# which the wrap may have separated from its URL. Knee entries are dropped
# here: the ask is brain only.
[ -r "$URLS_FILE" ] || {
  echo "no URL file at $URLS_FILE" >&2
  echo "paste the curl block from the NYU approval email into it (chmod 600)," >&2
  echo "or point URLS_FILE at wherever you saved it." >&2
  exit 1
}

ALL_PAIRS="$WORK_DIR/all_pairs.tsv"
PAIRS="$WORK_DIR/brain_pairs.tsv"

# The `|| true` matters: under `set -e` with `pipefail`, a URL file that
# contains no links at all (wrong file saved, empty paste) would otherwise
# kill the script right here on grep's exit 1, with no message at all, instead
# of reaching the diagnostic below. http as well as https only so that the
# local smoke test can point this at a throwaway server; NYU serves https.
{ grep -oE 'https?://[^"'"'"'[:space:]]+' "$URLS_FILE" || true; } | while read -r url; do
  path="${url%%\?*}"
  printf '%s\t%s\n' "${path##*/}" "$url"
done > "$ALL_PAIRS.raw"

# Last occurrence of a name wins, so a re-issued URL pasted below an expired
# one supersedes it without having to clean up the file by hand.
awk -F'\t' '{ seen[$1] = $2 } END { for (n in seen) print n "\t" seen[n] }' \
  "$ALL_PAIRS.raw" | sort > "$ALL_PAIRS"

# EXCLUDE drops matching names without editing the URL file. The default is
# empty, i.e. everything named brain_*, which includes brain_fastMRI_DICOM --
# that one is DICOM images, not multi-coil k-space, so set
# EXCLUDE='DICOM' if the ask is read strictly as multi-coil only.
EXCLUDE="${EXCLUDE:-}"
grep -E '^brain' "$ALL_PAIRS" > "$PAIRS.brain" || true
if [ -n "$EXCLUDE" ]; then
  grep -Ev "$EXCLUDE" "$PAIRS.brain" > "$PAIRS" || true
else
  cp "$PAIRS.brain" "$PAIRS"
fi
SHA_URL="$(awk -F'\t' '$1 == "SHA256" { print $2 }' "$ALL_PAIRS" | tail -1)"

NFILES="$(wc -l < "$PAIRS" | tr -d ' ')"
if [ "$NFILES" -eq 0 ]; then
  echo "found no brain_* URLs in $URLS_FILE" >&2
  echo "names seen there: $(cut -f1 "$ALL_PAIRS" | tr '\n' ' ')" >&2
  exit 1
fi

say "destination : $DEST"
say "url file    : $URLS_FILE"
say "brain files : $NFILES"
say "mode        : $MODE"
echo

# --------------------------------------------------------- expiry check -----
#
# A presigned URL carries its own deadline in Expires=<epoch>. A 1.37 TB pull
# runs for hours, so a URL that dies mid-transfer is a real failure mode and
# worth catching before the first byte rather than at hour six.
NOW="$(date +%s)"
SOONEST=""
while IFS=$'\t' read -r name url; do
  exp="$(printf '%s' "$url" | grep -oE 'Expires=[0-9]+' | head -1 | cut -d= -f2)"
  [ -n "$exp" ] || continue
  if [ -z "$SOONEST" ] || [ "$exp" -lt "$SOONEST" ]; then SOONEST="$exp"; fi
done < "$PAIRS"

if [ -n "$SOONEST" ]; then
  left=$(( SOONEST - NOW ))
  when="$(date -d "@$SOONEST" 2>/dev/null || echo "epoch $SOONEST")"
  if [ "$left" -le 0 ]; then
    echo "these URLs expired on $when" >&2
    echo "request a fresh batch from fastmri@med.nyu.edu before starting" >&2
    exit 1
  fi
  say "urls valid another $(( left / 86400 ))d $(( left % 86400 / 3600 ))h (until $when)"
  [ "$left" -lt 86400 ] && say "WARNING: under 24h left, and this transfer takes hours"
fi

# ---------------------------------------------------------- quota report ----
say "group staging quota:"
if command -v get_quotas >/dev/null 2>&1; then
  get_quotas "$GROUP_STAGING" 2>&1 | sed 's/^/   /' || true
else
  say "   get_quotas not on PATH -- not on a CHTC host?"
fi

if [ -d "$GROUP_STAGING" ]; then
  used_items="$(find "$GROUP_STAGING" 2>/dev/null | wc -l | tr -d ' ')"
  used_bytes="$(du -s --block-size=1 "$GROUP_STAGING" 2>/dev/null | cut -f1)"
  say "   in use now: ${used_items:-?} items, $(human "${used_bytes:-0}")"
else
  echo "cannot see $GROUP_STAGING -- is the group ACL in place for ${USER:-this account}?" >&2
  [ "$MODE" = check ] || exit 1
fi
echo

# ------------------------------------------------------------ size survey ---
SIZES="$WORK_DIR/sizes.tsv"
say "resolving remote sizes ($NFILES ranged GETs)..."
: > "$SIZES"
total=0
while IFS=$'\t' read -r name url; do
  sz="$(remote_size "$url")"
  case "${sz:-x}" in
    *[!0-9]* | "" )
      echo "   $name: could not read a remote size (expired or bad URL?)" >&2
      sz=0 ;;
    * )
      printf '   %-46s %s\n' "$name" "$(human "$sz")" ;;
  esac
  printf '%s\t%s\n' "$name" "$sz" >> "$SIZES"
  total=$(( total + sz ))
done < "$PAIRS"
echo
say "total to transfer: $(human "$total") across $NFILES files"
say "item cost under $DATASET_DIR: $(( NFILES + 2 )) (tarballs, manifest, directory)"
echo

if [ "$MODE" = check ]; then
  say "check complete -- nothing was downloaded."
  say "if those numbers look right, run '$0 fetch' inside tmux."
  exit 0
fi

# ------------------------------------------------------------- the fetch ----
if [ "$MODE" = fetch ]; then
  mkdir -p "$DEST"

  # The manifest is small and is not a brain tarball, so it is fetched on its
  # own. It costs one item and is worth it: without it there is nothing to
  # verify 1.37 TB of tarballs against.
  if [ -n "$SHA_URL" ]; then
    say "fetching the SHA256 manifest"
    curl -sS --fail "$SHA_URL" --output "$DEST/SHA256" \
      || say "   manifest fetch failed (continuing; verification will be skipped)"
  else
    say "no SHA256 URL in $URLS_FILE -- verification will fall back to recording hashes"
  fi

  fetch_one() {  # fetch_one <name> <url> <expected_size>
    local name="$1" url="$2" want="$3"
    local dest="$DEST/$name" log="$LOG_DIR/$name.log" have

    if [ -s "$dest" ]; then
      have="$(local_size "$dest")"
      if [ "$have" = "$want" ]; then
        say "   $name already complete ($(human "$have")), skipping"
        touch "$DONE_DIR/$name"
        return 0
      fi
      say "   $name resuming at $(human "$have") of $(human "$want")"
    else
      say "   $name starting ($(human "$want"))"
    fi

    # --fail so that an expired-URL XML error body is an error rather than a
    # 400-byte "tarball"; --retry rides out the transient 5xx that S3 throws
    # on multi-hour pulls.
    if curl -C - --fail --retry 5 --retry-delay 10 --retry-connrefused \
            -sS --show-error "$url" --output "$dest" 2>>"$log"; then
      have="$(local_size "$dest")"
      if [ "$have" = "$want" ]; then
        touch "$DONE_DIR/$name"
        say "   $name complete ($(human "$have"))"
        return 0
      fi
      say "   $name stopped at $(human "$have"), expected $(human "$want") -- re-run to resume"
      return 1
    fi

    say "   $name FAILED (see $log)"
    return 1
  }

  say "starting transfer, $JOBS files at a time; logs in $LOG_DIR"
  echo

  running=0
  while IFS=$'\t' read -r name url; do
    want="$(awk -F'\t' -v n="$name" '$1 == n { print $2 }' "$SIZES")"
    if [ "${want:-0}" -le 0 ]; then
      say "   $name has no usable size, skipping"
      continue
    fi
    if [ -f "$DONE_DIR/$name" ] && [ "$(local_size "$DEST/$name" 2>/dev/null || echo 0)" = "$want" ]; then
      say "   $name already done, skipping"
      continue
    fi

    fetch_one "$name" "$url" "$want" &
    running=$(( running + 1 ))
    if [ "$running" -ge "$JOBS" ]; then
      wait -n || true
      running=$(( running - 1 ))
    fi
  done < "$PAIRS"
  wait || true

  echo
  # Workers run as background subshells and cannot increment a counter in this
  # shell, so the tally is the marker count, which every worker writes to the
  # same $HOME directory.
  ndone="$(ls -1 "$DONE_DIR" 2>/dev/null | wc -l | tr -d ' ')"
  say "$ndone/$NFILES files complete"
  [ "$ndone" -eq "$NFILES" ] || say "re-run '$0 fetch' to resume the rest"
fi

# ------------------------------------------------------------- verify -------
#
# This reads all 1.37 TB back off staging, so it takes a while. It is the only
# thing standing between a silently truncated tarball and a job that dies
# three weeks from now, so it is not optional.
BAD_LIST="$WORK_DIR/failed_verification.txt"
: > "$BAD_LIST"

if [ -s "$DEST/SHA256" ]; then
  say "verifying against NYU's SHA256 manifest (reads every byte back -- slow)"
  ( cd "$DEST" && while IFS=$'\t' read -r name _; do
      [ -s "$name" ] || { say "   $name missing, skipping"; echo "$name" >> "$BAD_LIST"; continue; }
      if grep -qF "$name" SHA256; then
        if grep -F "$name" SHA256 | sha256sum -c - ; then :; else
          say "   $name FAILED verification"
          echo "$name" >> "$BAD_LIST"
        fi
      else
        say "   $name not listed in the manifest, hashing it for the record"
        sha256sum "$name" >> SHA256.local
      fi
    done < "$PAIRS" )

  # A bad hash means the bytes on staging are wrong, and curl -C - would
  # happily append to them forever without ever fixing what is already there.
  # The only cure is to delete the file and let fetch pull it again, so say so
  # rather than leaving a wall of FAILED lines and no instruction.
  if [ -s "$BAD_LIST" ]; then
    echo
    say "$(wc -l < "$BAD_LIST" | tr -d ' ') file(s) did not verify:"
    sed 's/^/   /' "$BAD_LIST"
    echo
    say "resume cannot repair a bad file -- delete each one and re-fetch it:"
    while read -r bad; do echo "   rm '$DEST/$bad' '$DONE_DIR/$bad'"; done < "$BAD_LIST"
    say "then re-run: $0 fetch"
  else
    say "all $NFILES files verified against the manifest"
  fi
else
  say "no SHA256 manifest present; skipping verification"
fi

echo
say "contents of $DEST:"
ls -la "$DEST"
echo
say "items under $DATASET_DIR: $(find "$DEST" | wc -l | tr -d ' ')"
say "nothing here is extracted, and nothing should be -- unpack into job scratch."
