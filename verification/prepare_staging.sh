#!/bin/bash
# prepare_staging.sh: one-time data preparation on the CHTC access point.
#
# Puts the two Tier 1 inputs into /staging/<netid>/fastmri/ :
#   knee_multicoil_val.tar.xz          (93.8 GB, from the NYU presigned URL)
#   knee_leaderboard_state_dict.pt     (released checkpoint, ~120 MB)
# and verifies the tarball against NYU's SHA256 manifest.
#
# Usage (on ap2001.chtc.wisc.edu):
#   export FASTMRI_VAL_URL='https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_val.tar.xz?AWSAccessKeyId=...'
#   export FASTMRI_SHA_URL='https://fastmri-dataset.s3.amazonaws.com/v3.0/SHA256?AWSAccessKeyId=...'
#   ./prepare_staging.sh
#
# The presigned URLs come from the NYU email and expire ~90 days after issue
# (the current batch around 2026-12-08). Paste them from the email; do not
# commit them. curl -C - resumes a partial download, so re-running after an
# interruption is safe, and files that are already complete are skipped.
#
# CHTC note: large transfers into /staging are normally done through
# transfer.chtc.wisc.edu, but an outbound curl from the access point straight
# into /staging is the documented path for datasets hosted on the web. Run it
# under nohup or tmux; 94 GB takes 1 to 3 hours depending on the S3 link.
set -euo pipefail

NETID="${NETID:-$USER}"
DEST="${STAGING_DIR:-/staging/$NETID/fastmri}"
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
  curl -sS -L -r 0-0 -o /dev/null -D - "$1" 2>/dev/null \
    | tr -d '\r' | awk -F/ 'tolower($0) ~ /^content-range:/ {print $2}' | tail -1
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

echo "== SHA256 manifest"
curl -sS "$FASTMRI_SHA_URL" --output SHA256
grep -c . SHA256 >/dev/null || { echo "SHA256 manifest is empty" >&2; exit 1; }

echo "== released checkpoint"
fetch "$CKPT_URL" knee_leaderboard_state_dict.pt
ls -la knee_leaderboard_state_dict.pt

echo "== knee_multicoil_val.tar.xz (this is the long step)"
fetch "$FASTMRI_VAL_URL" knee_multicoil_val.tar.xz

echo "== verifying"
if grep -q "knee_multicoil_val.tar.xz" SHA256; then
  grep "knee_multicoil_val.tar.xz" SHA256 | sha256sum -c -
else
  echo "knee_multicoil_val.tar.xz not listed in the manifest; computing the hash for the record"
  sha256sum knee_multicoil_val.tar.xz | tee knee_multicoil_val.tar.xz.sha256
fi

echo
echo "done. In verify.sub set:"
echo "  netid = $NETID"
echo "and the default data/ckpt paths will resolve to osdf:///chtc/staging/$NETID/fastmri/..."
ls -la "$DEST"
