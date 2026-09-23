#!/usr/bin/env bash
# Download MovieLens 25M into data/ and verify it. Safe to re-run.
#
# The dataset is never committed (MovieLens license forbids redistribution); only its checksums
# are. MD5 is the one GroupLens publishes at ml-25m.zip.md5; SHA-256 was recorded on first download.
set -euo pipefail

URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
EXPECTED_MD5="6b51fb2759a8657d3bfcbfc42b592ada"
EXPECTED_SHA256_FILE="scripts/ml-25m.zip.sha256"
EXPECTED_RATING_LINES=25000096   # 25,000,095 ratings + 1 header line

cd "$(dirname "$0")/.."
mkdir -p data

md5_of()    { md5 -q "$1" 2>/dev/null || md5sum "$1" | cut -d' ' -f1; }
sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

zip_ok() {
  [[ -f data/ml-25m.zip ]] || return 1
  [[ "$(md5_of data/ml-25m.zip)" == "$EXPECTED_MD5" ]] || return 1
  if [[ -f "$EXPECTED_SHA256_FILE" ]]; then
    [[ "$(sha256_of data/ml-25m.zip)" == "$(cut -d' ' -f1 "$EXPECTED_SHA256_FILE")" ]] || return 1
  fi
}

if zip_ok; then
  echo "data/ml-25m.zip present and checksums match; skipping download."
else
  [[ -f data/ml-25m.zip ]] && echo "data/ml-25m.zip failed checksum; re-downloading."
  rm -f data/ml-25m.zip data/ml-25m.zip.part
  # Download to .part so an interrupted run never leaves a truncated zip that looks complete.
  curl -fL --retry 3 -o data/ml-25m.zip.part "$URL"
  mv data/ml-25m.zip.part data/ml-25m.zip
  zip_ok || { echo "ERROR: checksum mismatch on fresh download." >&2; exit 1; }
fi

if [[ ! -f "$EXPECTED_SHA256_FILE" ]]; then
  echo "$(sha256_of data/ml-25m.zip)  ml-25m.zip" > "$EXPECTED_SHA256_FILE"
  echo "Recorded SHA-256 in $EXPECTED_SHA256_FILE (commit it)."
fi

rm -rf data/ml-25m
unzip -q -o data/ml-25m.zip -d data

lines=$(wc -l < data/ml-25m/ratings.csv | tr -d ' ')
if [[ "$lines" != "$EXPECTED_RATING_LINES" ]]; then
  echo "ERROR: ratings.csv has $lines lines, expected $EXPECTED_RATING_LINES." >&2
  exit 1
fi
for f in movies.csv genome-scores.csv genome-tags.csv tags.csv links.csv; do
  [[ -f "data/ml-25m/$f" ]] || { echo "ERROR: missing data/ml-25m/$f" >&2; exit 1; }
done
echo "OK: data/ml-25m/ratings.csv has $lines lines (25,000,095 ratings + header)."
