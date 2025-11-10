#!/usr/bin/env bash
set -euo pipefail

IN_DIR="traces"
OUT_DIR="reports"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Enable nullglob so that the loop simply skips if no files match
shopt -s nullglob

# Loop over all .nsys-rep files in traces/
for rep_file in "$IN_DIR"/*.nsys-rep; do
    # Skip if no file exists
    [ -e "$rep_file" ] || continue

    # Extract base name (remove directory and .nsys-rep extension)
    base_name="$(basename "$rep_file" .nsys-rep)"

    # Compose output path
    out_csv="$OUT_DIR/${base_name}"

    echo "==> Generating report for: $rep_file"
    echo "    Output: $out_csv"

    # Run Nsight Systems stats command
    nsys stats --report nvtx_sum --format csv --output "$out_csv" "$rep_file"
done

echo "All nvtx_sum reports written to '$OUT_DIR/'"