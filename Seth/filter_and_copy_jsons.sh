#!/bin/bash

set -e

SOURCE_DIR="$1"
DEST_DIR="$2"

if [[ -z "$SOURCE_DIR" || -z "$DEST_DIR" ]]; then
    echo "Usage: $0 /path/to/source_dir /path/to/destination_dir"
    exit 1
fi

mkdir -p "$DEST_DIR"

LOG_FILE="$DEST_DIR/skipped_files.log"
> "$LOG_FILE"  # Clear log file before starting

REQUIRED_KEYS=("Determined Charges" "Avg Quadrupole Matrix" "Avg Diagonalized Quadrupole matrix" "Avg Spectral Anisotropy Matrix")

find "$SOURCE_DIR" -type f -name "*.json" ! -path "*/.*" | while read -r json_file; do
    valid=true

    # Ensure the top-level structure is an object
    if ! jq -e 'type == "object"' "$json_file" >/dev/null 2>&1; then
        echo "Skipping $json_file: Top-level JSON is not an object"
        echo "$json_file: Top-level JSON is not an object" >> "$LOG_FILE"
        continue
    fi

    # Check required top-level keys
    for key in "${REQUIRED_KEYS[@]}"; do
        non_empty=$(jq --arg k "$key" 'has($k) and (.[$k] | length > 0)' "$json_file")
        if [[ "$non_empty" != "true" ]]; then
            echo "Skipping $json_file: Missing or empty required key '$key'"
            echo "$json_file: Missing or empty required key '$key'" >> "$LOG_FILE"
            valid=false
            break
        fi
    done

    # # Check that all top-level keys are non-empty
    # if [[ "$valid" == true ]]; then
    #     empty_any=$(jq '[.[] | select((type == "object" or type == "array") and (length == 0))] | length' "$json_file")
    #     if [[ "$empty_any" -ne 0 ]]; then
    #         echo "Skipping $json_file: One or more top-level keys are empty"
    #         echo "$json_file: One or more top-level keys are empty" >> "$LOG_FILE"
    #         valid=false
    #     fi
    # fi

    # Copy if valid
    if [[ "$valid" == true ]]; then
        cp "$json_file" "$DEST_DIR/"
        echo "Copied $json_file"
    fi
done
