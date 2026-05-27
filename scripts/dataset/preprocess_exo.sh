#!/bin/bash

# Preprocess exocentric camera videos by extracting and undistorting frames into HDF5.
# Usage: DATA_DIR=/path/to/egoexo4d OUTPUT_DIR=/path/to/output bash scripts/dataset/preprocess_exo.sh scripts/dataset/sequences_train.txt [num_workers]

DATA_DIR=${DATA_DIR:?Error: DATA_DIR is required}
OUTPUT_DIR=${OUTPUT_DIR:?Error: OUTPUT_DIR is required}

SEQUENCES_FILE=${1:?Error: sequences file is required (e.g. bash scripts/dataset/preprocess_exo.sh scripts/dataset/sequences_train.txt)}
NUM_WORKERS=${2:-1}

if [ ! -f "$SEQUENCES_FILE" ]; then
    echo "Error: file not found: $SEQUENCES_FILE"
    exit 1
fi

SEQUENCE_LIST=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(echo -e "${line}" | xargs)"
    [ -z "$line" ] && continue
    SEQUENCE_LIST+=("$line")
done < "$SEQUENCES_FILE"

echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Sequences file: $SEQUENCES_FILE (${#SEQUENCE_LIST[@]} sequences)"
echo "Workers: $NUM_WORKERS"

for SEQ in "${SEQUENCE_LIST[@]}"; do
    echo ""
    echo "--- Processing sequence: $SEQ ---"

    python scripts/dataset/preprocess_exo.py \
        --data_dir "$DATA_DIR" \
        --sequences "$SEQ" \
        --output_dir "$OUTPUT_DIR" \
        --num_workers "$NUM_WORKERS"
done
