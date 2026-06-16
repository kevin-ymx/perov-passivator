#!/bin/bash
# ============================================================================
# Script to run filter_molecules.py iteratively for multiple input CSV files.
#
# Input CSV files are named: 000000001_000500000.csv through 176500001_177000000.csv
# Each CSV has columns: PUBCHEM_COMPOUND_CID, SMILES
#
# Usage:
#   1. Get interactive allocation first:
#      salloc -N 1 -C cpu -q interactive -t 04:00:00 -A m3342
#   2. Run this script:
#      bash run_filter_mol.sh
# ============================================================================

# Set paths
BASE_INPUT_DIR="/kfs3/scratch/yeming/ai4m/prediction/filtered_csv"
OUTPUT_DIR="/kfs3/scratch/yeming/ai4m/prediction/filtered_csv_02252026"
FILTER_SCRIPT="/kfs3/scratch/yeming/ai4m/prediction/dataset/ssl/filter_molecules.py"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Parameters
WORKERS=104
DEBUG_FIRST_SHARD=true

# Compound range parameters
START_COMPOUND=1
END_COMPOUND=177000000
STEP=500000

# Calculate total shards
TOTAL_SHARDS=$(( (END_COMPOUND - START_COMPOUND + STEP) / STEP ))
CURRENT_SHARD=0

echo "Starting batch processing..."
echo "Input directory: ${BASE_INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Total shards to process: ${TOTAL_SHARDS}"

# Iterate through all compound ranges
for (( RANGE_START=${START_COMPOUND}; RANGE_START<${END_COMPOUND}; RANGE_START+=${STEP} )); do
    RANGE_END=$((RANGE_START + STEP - 1))

    # Format with leading zeros (9 digits)
    RANGE_START_FMT=$(printf "%09d" ${RANGE_START})
    RANGE_END_FMT=$(printf "%09d" ${RANGE_END})

    # Compound range string
    COMPOUND_RANGE="${RANGE_START_FMT}_${RANGE_END_FMT}"

    # Input CSV file
    INPUT_CSV="${BASE_INPUT_DIR}/${COMPOUND_RANGE}.csv"

    # Output file
    OUTPUT_FILE="${OUTPUT_DIR}/${COMPOUND_RANGE}.csv"

    # Update progress counter
    CURRENT_SHARD=$((CURRENT_SHARD + 1))

    echo ""
    echo "============================================="
    echo "Shard ${CURRENT_SHARD}/${TOTAL_SHARDS}: ${COMPOUND_RANGE}"
    echo "============================================="

    # Check if input CSV exists
    if [ ! -f "${INPUT_CSV}" ]; then
        echo "Skipping: input CSV not found (${INPUT_CSV})"
        continue
    fi

    # Check if output file already exists
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "Skipping: output file already exists"
        continue
    fi

    # Add --debug flag for first shard if DEBUG_FIRST_SHARD is true
    DEBUG_FLAG=""
    if [ "${DEBUG_FIRST_SHARD}" = "true" ] && [ ${CURRENT_SHARD} -eq 1 ]; then
        DEBUG_FLAG="--debug"
        echo "DEBUG MODE: Will print first molecule's properties"
    fi

    echo "Running filter_molecules.py on ${COMPOUND_RANGE}.csv ..."
    srun -n 1 -c ${WORKERS} python ${FILTER_SCRIPT} \
        --input_csv ${INPUT_CSV} \
        --output ${OUTPUT_FILE} \
        --workers ${WORKERS} \
        ${DEBUG_FLAG}

    if [ $? -eq 0 ]; then
        echo "Completed successfully"
    else
        echo "FAILED"
    fi
done

echo ""
echo "============================================="
echo "Batch processing complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================="
