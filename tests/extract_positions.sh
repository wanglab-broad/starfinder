#!/bin/bash
# Script to extract a range of Position folders from a zip file
# while preserving the original folder structure

set -e

# Function to display usage
usage() {
    echo "Usage: $0 <zip_file> <output_dir> <position_start> <position_end>"
    echo ""
    echo "Arguments:"
    echo "  zip_file        Path to the input zip file"
    echo "  output_dir      Destination directory for extracted files"
    echo "  position_start  Starting position number (e.g., 1 for Position001)"
    echo "  position_end    Ending position number (e.g., 64 for Position064)"
    echo ""
    echo "Example:"
    echo "  $0 ~/data/dataset.zip ~/output 1 64"
    echo ""
    echo "This will extract Position001 through Position064 from dataset.zip"
    exit 1
}

# Check argument count
if [[ $# -ne 4 ]]; then
    echo "Error: Expected 4 arguments, got $#"
    echo ""
    usage
fi

# Parse arguments
ZIP_FILE="$1"
OUTPUT_DIR="$2"
POS_START="$3"
POS_END="$4"

# Validate inputs
if [[ ! -f "$ZIP_FILE" ]]; then
    echo "Error: Zip file not found: $ZIP_FILE"
    exit 1
fi

if ! [[ "$POS_START" =~ ^[0-9]+$ ]] || ! [[ "$POS_END" =~ ^[0-9]+$ ]]; then
    echo "Error: Position start and end must be positive integers"
    exit 1
fi

if [[ "$POS_START" -gt "$POS_END" ]]; then
    echo "Error: Position start ($POS_START) must be <= position end ($POS_END)"
    exit 1
fi

if [[ "$POS_START" -lt 1 ]] || [[ "$POS_END" -gt 999 ]]; then
    echo "Error: Position numbers must be between 1 and 999"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Format position numbers for display (3 digits with leading zeros)
POS_START_FMT=$(printf "%03d" "$POS_START")
POS_END_FMT=$(printf "%03d" "$POS_END")

echo "Extracting Position${POS_START_FMT}-Position${POS_END_FMT} from: $ZIP_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Generate patterns for the specified position range
# Handles 3-digit position numbers (Position001, Position002, ..., Position999)
PATTERNS=()
for i in $(seq "$POS_START" "$POS_END"); do
    # Format as 3-digit number with leading zeros
    POS_NUM=$(printf "%03d" "$i")
    PATTERNS+=("*/Position${POS_NUM}/*" "*/Position${POS_NUM}")
done

echo "Extracting positions ${POS_START_FMT}-${POS_END_FMT} ($(($POS_END - $POS_START + 1)) positions)..."

# Extract files matching the patterns
# -o: overwrite without prompting
# -d: destination directory
unzip -o "$ZIP_FILE" "${PATTERNS[@]}" -d "$OUTPUT_DIR"

echo ""
echo "Extraction complete!"
echo ""

# Show what was extracted
echo "Extracted directories:"
find "$OUTPUT_DIR" -type d -name "Position*" | sort | head -100

# Total count
POSITION_COUNT=$(find "$OUTPUT_DIR" -type d -name "Position*" | wc -l)
echo ""
echo "Total Position directories extracted: $POSITION_COUNT"
