#!/bin/bash 

# Input arguments
NUM_IMAGES=$1
SPLIT=$2
CLASS=$3

# Base directory
BASE_DIR=~/Data

# Validate input
if [[ -z "$NUM_IMAGES" || -z "$SPLIT" || -z "$CLASS" ]]; then
    echo "Usage: $0 <num_images> <split_folder> <class_name>"
    echo "Example: $0 50 train Skin"
    exit 1
fi

# Create the target directory if it doesn't exist
TARGET_DIR="${BASE_DIR}/${SPLIT}/${CLASS}"
mkdir -p "$TARGET_DIR"

# Camera config
TIMEOUT=1000   # milliseconds per capture
SLEEP_TIME=1   # seconds between captures

echo "Saving images to: $TARGET_DIR"
echo "Capturing $NUM_IMAGES images..."

# Loop to capture images
for ((i=1; i<=NUM_IMAGES; i++)); do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    FILENAME="${TARGET_DIR}/${CLASS}_${SPLIT}_${i}_${TIMESTAMP}.jpg"

    rpicam-jpeg -t $TIMEOUT -o "$FILENAME"
    echo "  -> Saved $FILENAME"
    sleep $SLEEP_TIME
done

echo "Done! Captured $NUM_IMAGES images in $TARGET_DIR"

# for ((i=1; i<=N; i++)); do
#	OUTPUT="${DIR}/${Class}_${i}.jpg" 
#	libcamera-jpeg -t $TIMEOUT -o "$OUTPUT"
#	echo "Captured ${DIR}/image${i}.jpg"
#done  

