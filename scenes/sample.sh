#!/bin/bash

# Set source and destination folders
SOURCE_FOLDER="/data/pwojcik/shapes4/train/cubes_1_3/"
DEST_FOLDER="/data/pwojcik/shapes4/test/cubes_1_3/"

# Create destination folder if it doesn't exist
mkdir -p "$DEST_FOLDER"

# Get list of PNG files and randomly sample 10%
find "$SOURCE_FOLDER" -type f -name "*.png" | shuf -n $(($(ls "$SOURCE_FOLDER"/*.png | wc -l) / 10)) | while read file; do
    mv "$file" "$DEST_FOLDER"
done