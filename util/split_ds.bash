#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_folder> <destination_folder>"
    exit 1
fi

# Assign arguments to variables
SOURCE_FOLDER=$1
DESTINATION_FOLDER=$2

# Check if the source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Source folder does not exist: $SOURCE_FOLDER"
    exit 1
fi

# Check if the destination folder exists, if not, create it
if [ ! -d "$DESTINATION_FOLDER" ]; then
    mkdir -p "$DESTINATION_FOLDER"
fi

# Select 58 random PNG files that do not contain 'mask' in their name
FILES=($(find "$SOURCE_FOLDER" -type f -name '*.png' ! -name '*mask*' | shuf -n 51))

# Copy the selected files to the destination folder
for FILE in "${FILES[@]}"; do
    mv "$FILE" "$DESTINATION_FOLDER"
done

# Copy the corresponding 'mask' files to the destination folder
for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    MASK_FILE="${SOURCE_FOLDER}/mask_${BASENAME}"
    if [ -f "$MASK_FILE" ]; then
        mv "$MASK_FILE" "$DESTINATION_FOLDER"
    else
        echo "Warning: Corresponding mask file not found for $FILE"
    fi
done