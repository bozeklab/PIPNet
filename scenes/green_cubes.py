import json
import os
import shutil

# Load JSON file
json_path = "/data/pwojcik/whole_scene_data.json"
image_folder = "/data/pwojcik/images/"  # Update with correct path
destination_folder = "/data/pwojcik/shapes4/train/cubes_4_10/"

with open(json_path, "r") as file:
    data = json.load(file)

# Store filenames that meet the condition
matching_filenames = []


for filename, details in data.items():
    green_cube_count = sum(
        1 for obj in details["objects"] if obj["shape"] == "cube" and obj["color"] == "green"
    )

    if 4 <= green_cube_count <= 10:
        matching_filenames.append(filename)

# Copy matching files
for fname in matching_filenames:
    image_file = os.path.join(image_folder, f"{fname}.png")  # Assuming images are PNGs
    if os.path.exists(image_file):
        shutil.copy(image_file, destination_folder)
        print(f"Copied {image_file} to {destination_folder}")

print(f"Copied {len(matching_filenames)} images to {destination_folder}")