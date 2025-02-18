import json
import shutil
import os


def process_json_file(filename):
    # Load JSON from the file
    with open(filename, "r") as f:
        data = json.load(f)

    # Process and print dataset info
    print("Dataset Info:")
    info = data.get("info", {})
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Process each scene in the JSON
    scenes = data.get("scenes", [])
    for scene in scenes:
        print("\n--- Scene ---")
        print(f"Image Index: {scene.get('image_index')}")
        print(f"Image Filename: {scene.get('image_filename')}")
        print(f"Split: {scene.get('split')}")

        # Process objects in the scene
        print("\nObjects:")
        for idx, obj in enumerate(scene.get("objects", [])):
            print(f"  Object {idx}:")
            for key, value in obj.items():
                print(f"    {key}: {value}")

        # Process relationships between objects
        print("\nRelationships:")
        relationships = scene.get("relationships", {})
        for relation, lists in relationships.items():
            print(f"  {relation}:")
            for obj_index, related in enumerate(lists):
                print(f"    Object {obj_index} -> {related}")

        # Process directional vectors
        print("\nDirections:")
        directions = scene.get("directions", {})
        for direction, vector in directions.items():
            print(f"  {direction}: {vector}")


def list_green_cube_objects(filename, color='green', shape='cube'):
    # Load the JSON from the file
    with open(filename, "r") as f:
        data = json.load(f)

    # Process each scene
    for scene in data.get("scenes", []):
        shape_indices = []
        for i, obj in enumerate(scene.get("objects", [])):
            if obj.get("color") == color and obj.get("shape") == shape:
                shape_indices.append(i)

        # If the scene contains one or more green cubes, print the scene's image index and the object numbers
        if shape_indices:
            print(
                f"Scene with image_index {scene.get('image_index')} has {color} {shape} at object numbers: {shape_indices}")


def copy_images_based_on_object_instances(json_filename, target_shape, target_color, source_folder,
                                          dest_folder_multiple, dest_folder_single):
    """
    Loads the JSON from the file and copies image files into two different folders:
      - dest_folder_multiple: for images where the specified target shape and target color appear more than once.
      - dest_folder_single: for images where the specified target shape and target color appear exactly once.

    :param json_filename: Path to the JSON file.
    :param target_shape: The shape to look for.
    :param target_color: The color to look for.
    :param source_folder: Directory where the image files are located.
    :param dest_folder_multiple: Destination folder for images with multiple instances.
    :param dest_folder_single: Destination folder for images with a single instance.
    """
    # Create destination directories if they do not exist
    os.makedirs(dest_folder_multiple, exist_ok=True)
    os.makedirs(dest_folder_single, exist_ok=True)

    # Open and load JSON data
    with open(json_filename, "r") as f:
        data = json.load(f)

    # Process each scene in the JSON data
    for scene in data.get("scenes", []):
        count = sum(
            1 for obj in scene.get("objects", [])
            if obj.get("shape") == target_shape and obj.get("color") == target_color
        )
        image_filename = scene.get("image_filename")
        if image_filename:
            src_path = os.path.join(source_folder, image_filename)
            # Check if the source image file exists before copying
            if not os.path.exists(src_path):
                print(f"Source file not found: {src_path}")
                continue

            if count > 1:
                dest_path = os.path.join(dest_folder_multiple, image_filename)
                print(f"Copying {image_filename} to '{dest_folder_multiple}' (Count: {count})")
                shutil.copy(src_path, dest_path)
            elif count == 1:
                dest_path = os.path.join(dest_folder_single, image_filename)
                print(f"Copying {image_filename} to '{dest_folder_single}' (Count: {count})")
                shutil.copy(src_path, dest_path)


def print_all_shapes_and_colors(filename):
    """
    Loads the JSON from the file and prints all unique shapes and colors present in the dataset.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    shapes = set()
    colors = set()

    for scene in data.get("scenes", []):
        for obj in scene.get("objects", []):
            shape = obj.get("shape")
            color = obj.get("color")
            if shape:
                shapes.add(shape)
            if color:
                colors.add(color)

    print("Shapes present in the dataset:")
    for shape in sorted(shapes):
        print(f" - {shape}")

    print("\nColors present in the dataset:")
    for color in sorted(colors):
        print(f" - {color}")


if __name__ == "__main__":
    # Specify the path to your JSON file here
    filename = "/data_ssd/pwojcik/CLEVR_v1.0/scenes/CLEVR_test_scenes.json"
    print_all_shapes_and_colors(filename)
    copy_images_based_on_object_instances(json_filename=filename, target_shape='cube', target_color= 'green',
                                          source_folder='/data_ssd/pwojcik/CLEVR_v1.0/images/test',
                                          dest_folder_multiple='/data_ssd/pwojcik/CLEVR_v1.0/multi_green_cube/test',
                                          dest_folder_single='/data_ssd/pwojcik/CLEVR_v1.0/single_green_cube/test')
    #list_green_cube_objects(filename)
    #process_json_file(filename)
