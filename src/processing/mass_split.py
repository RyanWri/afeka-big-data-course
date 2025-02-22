import os
import numpy as np
import yaml
from PIL import Image
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
from pyspark.sql import SQLContext, Row
from pyspark import SparkContext

def extract_patches(image_path, patch_size):
    """
    Extract patches from a single image.
    Returns a list of (image_id, row_index, col_index, patch_value)
    """
    try:
        file_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        width, height = image.size
        patches = []

        for row in range(0, height, patch_size):
            for col in range(0, width, patch_size):
                box = (col, row, col + patch_size, row + patch_size)
                patch = image.crop(box)
                patch_array = np.array(patch).astype(np.float32) / 255.0  # Normalize
                patches.append((file_name, row // patch_size, col // patch_size, patch_array.flatten().tolist()))

        return patches
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []  # Return an empty list if the image fails

def main(sqlContext):
    # Load Configuration
    with open("src/processing/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Paths
    input_folder = config["dataset"]["low_resolution_dir"]
    patches_path = config["dataset"]["patches_dir"]

    if not os.path.exists(patches_path):
        os.makedirs(patches_path)

    patch_size = config["processing"]["patch_size"]  # Patch size in pixels

    # Find all images
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_paths = [
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith(image_extensions)
    ]

    if not image_paths:
        print("No images found in the input folder:", input_folder)
        raise ValueError(f"No images found in the input folder: {input_folder}")

    # **STEP 1: Load all images & extract patches BEFORE parallelization**
    all_patches = []
    for image_path in image_paths:
        all_patches.extend(extract_patches(image_path, patch_size))

    if not all_patches:
        print("No patches extracted. Exiting...")
        raise ValueError("No patches extracted. Exiting...")

    # **STEP 2: Convert to DataFrame**
    schema = StructType([
        StructField("image_id", StringType(), True),
        StructField("row_index", IntegerType(), True),
        StructField("col_index", IntegerType(), True),
        StructField("patch_value", ArrayType(FloatType()), True)  # Explicitly define as an array of floats
    ])
    patches_df = sqlContext.createDataFrame(all_patches, schema)

    # **STEP 3: Save as Parquet**
    patches_df.write.mode("overwrite").parquet(patches_path)
    print(f"Patches saved to {patches_path}")
