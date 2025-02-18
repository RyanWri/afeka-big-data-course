import os
from pyspark.sql import SparkSession
from PIL import Image
import numpy as np


def extract_patches_spark(image_path, patch_size):
    """
    Distributed function: Each worker loads an image and extracts patches.
    Returns: List of (file_name, row_index, col_index, vector)
    """
    file_name = os.path.basename(image_path)
    image = Image.open(image_path).convert("L")  # Grayscale
    width, height = image.size
    patches = []

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            box = (col, row, col + patch_size, row + patch_size)
            patch = image.crop(box)
            patch_array = np.array(patch).astype(np.float32) / 255.0  # Normalize
            patches.append(
                (
                    file_name,
                    row // patch_size,
                    col // patch_size,
                    patch_array.flatten().tolist(),
                )
            )

    return patches


if __name__ == "__main__":
    # Create Spark Session
    spark = SparkSession.builder.appName("ImagePatchExtraction").getOrCreate()
    sc = spark.sparkContext

    # Parameters – adjust these as needed.
    input_folder = (
        "/home/ran/datasets/spark-picsum-images/LR"  # Folder containing the image files
    )
    output_path = (
        "/home/ran/datasets/spark-picsum-images/LR-Patches"  # Where to save the patches
    )
    patch_size = 16  # Size of each patch (in pixels)

    # List image files in the input folder. You can add or remove extensions as needed.
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_paths = [
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith(image_extensions)
    ]

    if not image_paths:
        print("No images found in the input folder:", input_folder)
        spark.stop()
        exit(1)

    # Parallelize the list of image paths.
    image_rdd = sc.parallelize(image_paths)

    # For each image, extract patches using the provided function.
    # Each call to extract_patches_spark returns a list of tuples:
    # (image_id, row_index, col_index, patch_value)
    patches_rdd = image_rdd.flatMap(
        lambda image_path: extract_patches_spark(image_path, patch_size)
    )

    # Define a schema for the DataFrame.
    schema = ["image_id", "row_index", "col_index", "patch_value"]

    # Create a DataFrame from the RDD of patches.
    patches_df = spark.createDataFrame(patches_rdd, schema=schema)

    # Save the patches DataFrame to disk.
    # Here we use Parquet, but you could change this to CSV or another format if desired.
    patches_df.write.mode("overwrite").parquet(output_path)
    print(f"Patches saved to {output_path}")

    spark.stop()
