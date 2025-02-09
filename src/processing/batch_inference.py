"""
Pipeline Steps:
Step 1: Split the image into grayscale patches.
Step 2: Create a Spark DataFrame with patch data.
Step 3: Apply the FSRCNN model to super-resolve each patch.
Step 4: Collect the super-resolved patches and reconstruct the final grayscale image.
Step 5: Save the resulting image to disk.
"""

import torch
import yaml
import os
import numpy as np
from pyspark.sql import SparkSession
from PIL import Image
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector, VectorUDT
from src.processing.fsrcnn import load_fsrcnn_model
from src.processing.image_processing import image_to_patches, patches_to_image


def super_resolve_patch(patch_vector, model_broadcast, patch_size):
    """
    Applies FSRCNN super-resolution to a single grayscale patch vector.
    """
    model = model_broadcast.value  # Access the broadcasted model
    patch_array = (
        np.array(patch_vector).reshape((patch_size, patch_size)).astype(np.float32)
        / 255.0
    )
    patch_tensor = (
        torch.from_numpy(patch_array).unsqueeze(0).unsqueeze(0)
    )  # Add batch and channel dimensions

    # Perform inference
    with torch.no_grad():
        sr_patch_tensor = model(patch_tensor)

    sr_patch_array = sr_patch_tensor.squeeze().numpy() * 255.0  # Rescale to [0, 255]
    return DenseVector(sr_patch_array.flatten())


# Define the UDF
@udf(VectorUDT())
def super_resolve_udf(patch_vector):
    return super_resolve_patch(patch_vector, model_broadcast, patch_size)


if __name__ == "__main__":
    # Load Configuration
    with open("src/processing/config.yaml", "r") as file:
        prime_service = yaml.safe_load(file)

    # Extract values from config
    local_image_path = os.path.join(
        prime_service["dataset"]["low_resolution_dir"],
        prime_service["paths"]["original_image"],
    )
    patch_size = prime_service["processing"]["patch_size"]
    upscale_factor = prime_service["processing"]["upscale_factor"]
    model_path = prime_service["processing"]["model_path"]

    # Spark Session
    spark = SparkSession.builder.appName("ImageSuperResolution").getOrCreate()
    sc = spark.sparkContext

    # Load and broadcast the model
    model = load_fsrcnn_model(upscale_factor, model_path)
    model_broadcast = sc.broadcast(model)

    # Step 1: Split the image into patches
    patches = image_to_patches(local_image_path, patch_size)

    # Step 2: Create a DataFrame
    patch_df = spark.createDataFrame(
        patches, schema=["row_index", "col_index", "vector"]
    )

    # Step 3: Apply Super-Resolution
    sr_patch_df = patch_df.withColumn(
        "sr_vector", super_resolve_udf(patch_df["vector"])
    )

    # Step 4: Collect the super-resolved patches and reconstruct the image
    sr_patches = (
        sr_patch_df.select("row_index", "col_index", "sr_vector")
        .rdd.map(tuple)
        .collect()
    )
    original_image = Image.open(local_image_path)
    sr_image = patches_to_image(
        sr_patches, original_image.size, patch_size, upscale_factor
    )

    # Step 5: Save Final Image
    output_image_path = os.path.join(
        prime_service["dataset"]["super_resolved_dir"],
        prime_service["paths"]["output_image"],
    )
    sr_image.save(output_image_path)
    print(f"✅ Super-resolved image saved at {output_image_path}.")

    # Stop the Spark session
    spark.stop()
