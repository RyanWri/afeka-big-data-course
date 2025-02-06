"""
Pipeline Steps:
Step 1: Split the image into grayscale patches.
Step 2: Create a Spark DataFrame with patch data.
Step 3: Apply the FSRCNN model to super-resolve each patch.
Step 4: Collect the super-resolved patches and reconstruct the final grayscale image.
Step 5: Save the resulting image to disk.
"""

import torch
import numpy as np
from pyspark.sql import SparkSession
from PIL import Image
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector, VectorUDT
from src.fsrcnn import FSRCNN


def image_to_patches(image_path, patch_size):
    """
    Splits a grayscale image into patches represented as flattened tensors (vectors).
    Returns patches as a list of (row_index, col_index, vector).
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    width, height = image.size
    patches = []

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            box = (col, row, col + patch_size, row + patch_size)
            patch = image.crop(box)
            patch_array = np.array(patch).flatten()  # Convert to 1D vector
            patches.append(
                (row // patch_size, col // patch_size, DenseVector(patch_array))
            )

    return patches


def patches_to_image(patches, original_image_size, patch_size, upscale_factor):
    """
    Reconstructs the full image from super-resolved grayscale patches and their indices.
    """
    original_width, original_height = original_image_size
    sr_image = Image.new(
        "L", (original_width * upscale_factor, original_height * upscale_factor)
    )  # Grayscale, resized to match the upscale factor

    for row_index, col_index, patch_vector in patches:
        patch_array = (
            np.array(patch_vector)
            .reshape((patch_size * upscale_factor, patch_size * upscale_factor))
            .astype(np.uint8)
        )
        patch_image = Image.fromarray(patch_array, mode="L")
        x, y = (
            col_index * patch_size * upscale_factor,
            row_index * patch_size * upscale_factor,
        )
        sr_image.paste(patch_image, (x, y))

    return sr_image


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
    # const
    dataset_path = "/home/ran/datasets/spark-picsum-images"
    image_id = "grayscale_003.jpg"
    local_image_path = f"{dataset_path}/LR/{image_id}"
    model_path = (
        "/home/ran/Documents/afeka/big-data/models/fsrcnn_x2-T91-f791f07f.pth.tar"
    )
    patch_size = 16  # Patch size for splitting
    upscale_factor = 2  # Upscale factor used by the FSRCNN model

    # Spark Session
    spark = SparkSession.builder.appName("ImageSuperResolution").getOrCreate()
    sc = spark.sparkContext

    # Load and broadcast the model
    model = FSRCNN(upscale_factor=upscale_factor)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
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

    # Step 5: Save the super-resolved image
    output_image = f"{dataset_path}/SR/{image_id}"
    sr_image.save(output_image)
    print(f"Super-resolved image saved as {output_image}.")
