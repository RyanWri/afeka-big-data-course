import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, struct, udf
from pyspark.sql.types import StringType
import yaml
from PIL import Image


def patches_to_image(patches, original_size, patch_size, upscale_factor):
    """
    Reconstructs the final grayscale image from super-resolved patches.
    """
    original_width, original_height = original_size
    sr_image = Image.new(
        "L", (original_width * upscale_factor, original_height * upscale_factor)
    )

    for row_index, col_index, patch_vector in patches:
        patch_array = np.array(patch_vector).reshape(
            (patch_size * upscale_factor, patch_size * upscale_factor)
        )
        patch_image = Image.fromarray(patch_array.astype(np.uint8), mode="L")
        x, y = (
            col_index * patch_size * upscale_factor,
            row_index * patch_size * upscale_factor,
        )
        sr_image.paste(patch_image, (x, y))

    return sr_image


if __name__ == "__main__":
    # Load configuration parameters
    with open("src/processing/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    patch_size = config["processing"]["patch_size"]  # e.g., 16
    upscale_factor = config["processing"]["upscale_factor"]  # e.g., 2
    inference_result_dir = config["dataset"][
        "inference_result_dir"
    ]  # where inference results are saved
    reconstructed_images_dir = config["dataset"][
        "reconstructed_images_dir"
    ]  # output folder for final images

    # Use original image dimensions from config
    original_width = config["dataset"]["original_image_width"]
    original_height = config["dataset"]["original_image_height"]
    original_size = (original_width, original_height)

    # Ensure output directory exists
    if not os.path.exists(reconstructed_images_dir):
        os.makedirs(reconstructed_images_dir)

    # Create Spark session
    spark = SparkSession.builder.appName("ReconstructImages").getOrCreate()

    # Read the inference result DataFrame.
    # Expected schema: image_id, row_index, col_index, patch_value (DenseVector)
    df = spark.read.parquet(inference_result_dir)

    # Group by image_id and collect all patch tuples.
    grouped_df = df.groupBy("image_id").agg(
        collect_list(struct("row_index", "col_index", "patch_value")).alias("patches")
    )

    def reconstruct_image(image_id, patches):
        """
        Given an image_id and its list of patches (each a dict with row_index, col_index, patch_value),
        reconstruct the full image using the original_size from config and your patches_to_image function.
        The reconstructed image is saved to disk using the image_id, and the output path is returned.
        """
        # Convert each patch dict into a tuple (row_index, col_index, patch_vector)
        patch_tuples = [
            (p["row_index"], p["col_index"], p["patch_value"]) for p in patches
        ]

        # Reconstruct the image using the existing function.
        # patches_to_image expects original_size as (width, height) in pixels.
        reconstructed_img = patches_to_image(
            patch_tuples, original_size, patch_size, upscale_factor
        )

        # Save the reconstructed image using the image_id.
        output_path = os.path.join(
            reconstructed_images_dir, f"{image_id}_reconstructed.png"
        )
        reconstructed_img.save(output_path)
        return output_path

    # Register the UDF. It takes image_id and patches and returns the output file path.
    reconstruct_udf = udf(reconstruct_image, StringType())

    # Apply the UDF to reconstruct each image.
    result_df = grouped_df.withColumn(
        "output_path", reconstruct_udf("image_id", "patches")
    )

    # Optionally, collect and print the output paths.
    for row in result_df.select("image_id", "output_path").collect():
        print(f"Image {row['image_id']} reconstructed at {row['output_path']}")

    spark.stop()
