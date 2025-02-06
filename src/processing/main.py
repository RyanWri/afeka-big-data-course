import os
import yaml
from src.processing.fsrcnn_model import broadcast_model
from src.processing.image_processing import image_to_patches, patches_to_image
from src.processing.spark_processing import initialize_spark, process_image_with_spark
from PIL import Image


if __name__ == "__main__":
    with open("src/processing/config.yaml", "r") as file:
        prime_service = yaml.safe_load(file)

    # extract values from config
    local_image_path = os.path.join(
        prime_service["dataset"]["low_resolution_dir"],
        prime_service["paths"]["original_image"],
    )
    patch_size = prime_service["processing"]["patch_size"]
    upscale_factor = prime_service["processing"]["upscale_factor"]
    model_path = prime_service["processing"]["model_path"]

    # Initialize Spark
    spark = initialize_spark()
    sc = spark.sparkContext

    # Broadcast FSRCNN Model
    model_broadcast = broadcast_model(sc, upscale_factor, model_path)

    # Split Image into Patches
    patches = image_to_patches(local_image_path, patch_size)

    # Apply Super-Resolution using Spark
    sr_patch_df = process_image_with_spark(spark, patches, model_broadcast, patch_size)

    # Collect and Reconstruct Image
    sr_patches = (
        sr_patch_df.select("row_index", "col_index", "sr_vector")
        .rdd.map(tuple)
        .collect()
    )
    original_image = Image.open(local_image_path)
    sr_image = patches_to_image(
        sr_patches, original_image.size, patch_size, upscale_factor
    )

    # Save Final Image
    output_image_path = os.path.join(
        prime_service["dataset"]["super_resolved_dir"],
        prime_service["paths"]["output_image"],
    )
    sr_image.save(output_image_path)
    print(f"Super-resolved image saved at {output_image_path}.")
