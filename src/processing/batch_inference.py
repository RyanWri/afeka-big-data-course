import os
import torch
import yaml
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector, VectorUDT
from src.processing.fsrcnn import load_fsrcnn_model


def super_resolve_patch(patch_vector, model_broadcast, patch_size):
    """
    Applies FSRCNN super-resolution to a single grayscale patch vector.
    The patch_vector is assumed to be a DenseVector, which is reshaped
    into (patch_size, patch_size), normalized, and then fed to the model.
    """
    model = model_broadcast.value  # Access the broadcasted model
    patch_array = (
        np.array(patch_vector).reshape((patch_size, patch_size)).astype(np.float32)
    )
    patch_tensor = (
        torch.from_numpy(patch_array).unsqueeze(0).unsqueeze(0)
    )  # Add batch and channel dims

    # Perform inference
    with torch.no_grad():
        sr_patch_tensor = model(patch_tensor)

    sr_patch_array = sr_patch_tensor.squeeze().numpy() * 255.0  # Rescale to [0, 255]
    return DenseVector(sr_patch_array.flatten())


if __name__ == "__main__":
    # Load configuration
    with open("src/processing/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Extract necessary parameters from config
    patch_size = config["processing"]["patch_size"]
    upscale_factor = config["processing"]["upscale_factor"]
    model_path = os.path.join(os.getcwd(), config["processing"]["model_path"])

    # Define paths for input patches and inference results.
    # These are hardcoded now, but can be parameterized later.
    patches_input_path = config["dataset"][
        "patches_dir"
    ]  # e.g., where your first job saved patches
    patches_output_path = config["dataset"][
        "inference_result_dir"
    ]  # output directory for inference results

    if not os.path.exists(patches_output_path):
        os.makedirs(patches_output_path)

    # Create Spark Session
    spark = SparkSession.builder.appName("BatchInferenceOnPatches").getOrCreate()
    sc = spark.sparkContext

    # Load and broadcast the FSRCNN model
    model = load_fsrcnn_model(upscale_factor, model_path)
    model_broadcast = sc.broadcast(model)

    # Define the super resolution UDF inside main so that it captures the broadcast variables.
    super_resolve_udf = udf(
        lambda patch_vector: super_resolve_patch(
            patch_vector, model_broadcast, patch_size
        ),
        VectorUDT(),
    )

    # Read the patches DataFrame from disk (Parquet format)
    # The DataFrame is expected to have the schema: (image_id, row_index, col_index, patch_value)
    patch_df = spark.read.parquet(patches_input_path)

    # Apply the FSRCNN model to each patch.
    # This replaces the original patch_value with the inferred (super-resolved) value.
    result_df = patch_df.withColumn(
        "patch_value", super_resolve_udf(patch_df["patch_value"])
    )

    # Save the resulting DataFrame back to disk using Parquet (optimal for storage and downstream processing)
    result_df.write.mode("overwrite").parquet(patches_output_path)
    print(f"Inference results saved at {patches_output_path}")

    spark.stop()
