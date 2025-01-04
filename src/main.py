from patch_processing import image_to_patches, patches_to_image
from pyspark.sql import SparkSession
from PIL import Image
from pyspark.sql.functions import udf
import numpy as np
from pyspark.ml.linalg import DenseVector, VectorUDT


def super_resolve_patch(patch_vector):
    """
    Applies bicubic interpolation to simulate super-resolution.
    Works entirely with numpy tensors and vectors.
    """
    patch_size = 16  # Define patch size globally or pass as a parameter
    patch_array = np.array(patch_vector).reshape((patch_size, patch_size, 3))  # Tensor
    patch_image = Image.fromarray(patch_array.astype(np.uint8))  # Convert to Image
    sr_patch_image = patch_image.resize((patch_size * 2, patch_size * 2), Image.BICUBIC)
    sr_patch_array = np.array(sr_patch_image)  # Convert back to numpy array
    return DenseVector(sr_patch_array.flatten())  # Flatten and return as DenseVector


# Define the UDF without additional parameters
@udf(VectorUDT())
def super_resolve_udf(patch_vector):
    return super_resolve_patch(patch_vector)


if __name__ == "__main__":
    local_image_path = "/home/ran/datasets/spark-picsum-images/001.jpg"
    patch_size = 16

    # Spark Session
    spark = SparkSession.builder.appName("ImageSuperResolution").getOrCreate()

    # Step 2: Split into patches
    patches = image_to_patches(local_image_path, patch_size)

    # Step 3: Create DataFrame
    patch_df = spark.createDataFrame(
        patches, schema=["row_index", "col_index", "vector"]
    )

    # Show first row of the dataframe
    print("First row of the DataFrame:")
    first_row = patch_df.first()
    print(
        f"Row Index: {first_row['row_index']}, Column Index: {first_row['col_index']}, Vector (length): {len(first_row['vector'])}"
    )
    print(f"Sample Vector Data: {len(first_row['vector'])}")

    # Step 4: Apply Super Resolution
    sr_patch_df = patch_df.withColumn(
        "sr_vector", super_resolve_udf(patch_df["vector"])
    )

    # Step 5: Collect and Reconstruct
    sr_patches = (
        sr_patch_df.select("row_index", "col_index", "sr_vector")
        .rdd.map(tuple)
        .collect()
    )
    original_image = Image.open(local_image_path)
    sr_image = patches_to_image(sr_patches, original_image.size, patch_size)

    # Step 6: Save Final Image
    sr_image.save("super_resolved_image.jpg")
    print("Super-resolved image saved as 'super_resolved_image.jpg'.")
