from PIL import Image
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType
from pyspark.sql.functions import udf, col

# Initialize Spark Session
spark = SparkSession.builder.appName("ImageSuperResolution").getOrCreate()


# Step 2: Split Image into Patches
def image_to_patches(image_path, patch_size):
    """
    Load an image and split it into patches.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    patches = []
    indices = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)
            patches.append(
                np.array(patch).flatten()
            )  # Flatten the patch to a 1D vector
            indices.append(
                (x // patch_size, y // patch_size)
            )  # Store patch index (row, column)

    return patches, indices


# Step 3: Store Patches and Indices in Spark DataFrame
def create_patch_dataframe(patches, indices):
    """
    Create a Spark DataFrame from patches and indices.
    """
    schema = StructType(
        [
            StructField("index", ArrayType(IntegerType()), True),
            StructField("vector", ArrayType(IntegerType()), True),
        ]
    )
    data = [(indices[i], patches[i].tolist()) for i in range(len(patches))]
    return spark.createDataFrame(data, schema)


# Step 4: Define Super-Resolution Function
def super_resolve_patch(patch_vector, patch_size):
    """
    Simulate a super-resolution transformation by upscaling the patch.
    Here we use bicubic interpolation to double the size.
    """
    patch_array = np.array(patch_vector, dtype=np.uint8).reshape(
        (patch_size, patch_size, 3)
    )
    patch_image = Image.fromarray(patch_array)
    sr_patch_image = patch_image.resize((patch_size * 2, patch_size * 2), Image.BICUBIC)
    return np.array(sr_patch_image).flatten().tolist()  # Return as flattened vector


# Register UDF for Spark
@udf(ArrayType(IntegerType()))
def super_resolve_udf(patch_vector):
    return super_resolve_patch(patch_vector, patch_size)


# Step 5: Reconstruct Image
def patches_to_image(patches, indices, original_image_size, patch_size):
    """
    Reconstruct the full image from patches and their indices.
    """
    original_width, original_height = original_image_size
    sr_image = Image.new(
        "RGB", (original_width * 2, original_height * 2)
    )  # Double the size

    for patch, (row, col) in zip(patches, indices):
        patch_array = np.array(patch, dtype=np.uint8).reshape(
            (patch_size * 2, patch_size * 2, 3)
        )
        patch_image = Image.fromarray(patch_array)
        x, y = col * patch_size * 2, row * patch_size * 2
        sr_image.paste(patch_image, (x, y))

    return sr_image


# Main Execution
if __name__ == "__main__":
    # Parameters
    local_image_path = "/home/ran/datasets/spark-picsum-images/001.jpg"
    patch_size = 16  # Size of each patch

    # Step 2: Split the image into patches
    patches, indices = image_to_patches(local_image_path, patch_size)

    # Step 3: Create Spark DataFrame
    patch_df = create_patch_dataframe(patches, indices)
    # patch_df.show(truncate=False)  # Display the DataFrame content

    # Optionally save DataFrame to disk (Parquet/CSV) for future use
    # patch_df.write.mode('overwrite').parquet("patches.parquet")

    print("Patches stored in Spark DataFrame.")

    # Step 4: Apply Super Resolution
    sr_patch_df = patch_df.withColumn("sr_vector", super_resolve_udf(col("vector")))

    # Step 5: Collect Transformed Patches
    sr_patches = sr_patch_df.select("sr_vector").rdd.map(lambda row: row[0]).collect()
    patch_indices = patch_df.select("index").rdd.map(lambda row: row[0]).collect()

    # Step 6: Reconstruct the Super-Resolved Image
    original_image = Image.open(local_image_path)
    sr_image = patches_to_image(
        sr_patches, patch_indices, original_image.size, patch_size
    )

    # Save the final super-resolved image
    sr_image.save("super_resolved_image.jpg")
    print("Super-resolved image saved as 'super_resolved_image.jpg'.")
