from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
from src.processing.fsrcnn_model import run_inference


def initialize_spark(app_name="ImageSuperResolution"):
    """
    Initializes and returns a Spark session.
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark


def get_super_resolve_udf(model_broadcast, patch_size):
    """
    Returns a UDF that applies FSRCNN super-resolution.
    Ensures Spark workers can access the broadcasted model.
    """

    @udf(VectorUDT())
    def super_resolve_udf(patch_vector):
        return run_inference(patch_vector, model_broadcast, patch_size)

    return super_resolve_udf


def process_image_with_spark(spark, patches, model_broadcast, patch_size):
    """
    Runs the Spark pipeline to apply super-resolution.
    """
    # Create DataFrame from patches
    patch_df = spark.createDataFrame(
        patches, schema=["row_index", "col_index", "vector"]
    )

    # Register UDF using the correct model reference
    super_resolve_udf = get_super_resolve_udf(model_broadcast, patch_size)

    # Apply FSRCNN model to patches
    sr_patch_df = patch_df.withColumn(
        "sr_vector", super_resolve_udf(patch_df["vector"])
    )

    return sr_patch_df
