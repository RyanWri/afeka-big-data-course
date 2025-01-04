from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import udf
from super_resolution import super_resolve_patch


@udf(ArrayType(IntegerType()))
def super_resolve_udf(patch_vector, patch_size):
    """
    Spark UDF to apply super-resolution to a vectorized patch.
    """
    return super_resolve_patch(patch_vector, patch_size)
