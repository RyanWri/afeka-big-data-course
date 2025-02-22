from src.processing.mass_split import main as msmain
from src.processing.batch_inference import main as bmain
from src.processing.mass_reconstruct import main as mrmain


def main(spark, sc, sqlContext):
    """
    PIPELINE STEPS:
    1. Split images into patches
    2. Run inference on patches
    3. Reconstruct images
    """
    msmain(sqlContext)
    bmain(sc, spark)
    mrmain(spark)

    print("All steps completed successfully.")
