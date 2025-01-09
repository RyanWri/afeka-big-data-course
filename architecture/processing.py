from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import Glue
from diagrams.aws.ml import Sagemaker
from diagrams.aws.storage import S3

# Define the architecture diagram for the Processing Cluster
with Diagram("architecture/images/Processing Cluster", show=False):
    # Input bucket for raw data
    s3_input = S3("S3 Input Bucket (Raw Images)")

    # Spark ETL pipeline
    with Cluster("Spark ETL Pipeline"):
        split_images = Glue("Split Images to Patches")
        inference = Sagemaker("Model Inference (Super-Resolution)")
        reconstruct_columns = Glue("Reconstruct Columns")
        reconstruct_rows = Glue("Reconstruct Rows")

    # Output bucket for processed data
    s3_output = S3("S3 Output Bucket (Processed Images)")

    # Data Flow
    s3_input >> Edge(label="Fetch Batch") >> split_images
    split_images >> Edge(label="Patches") >> inference
    inference >> Edge(label="Processed Patches") >> reconstruct_columns
    reconstruct_columns >> Edge(label="Reconstructed Columns") >> reconstruct_rows
    reconstruct_rows >> Edge(label="Final Images") >> s3_output
