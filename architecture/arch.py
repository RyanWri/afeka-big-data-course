from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.queue import Kafka
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet
from diagrams.aws.storage import S3
from diagrams.aws.analytics import Glue
from diagrams.aws.ml import Sagemaker
from diagrams.onprem.container import Docker

# Define the combined architecture diagram
with Diagram("architecture/images/Complete Architecture", show=False, direction="LR"):
    # Ingestion Cluster
    with Cluster("Ingestion Cluster"):
        user = User("User")
        kafka = Kafka("Kafka (Topics)")

        with Cluster("Python Services"):
            producer_service = Docker("Producer Service")
            consumer_service = Docker("Consumer Service")

        api = Internet("Picsum API")
        s3_input = S3("S3 Input Bucket (Raw Images)")

        # Ingestion Data Flow
        user >> producer_service >> kafka
        kafka >> consumer_service >> Edge(label="API Calls") >> api
        api >> Edge(label="Image Data") >> consumer_service >> s3_input

    # Processing Cluster
    with Cluster("Processing Cluster"):
        spark = Glue("Spark ETL Pipeline")
        split_images = Glue("Split Images to Patches")
        inference = Sagemaker("Model Inference (Super-Resolution)")
        reconstruct_columns = Glue("Reconstruct Columns")
        reconstruct_rows = Glue("Reconstruct Rows")
        s3_output = S3("S3 Output Bucket (Processed Images)")

        # Processing Data Flow
        s3_input >> Edge(label="Fetch Batch") >> split_images
        split_images >> Edge(label="Patches") >> inference
        inference >> Edge(label="Processed Patches") >> reconstruct_columns
        reconstruct_columns >> Edge(label="Reconstructed Columns") >> reconstruct_rows
        reconstruct_rows >> Edge(label="Final Images") >> s3_output

    # Notification to User
    kafka_notification = Kafka("Notification Consumer")
    s3_output >> kafka_notification >> user
