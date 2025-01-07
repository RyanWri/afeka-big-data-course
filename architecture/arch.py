from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.queue import Kafka
from diagrams.onprem.analytics import Spark
from diagrams.aws.storage import S3
from diagrams.aws.ml import SagemakerModel
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet

# Define the architecture diagram using Python Diagrams
with Diagram("architecture/data_flow", show=False):
    user = User("User")

    with Cluster("Data Ingestion Cluster"):
        kafka_producer = Kafka("Kafka Producer\n(API Requests)")
        kafka_consumer = Kafka("Kafka Consumer\n(Image Downloader)")
        api = Internet("Picsum API")
        s3_input = S3("S3 Input Bucket\n(Raw Partitions)")

        # Data Flow for Ingestion
        user >> kafka_producer
        kafka_producer >> Edge(label="API Calls") >> api >> kafka_consumer
        kafka_consumer >> Edge(label="Download Images") >> s3_input

    with Cluster("Data Processing Cluster"):
        spark = Spark("Apache Spark\nBatch Processor")
        ml_model = SagemakerModel("ML Model\n(Super-Resolution)")
        s3_output = S3("S3 Output Bucket\n(Processed Partitions)")
        kafka_notification = Kafka("Notification Consumer\n(User Notifier)")

        # Data Flow for Processing
        s3_input >> Edge(label="Fetch Partition") >> spark
        spark >> Edge(label="Load Model") >> ml_model
        spark >> Edge(label="Run Batch ETL") >> spark
        spark >> Edge(label="Reconstruct & Save") >> s3_output
        s3_output >> Edge(label="Final Partition") >> kafka_notification >> user
