from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.queue import Kafka
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet
from diagrams.onprem.container import Docker
from diagrams.aws.storage import S3

# Define the architecture diagram for the Ingestion Cluster
with Diagram("architecture/images/Ingestion Cluster", show=False):
    user = User("User")
    kafka = Kafka("Kafka (Topics)")

    with Cluster("Python Services"):
        producer_service = Docker("Producer Service")
        consumer_service = Docker("Consumer Service")

    api = Internet("Picsum API")
    storage = S3("Object Storage (Partitions)")

    # Connections
    user >> producer_service >> kafka
    kafka >> consumer_service >> Edge(label="API Calls") >> api
    api >> Edge(label="Image Data") >> consumer_service >> storage
