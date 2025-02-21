import time
from src.ingestion.ImageKafkaConsumer import ImageKafkaConsumer
from src.ingestion.ImageKafkaProducer import ImageKafkaProducer

def run_kafka_threads():
    kafka_config_producer = {
        "bootstrap.servers": "localhost:9092",
        "client.id": "image-producer",
    }

    kafka_config_consumer = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "image-consumer-group",
        "auto.offset.reset": "earliest",
    }

    topic_name = "image-topic"

    # Start Producer Thread
    producer_thread = ImageKafkaProducer(kafka_config_producer, topic_name, produce_timer=30)
    producer_thread.start()

    # Start Consumer Thread
    consumer_thread = ImageKafkaConsumer(kafka_config_consumer, topic_name, consume_timer=60)
    consumer_thread.start()

