import time
from src.ingestion.ImageKafkaConsumer import ImageKafkaConsumer
from src.ingestion.ImageKafkaProducer import ImageKafkaProducer

if __name__ == "__main__":
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
    producer_thread = ImageKafkaProducer(kafka_config_producer, topic_name)
    producer_thread.start()

    # Start Consumer Thread
    consumer_thread = ImageKafkaConsumer(kafka_config_consumer, topic_name)
    consumer_thread.start()

    # Let them run for 20 seconds, then stop
    try:
        time.sleep(21)
    except KeyboardInterrupt:
        pass

    # Stop threads gracefully
    producer_thread.stop()
    producer_thread.join()

    consumer_thread.stop()
    consumer_thread.join()
