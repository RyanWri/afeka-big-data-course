import io
import time
import requests
import threading
from PIL import Image
from confluent_kafka import Producer


class ImageKafkaProducer(threading.Thread):
    def __init__(self, kafka_config, topic, image_size=128, produce_timer=5):
        super().__init__()
        self.producer = Producer(kafka_config)
        self.topic = topic
        self.running = True
        self.image_size = image_size
        self.produce_timer = produce_timer

    def fetch_image(self):
        """Fetch an image from the API and return its binary content."""
        try:
            response = requests.get(
                f"https://picsum.photos/{self.image_size}?grayscale", timeout=3
            )
            if response.status_code == 200:
                return response.content  # Return binary image data
            else:
                print(f"Failed to fetch image, status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error fetching image: {e}")
        return None

    def validate_image_shape(self, image_data):
        """Validate that the image meets the minimum size requirements."""
        try:
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size

            if width == self.image_size and height == self.image_size:
                print(f"Image shape is valid: {width}x{height}")
                return True
            else:
                print(f"Image shape is invalid ({width}x{height}), discarding...")
                return False
        except Exception as e:
            print(f"Error validating image: {e}")
            return False

    def delivery_report(self, err, msg):
        """Kafka delivery report callback."""
        if err:
            print(f"Message delivery failed: {err}")
        else:
            print(
                f"Image sent to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

    def produce_image(self):
        """Fetch, validate, and send an image to Kafka."""
        image_data = self.fetch_image()
        if image_data and self.validate_image_shape(image_data):
            self.producer.produce(
                self.topic, key="image", value=image_data, callback=self.delivery_report
            )
            self.producer.flush()  # Ensure delivery

    def run(self):
        """Thread execution loop: Fetch and send images periodically."""
        while self.running:
            self.produce_image()
            time.sleep(
                self.produce_timer
            )  # Fetch an image every few seconds (5 by default).

    def stop(self):
        """Stop the producer thread gracefully."""
        self.running = False
        self.producer.flush()
        print("Kafka Image Producer Stopped.")
