import io
import os
import time
import threading
from PIL import Image
from confluent_kafka import Consumer


class ImageKafkaConsumer(threading.Thread):
    def __init__(self, kafka_config, topic, consume_timer=20, high_res_dir="HR", low_res_dir="LR"):
        super().__init__()
        self.consumer = Consumer(kafka_config)
        self.topic = topic
        self.running = True
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.consume_timer = consume_timer

        # Create directories if they don't exist
        os.makedirs(self.high_res_dir, exist_ok=True)
        os.makedirs(self.low_res_dir, exist_ok=True)

        self.consumer.subscribe([self.topic])

    def validate_downscaled_image(self, original_image, downscaled_image):
        """Ensure that the downscaled image is exactly half the size of the original."""
        original_width, original_height = original_image.size
        downscaled_width, downscaled_height = downscaled_image.size

        expected_width, expected_height = original_width // 2, original_height // 2

        if (downscaled_width, downscaled_height) == (expected_width, expected_height):
            print(
                f"Valid shape after downscaling: {original_width}x{original_height} → {downscaled_width}x{downscaled_height}"
            )
            return True
        else:
            print(
                f"Invalid shape after downscaling! Expected {expected_width}x{expected_height} but got {downscaled_width}x{downscaled_height}"
            )
            return False

    def save_images(self, image_data, image_index):
        """Save the original and downscaled images to their respective directories."""

        # Convert binary data to PIL image
        image = Image.open(io.BytesIO(image_data))

        # Downscale image by half
        width, height = image.size
        image_resized = image.resize((width // 2, height // 2), Image.BICUBIC)

        # Validate that the downscaled image is exactly half the original
        if not self.validate_downscaled_image(image, image_resized):
            print("Skipping save due to incorrect downscaling!")
            return  # Skip saving invalid images

        # Save original image
        high_res_path = os.path.join(self.high_res_dir, f"image_{image_index}.jpg")
        image.save(high_res_path, "JPEG")
        print(f"Saved high-resolution image: {high_res_path}")

        # Save downscaled image
        low_res_path = os.path.join(self.low_res_dir, f"image_{image_index}.jpg")
        image_resized.save(low_res_path, "JPEG")
        print(f"Saved low-resolution image: {low_res_path}")

    def run(self):
        """Thread execution loop: Poll for Kafka messages and save images."""
        image_index = 0
        while self.running:
            while True:
                msg = self.consumer.poll(1.0)  # Poll for messages
                if msg is None:
                    break
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue

                image_data = msg.value()
                # Save images (original + downscaled)
                self.save_images(image_data, image_index)
                image_index += 1
            time.sleep(
                self.consume_timer
            )  # Consume images from Kafka every few seconds (20 by default).

    def stop(self):
        """Stop the consumer thread gracefully."""
        self.running = False
        self.consumer.close()
        print("Kafka Image Consumer Stopped.")
