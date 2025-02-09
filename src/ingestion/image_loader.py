import requests
import cv2


def fetch_grayscale_image(base_url, height, width, save_path):
    url = f"{base_url}/{height}/{width}?grayscale"
    fetch_and_save_image(url, save_path)


def fetch_and_save_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved to {save_path}")
    except Exception:
        raise Exception(f"Failed to fetch image. Status code: {response.status_code}")


def to_low_resolution(hr_image_path, lr_image_path):
    # Read the input grayscale image
    image = cv2.imread(hr_image_path)
    # Convert to grayscale (if not already)
    input_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(input_grayscale, (64, 64), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(lr_image_path, resized_image)


if __name__ == "__main__":
    # this is the pipeline for the ingestion cluster, you will need to enhance it

    # const
    base_url = "https://picsum.photos"
    dataset_path = "/home/ran/datasets/spark-picsum-images"

    # get grayscale image (this will be the high resolution image)
    image_height, image_width = 128, 128
    image_id = "grayscale_002.jpg"
    hr_image_path = f"{dataset_path}/HR/{image_id}"
    fetch_grayscale_image(base_url, image_height, image_width, hr_image_path)

    # create low res image
    lr_image_path = f"{dataset_path}/LR/{image_id}"
    to_low_resolution(hr_image_path, lr_image_path)
