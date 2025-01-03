import requests


def fetch_and_save_image(url, save_path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved to {save_path}")
    except Exception:
        raise Exception(f"Failed to fetch image. Status code: {response.status_code}")


if __name__ == "__main__":
    image_size = 64
    url = f"https://picsum.photos/{image_size}"
    dataset_path = "/home/ran/datasets/spark-picsum-images/001.jpg"
    fetch_and_save_image(url, dataset_path)
