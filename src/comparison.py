import numpy as np
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from PIL import Image


def evaluate_image_quality(original_image_path, super_resolved_image_path):
    """
    Calculates PSNR and SSIM between the original high-resolution image
    and the super-resolved image.

    Args:
        original_image_path (str): Path to the original high-resolution image.
        super_resolved_image_path (str): Path to the super-resolved image.

    Returns:
        dict: Dictionary containing PSNR and SSIM values.
    """
    # Load images as grayscale arrays
    original_image = np.array(
        Image.open(original_image_path).convert("L"), dtype=np.float32
    )
    super_resolved_image = np.array(
        Image.open(super_resolved_image_path).convert("L"), dtype=np.float32
    )

    # Ensure the dimensions of both images match
    if original_image.shape != super_resolved_image.shape:
        raise ValueError(
            "Original and super-resolved images must have the same dimensions for PSNR/SSIM calculation."
        )

    # Calculate PSNR
    psnr = calculate_psnr(original_image, super_resolved_image, data_range=255)

    # Calculate SSIM
    ssim = calculate_ssim(original_image, super_resolved_image, data_range=255)

    return {"PSNR": psnr, "SSIM": ssim}


if __name__ == "__main__":
    image_id = "grayscale_002.jpg"
    original_image_path = f"/home/ran/datasets/spark-picsum-images//HR/{image_id}"
    super_resolved_image_path = f"/home/ran/datasets/spark-picsum-images/SR/{image_id}"
    result = evaluate_image_quality(original_image_path, super_resolved_image_path)
    for k, v in result.items():
        print(f"{k}: {v}")
