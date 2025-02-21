import numpy as np
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from PIL import Image
import os


def evaluate_image_quality(image_name):
    """
    Calculates PSNR and SSIM between the original high-resolution image
    and the super-resolved image.

    Args:
        image_name (str): The image name.

    Returns:
        dict: Dictionary containing PSNR and SSIM values.
    """
    cwd = os.getcwd()
    original_image_path = f"{cwd}/HR/{image_name}"
    super_resolved_image_path = f"{cwd}/SR/{image_name}"
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


def main(image_name):
    print(f"Evaluating metrics on image: {image_name}")
    result = evaluate_image_quality(image_name)
    for k, v in result.items():
        print(f"{k}: {v}")
