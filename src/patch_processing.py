from PIL import Image
import numpy as np
from pyspark.ml.linalg import DenseVector


def image_to_patches(image_path, patch_size):
    """
    Splits an image into patches represented as tensors and flattens them into vectors.
    Returns patches as a list of (row_index, col_index, vector).
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    patches = []

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            box = (col, row, col + patch_size, row + patch_size)
            patch = image.crop(box)
            patch_tensor = np.array(patch).flatten()  # Convert to vector
            patches.append(
                (row // patch_size, col // patch_size, DenseVector(patch_tensor))
            )

    return patches


def patches_to_image(patches, original_image_size, patch_size):
    """
    Reconstructs the full image from patches and their indices.
    """
    original_width, original_height = original_image_size
    sr_image = Image.new(
        "RGB", (original_width * 2, original_height * 2)
    )  # Double size

    for row_index, col_index, patch_vector in patches:
        patch_array = (
            np.array(patch_vector)
            .reshape((patch_size * 2, patch_size * 2, 3))
            .astype(np.uint8)
        )
        patch_image = Image.fromarray(patch_array)
        x, y = col_index * patch_size * 2, row_index * patch_size * 2
        sr_image.paste(patch_image, (x, y))

    return sr_image
