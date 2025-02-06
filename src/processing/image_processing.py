import numpy as np
from PIL import Image
from pyspark.ml.linalg import DenseVector


def image_to_patches(image_path, patch_size):
    """
    Splits a grayscale image into patches and converts them into flattened vectors.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    width, height = image.size
    patches = []

    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            box = (col, row, col + patch_size, row + patch_size)
            patch = image.crop(box)
            patch_array = np.array(patch).flatten()
            patches.append(
                (row // patch_size, col // patch_size, DenseVector(patch_array))
            )

    return patches


def patches_to_image(patches, original_size, patch_size, upscale_factor):
    """
    Reconstructs the final grayscale image from super-resolved patches.
    """
    original_width, original_height = original_size
    sr_image = Image.new(
        "L", (original_width * upscale_factor, original_height * upscale_factor)
    )

    for row_index, col_index, patch_vector in patches:
        patch_array = np.array(patch_vector).reshape(
            (patch_size * upscale_factor, patch_size * upscale_factor)
        )
        patch_image = Image.fromarray(patch_array.astype(np.uint8), mode="L")
        x, y = (
            col_index * patch_size * upscale_factor,
            row_index * patch_size * upscale_factor,
        )
        sr_image.paste(patch_image, (x, y))

    return sr_image
