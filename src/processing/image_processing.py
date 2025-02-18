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
