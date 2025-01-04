from PIL import Image
import numpy as np


def super_resolve_patch(patch_vector, patch_size):
    """
    Applies bicubic interpolation to simulate super-resolution.
    """
    patch_array = np.array(patch_vector, dtype=np.uint8).reshape(
        (patch_size, patch_size, 3)
    )
    patch_image = Image.fromarray(patch_array)
    sr_patch_image = patch_image.resize((patch_size * 2, patch_size * 2), Image.BICUBIC)
    return np.array(sr_patch_image).flatten().tolist()
