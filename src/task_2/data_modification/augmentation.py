import numpy as np
import pandas as pd


def add_noise(data, noise_factor=0.05):
    noise = np.random.randn(*data.shape) * noise_factor
    return data + noise


def scale_data(data, scale_factor=1.1):
    return data * scale_factor


def shift_data(data, shift_factor=10):
    return np.roll(data, shift_factor, axis=1)


def window_slicing(data, slice_factor=5):
    return np.array([np.roll(x, -slice_factor) for x in data])


def augment_sequence_data(data, y):
    """
    Target: implement data augmentation
    Hint: you can use add_noise, scale_data, shift_data, window_slicing
    """
    noise = add_noise(data)
    scaled = scale_data(data)
    shifted = shift_data(data)
    windowed = window_slicing(data)
    augmented_data = np.concatenate([data, noise, scaled, shifted, windowed], axis=0)
    augmented_labels = np.tile(y, 5)  # Repeat y to match the augmented X
    return augmented_data, augmented_labels


def reduce_data_randomly(X, y, reduction_factor=0.1):
    """
    Target: implement data reduction
    Hint: every file in data is exactly 10% of the data
    """
    total_samples = len(X)
    reduced_samples = int(total_samples * (1 - reduction_factor))

    indices = np.random.choice(
        range(total_samples), size=reduced_samples, replace=False
    )
    X_reduced = X[indices]
    y_reduced = y[indices]

    return X_reduced, y_reduced
