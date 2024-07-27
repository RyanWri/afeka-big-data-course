import pandas as pd
import numpy as np


def resample_data_from_1min_to_2min(full_df):
    # Assuming 'full_df' has a datetime index 'dt'
    full_df["Datetime"] = full_df.index  # Ensure the datetime index is a column

    # Resample to 2-minute intervals by taking the mean
    resampled_df = full_df.resample('2T', on="Datetime").mean()

    # Drop rows with NaN values after resampling
    resampled_df.dropna(inplace=True)

    # Print the first few rows to check the result
    print(resampled_df.head())

    return resampled_df

def resample_data(full_df, res = '2min'):
    # Assuming 'full_df' has a datetime index 'dt'
    full_df["Datetime"] = full_df.index  # Ensure the datetime index is a column

    # Resample to 2-minute intervals by taking the mean
    resampled_df = full_df.resample(res, on="Datetime").mean()

    # Drop rows with NaN values after resampling
    resampled_df.dropna(inplace=True)

    # Print the first few rows to check the result
    print(resampled_df.head())

    return resampled_df