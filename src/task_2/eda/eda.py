import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import nest_asyncio
import os

# Allow nested use of asyncio.run()
nest_asyncio.apply()


async def load_and_process_chunk(file_path: str) -> pd.DataFrame:
    # Load a single chunk
    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    # Combine Date and Time into a single datetime column and set as index
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("dt")
    # Drop the original Date and Time columns
    df = df.drop(columns=["Date", "Time"])
    # Drop rows with missing values
    df.replace("?", np.nan, inplace=True)

    # set all columns as float
    for col in df.columns:
        df[col] = df[col].astype("float64")

    # Handle missing values by filling them with the median of each column
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    # Step 5: Identify and Handle Outliers
    # Detect and handle outliers by capping at the 99th percentile
    upper_limit = df["Global_active_power"].quantile(0.99)
    df["Global_active_power"] = df["Global_active_power"].clip(upper=upper_limit)

    return df


def plot_dataframe_stats(df: pd.DataFrame):
    # Display basic info about the DataFrame
    print(df.info())
    print(df.head())


# Step 2: Visualize Time Series Trends
def visualize_time_series_trends(df: pd.DataFrame):
    # Plot Global_active_power over time
    plt.figure(figsize=(12, 6))
    plt.plot(df["Global_active_power"], label="Global Active Power")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power (kilowatts)")
    plt.title("Global Active Power over Time")
    plt.legend()
    plt.show()


# Step 3: Check for Seasonality and Cyclical Patterns
def check_seasonality_and_cyclical_patterns(df: pd.DataFrame):
    # Decompose the time series
    decomposition = seasonal_decompose(
        df["Global_active_power"].dropna(), model="additive", period=24 * 60
    )

    # Plot decomposition results
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(decomposition.observed, label="Observed")
    plt.legend(loc="upper right")
    plt.subplot(412)
    plt.plot(decomposition.trend, label="Trend")
    plt.legend(loc="upper right")
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label="Seasonal")
    plt.legend(loc="upper right")
    plt.subplot(414)
    plt.plot(decomposition.resid, label="Residual")
    plt.legend(loc="upper right")
    plt.show()


# Step 4: Analyze Distribution of Power Consumption
def analyze_distribution_of_power_consumption(df: pd.DataFrame):
    # Plot histogram
    plt.figure(figsize=(12, 6))
    df["Global_active_power"].hist(bins=50)
    plt.xlabel("Global Active Power (kilowatts)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Global Active Power")
    plt.show()

    # Plot boxplot
    plt.figure(figsize=(12, 6))
    df.boxplot(column="Global_active_power")
    plt.ylabel("Global Active Power (kilowatts)")
    plt.title("Boxplot of Global Active Power")
    plt.show()


async def process_data_parallel():
    rootdir = os.path.join(os.getcwd(), "data")

    # Traverse Data Directory and get paths to all chunk files
    file_names = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_names.append(os.path.join(subdir, file))

    # Load and process each chunk
    tasks = [load_and_process_chunk(file) for file in file_names]
    results = await asyncio.gather(*tasks)

    # Concatenate all chunks into a single DataFrame
    full_df = pd.concat(results)
    return full_df


# Run the asynchronous processing
full_df = asyncio.run(process_data_parallel())
