import pandas as pd
import numpy as np


def read_single_file(file_path: str) -> pd.DataFrame:
    # Load a single chunk
    chunk_file = "C:/Afeka/Afeka_DL_course_labs/src/task_2/data/household_power_consumption_0.csv"
    df = pd.read_csv(
        chunk_file,
        low_memory=False,
    )

    # Combine Date and Time into a single datetime column and set as index
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.set_index("dt", inplace=True)
    # Drop the original Date and Time columns
    df.drop(columns=["Date", "Time"], inplace=True)
    # Drop rows with missing values
    df.replace("?", np.nan, inplace=True)

    # set all columns as float
    for col in df.columns:
        df[col] = df[col].astype("float64")

    # Handle missing values by filling them with the median of each column
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].median(), inplace=True)

    # Step 5: Identify and Handle Outliers
    # Detect and handle outliers by capping at the 99th percentile
    upper_limit = df["Global_active_power"].quantile(0.99)
    df["Global_active_power"] = df["Global_active_power"].clip(upper=upper_limit)

    return df


# Display basic info about the DataFrame
print(df.info())
print(df.head())


# Step 2: Visualize Time Series Trends
import matplotlib.pyplot as plt

# Plot Global_active_power over time
plt.figure(figsize=(12, 6))
plt.plot(df["Global_active_power"], label="Global Active Power")
plt.xlabel("Time")
plt.ylabel("Global Active Power (kilowatts)")
plt.title("Global Active Power over Time")
plt.legend()
plt.show()

# Step 3: Check for Seasonality and Cyclical Patterns
from statsmodels.tsa.seasonal import seasonal_decompose

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
