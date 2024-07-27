import pandas as pd
import os

HEADER_ROW = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def read_and_split_data():
    t = os.getcwd()
    df = pd.read_csv(
        "src/task_2/data/household_power_consumption.txt", sep=";", header=None
    )
    chunks = 10
    step = len(df) // chunks
    for i in range(0, len(df), step):
        df_i = df.iloc[i : i + step, :]
        df_i.columns = HEADER_ROW
        df_i.to_csv(f"src/task_2/data/household_power_consumption_{i}.csv", index=False)


def read_household_power_consumption():
    flist = []
    rootdir = os.path.join(os.getcwd(), "data")

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            df = pd.read_csv(os.path.join(subdir, file), low_memory=False)
            flist.append(df)

    df_out = pd.concat(flist, axis=0, ignore_index=False)
    return df_out


if __name__ == "__main__":
    read_and_split_data()
    # df = read_household_power_consumption()
    # print(df.shape)
    # print(df.head())
