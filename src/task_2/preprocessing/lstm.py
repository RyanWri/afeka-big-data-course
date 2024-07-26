from sklearn.model_selection import train_test_split


def split_data_append_lagged_features(full_df, scaler) -> tuple:
    # Prepare the data with lagged features
    full_df["lag_1"] = full_df["Global_active_power"].shift(1)
    full_df["lag_2"] = full_df["Global_active_power"].shift(2)
    full_df["lag_3"] = full_df["Global_active_power"].shift(3)

    # Drop any rows with NaN values created by the shift operation
    full_df.dropna(inplace=True)

    # Define the target variable and features
    features = ["lag_1", "lag_2", "lag_3"]
    X = full_df[features].values
    y = full_df["Global_active_power"].values

    # Scale the data
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Reshape the input data to 3D for LSTM [samples, time steps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, y_train, y_test
