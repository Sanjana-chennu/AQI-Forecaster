import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(file_path):
    """
    Loads the dataset from a CSV file, cleans it, and engineers time-based features.
    """
    df = pd.read_csv(file_path)
    df['pollution'] = df['pollution'].ffill()
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Converted the non-numerical date index into useful numerical features
    # that the model can be trained on
    # previously model didnt know the date context.
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    # -----------------------------------------

    # One-hot encode wind direction after adding other features
    df = pd.get_dummies(df, columns=['wnd_dir'], prefix='wind')
    
    print("Data loaded, cleaned, and features engineered successfully.")
    return df


def scale_and_create_sequences(df, train_split_ratio=0.8, window_size=48):
    # These are the features we want our model to predict for the next time step.
    target_columns = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
    
    # All columns will be used as input features
    all_columns = df.columns
    
    # Convert DataFrame to NumPy array
    data = df.values

    #Scaling in 0-1 range.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    #Splitting 80-20 ratio
    training_data_len = int(len(scaled_data) * train_split_ratio)
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len:]
    
    # Find the indices of our target columns in the scaled data
    target_indices=[df.columns.get_loc(col) for col in target_columns]

    #Sliding window to create sequences

    def create_dataset(dataset, window_size):
        X, y = [], []
        for i in range(len(dataset) - window_size):
            X.append(dataset[i:(i + window_size), :])
            
            y.append(dataset[i + window_size, target_indices])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)

    print("Training and testing sequences created for MULTI-OUTPUT model.")
    print(f"X_train shape: {X_train.shape}") 
    print(f"y_train shape: {y_train.shape}") 

    # Return the main scaler, as it's needed for inverse transforming everything
    return X_train, y_train, X_test, y_test, scaler, df.columns, target_columns