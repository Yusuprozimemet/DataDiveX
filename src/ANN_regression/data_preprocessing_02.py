import pandas as pd
import pickle
import os
from data_preprocessing import DataPreprocessing
from data_ingestion import DataIngestion
from utils import get_config, save_model

if __name__ == "__main__":
    config = get_config()

    # Load data from the updated CSV file
    updated_csv_path = 'artifacts/kc_house_data_updated.csv'  # Adjust this path if necessary
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data(updated_csv_path)
    
    if df is not None:
        print("Data loaded successfully.")
    else:
        print("Error loading data. Check previous messages for details.")
        exit(1)

    # Data Preprocessing
    data_preprocessing = DataPreprocessing(config)
    X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessing.preprocess_data(df)
    
    print("Data preprocessing completed successfully.")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Save X_train and X_test to CSV
    artifacts_dir = 'artifacts'
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    X_train_df = pd.DataFrame(X_train_scaled, columns=data_preprocessing.independent_vars)
    X_test_df = pd.DataFrame(X_test_scaled, columns=data_preprocessing.independent_vars)

    X_train_df.to_csv(os.path.join(artifacts_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(artifacts_dir, 'X_test.csv'), index=False)

    # Save y_train and y_test to CSV
    pd.DataFrame(y_train, columns=['price']).to_csv(os.path.join(artifacts_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test, columns=['price']).to_csv(os.path.join(artifacts_dir, 'y_test.csv'), index=False)

    # Save the scaler object
    scaler_filepath = 'artifacts/scaler.pkl'
    with open(scaler_filepath, 'wb') as file:
        pickle.dump(data_preprocessing.scaler, file)
    
    print(f"Scaler saved as {scaler_filepath}")

    # Select and Train Model (Continuing from here)
    # ...

    # Evaluate Model
    # ...

    # Make Predictions
    # ...

    # Save Model
    # ...
