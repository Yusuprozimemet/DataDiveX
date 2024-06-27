import pandas as pd
from utils import get_config

class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.filepath = config['data']['filepath']
        print(f"Filepath: {self.filepath}")  # Added for verification

    def load_data(self, csv_path=None):
        if csv_path is None:
            filepath = self.filepath
        else:
            filepath = csv_path
        
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully from {filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None
        except Exception as e:
            print(f"Error: Failed to load data from '{filepath}'. Error message: {str(e)}")
            return None

if __name__ == "__main__":
    config = get_config()  # Assuming get_config() function fetches your configuration
    data_ingestion = DataIngestion(config)
    
    # Load data from the updated CSV file
    updated_csv_path = 'artifacts/kc_house_data_updated.csv'  # Adjust this path as necessary
    df = data_ingestion.load_data(updated_csv_path)
    
    if df is not None:
        print("Data loading process successful.")
        print(df.head())  # Example: Print first few rows of the loaded dataframe
    else:
        print("Data loading process failed. Check error messages above.")
