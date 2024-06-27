import pandas as pd
from utils import get_config

class DataIngestion:
    def __init__(self, config):
        self.filepath = config['data']['filepath']
        print(f"Filepath: {self.filepath}")  # Added for verification

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath)
            print(f"Data loaded successfully from {self.filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: File '{self.filepath}' not found.")
            return None
        except Exception as e:
            print(f"Error: Failed to load data from '{self.filepath}'. Error message: {str(e)}")
            return None

if __name__ == "__main__":
    config = get_config()  # Assuming get_config() function fetches your configuration
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data()
    
    if df is not None:
        print("Data loading process successful.")
        print(df.head())  # Example: Print first few rows of the loaded dataframe
    else:
        print("Data loading process failed. Check error messages above.")
