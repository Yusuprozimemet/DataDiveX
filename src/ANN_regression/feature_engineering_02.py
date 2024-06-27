import pandas as pd
import yaml
from flask import Config
from data_ingestion import DataIngestion
from utils import get_config, update_config_with_columns

class FeatureEngineering:
    def add_features(self, df):
        try:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['house_age'] = df['year'] - df['yr_built']
            df = df.drop(columns=['id', 'date', 'zipcode'])
            return df
        except KeyError as e:
            print(f"Error: Missing expected column {e}")
            return None

try:
    # Initialize Flask Config object
    config = Config(root_path='E:/DataAnalysisTool/src/ANN_regression')  # Adjust root_path as per your project structure
    
    # Initialize DataIngestion with config
    data_ingestion = DataIngestion(config)
    
    # Load data
    df = data_ingestion.load_data()
    
    if df is not None:
        # Perform feature engineering
        feature_engineering = FeatureEngineering()
        df = feature_engineering.add_features(df)
        
        if df is not None:
            # Update config with new columns from feature engineering
            update_config_with_columns(df, 'config.yaml')
            
            # Reload config with updated columns
            config = get_config('config.yaml')
            
            # Save the updated config
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f)
            print("Configuration updated successfully.")
        else:
            print("Feature engineering failed.")
    else:
        print("Data ingestion failed.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
