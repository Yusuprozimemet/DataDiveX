import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import get_config, save_model

class DataPreprocessing:
    def __init__(self, config):
        self.independent_vars = config['data']['independent_vars']
        self.dependent_var = config['data']['dependent_var']
        self.test_size = 0.3
        self.random_state = 101
        self.scaler = StandardScaler()

    def preprocess_data(self, df):
        # Ensure only numeric columns are used
        df = df.select_dtypes(include=['float64', 'int64'])
        
        # Check if all independent_vars are present in df
        missing_vars = [var for var in self.independent_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"Columns {missing_vars} are missing in the dataframe.")

        X = df[self.independent_vars]
        y = df[self.dependent_var]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    config = get_config()  # Assuming config.yaml is correctly updated with house_age
    print(config)  # Debug print to check the content of the config

    # Load your data (assuming df is your DataFrame)
    df = pd.read_csv('artifacts/kc_house_data_updated.csv')  # Load from updated CSV path

    # Initialize data preprocessing with config
    data_preprocessor = DataPreprocessing(config)
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessor.preprocess_data(df)
