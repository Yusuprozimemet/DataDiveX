import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from utils import get_config

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
        X = df[self.independent_vars]
        y = df[self.dependent_var]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, scaler
