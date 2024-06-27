import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml

class DataProcessor:
    def __init__(self, config):
        self.filepath = config['data']['filepath']
        self.test_size = config['test_size']
        self.random_state = config['random_state']

    def load_data(self):
        df = pd.read_csv(self.filepath)
        return df

    def preprocess_data(self, df):
        X = df[['feature1', 'feature2']].values
        y = df['price'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        scaler = MinMaxScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler
