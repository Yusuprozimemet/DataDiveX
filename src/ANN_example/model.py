import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import yaml

class ANNModel:
    def __init__(self, config):
        self.model = Sequential()
        input_shape = config['model']['input_shape']
        for layer in config['model']['layers']:
            units = layer['units']
            activation = layer['activation']
            if activation:
                self.model.add(Dense(units, activation=activation, input_shape=(input_shape,) if input_shape else ()))
            else:
                self.model.add(Dense(units))
            input_shape = None  # Only specify input shape for the first layer
        
        optimizer = config['model']['optimizer']
        loss = config['model']['loss']
        metrics = config['model'].get('metrics', [])

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, X_train, y_train, epochs):
        self.history = self.model.fit(X_train, y_train, epochs=epochs)
        return self.history

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    @staticmethod
    def load(filepath):
        from tensorflow.keras.models import load_model
        loaded_model = ANNModel.__new__(ANNModel)  # Create an empty instance of ANNModel
        loaded_model.model = load_model(filepath)
        return loaded_model
