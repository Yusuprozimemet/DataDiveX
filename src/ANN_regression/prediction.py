import numpy as np

class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def make_prediction(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        prediction = self.model.predict(new_data_scaled)
        return prediction
