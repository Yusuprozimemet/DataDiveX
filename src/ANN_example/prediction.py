import numpy as np

class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def make_prediction(self, new_data):
        scaled_data = self.scaler.transform(new_data)
        prediction = self.model.predict(scaled_data)
        return prediction
