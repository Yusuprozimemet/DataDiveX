import pandas as pd
import numpy as np


def prepare_input_data(input_data, scaler):
    # Ensure input_data is numpy array and reshape if necessary
    if isinstance(input_data, list):
        input_data = np.array(input_data).reshape(1, -1)
    elif isinstance(input_data, np.ndarray):
        input_data = input_data.reshape(1, -1)
    else:
        raise ValueError("Unsupported input_data type. Must be list or numpy array.")

    # Scale the input data using the provided scaler
    input_data_scaled = scaler.transform(input_data)
    
    return input_data_scaled

class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def make_prediction(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        prediction = self.model.predict(new_data_scaled)
        return prediction
